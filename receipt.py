"""Local receipt recognition and conservative parsing; no receipt files are retained."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
import subprocess
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from storage import CATEGORIES
from receipt_items import dump_items, parse_items

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


class ReceiptError(ValueError):
    """A safe, user-facing receipt processing error."""


@dataclass
class Receipt:
    merchant: str = ''
    date: date | None = None
    amount: float | None = None
    currency: str | None = None
    category: str = 'Other'
    description: str = ''
    warnings: list[str] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)


def decode_image(data, filename):
    if Path(filename).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise ReceiptError('Upload a JPG, JPEG, or PNG image.')
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ReceiptError('Upload a nonempty image no larger than 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG'}:
                    raise ReceiptError('The file contents must be a JPEG or PNG image.')
                if source.width * source.height > MAX_PIXELS:
                    raise ReceiptError('Image is too large. Use an image with at most 20 million pixels.')
                source.verify()
            with Image.open(BytesIO(data)) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.info.clear()
                return image
    except ReceiptError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ReceiptError('This image is damaged or cannot be decoded. Try another photo.') from exc


def recognize_text(image, language='eng'):
    """Use Tesseract stdin/stdout, avoiding temporary receipt files and shell commands."""
    executable = os.environ.get('TESSERACT_CMD') or shutil.which('tesseract')
    if not executable:
        default = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Tesseract-OCR/tesseract.exe'
        if default.is_file():
            executable = str(default)
    if not executable:
        raise ReceiptError('OCR is not installed. Install Tesseract with English language data '
                           'and set TESSERACT_CMD (see README), or enter the expense manually below.')
    prepared = ImageOps.autocontrast(ImageOps.grayscale(image))
    prepared.thumbnail((3000, 5000))
    buffer = BytesIO()
    prepared.save(buffer, format='PNG')
    try:
        result = subprocess.run(
            [executable, 'stdin', 'stdout', '-l', language, '--psm', '6'],
            input=buffer.getvalue(), capture_output=True, timeout=45, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise ReceiptError('Scanning took too long. Try a clearer, cropped image or enter details manually.') from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReceiptError('OCR could not run. Check the Tesseract executable and English language data, '
                           'or enter details manually.') from exc
    text = result.stdout.decode('utf-8', errors='replace').strip()
    if not text:
        raise ReceiptError('No readable text found. Try a well-lit photo or enter the expense manually.')
    return text


def currencies(text):
    found = set()
    if re.search(r'\bUSD\b|US\s*\$|(?<![A-Za-z])\$', text, re.I):
        found.add('USD')
    if re.search(r'\bKHR\b|\briels?\b|៛', text, re.I):
        found.add('KHR')
    return found


def parse_date(text):
    """Return a date only when every recognized interpretation agrees."""
    found = set()
    ambiguous = False
    for match in re.finditer(r'\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b', text):
        try:
            found.add(date(*map(int, match.groups())))
        except ValueError:
            pass
    for match in re.finditer(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b', text):
        first, second, year = map(int, match.groups())
        year = year + 2000 if year < 100 else year
        choices = set()
        for month, day in ((first, second), (second, first)):
            try:
                choices.add(date(year, month, day))
            except ValueError:
                pass
        if len(choices) > 1:
            ambiguous = True
        found.update(choices)
    for pattern, formats in [
        (r'\b\d{1,2}[- ](?:[A-Za-z]{3,9})[- ,]+\d{4}\b', ('%d %b %Y', '%d %B %Y')),
        (r'\b[A-Za-z]{3,9}[- ]\d{1,2}[- ,]+\d{4}\b', ('%b %d %Y', '%B %d %Y')),
    ]:
        for match in re.finditer(pattern, text):
            normalized = re.sub(r'[- ,]+', ' ', match.group())
            for fmt in formats:
                try:
                    found.add(datetime.strptime(normalized, fmt).date())
                except ValueError:
                    pass
    return next(iter(found)) if len(found) == 1 and not ambiguous else None


MONEY = re.compile(r'(?<![\w.,\-/])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?![\w.,/\-])')
TOTAL = re.compile(r'\b(grand\s+total|total\s+(?:amount|due|payable)|amount\s+(?:due|payable)|balance\s+due|net\s+total|total)\b', re.I)
EXCLUDED = re.compile(r'\b(sub\s*total|subtotal|tax|vat|discount|savings?|change|cash|tendered|rounding|items?|quantity)\b', re.I)


def parse_receipt(text):
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip()]
    receipt = Receipt(date=parse_date(text))
    for line in lines[:8]:
        if (re.search(r'[A-Za-z]{2}', line) and not re.search(r'\d', line)
                and not re.search(r'\b(receipt|invoice|welcome|thank|date|time|tel|phone|cashier|address|www)\b', line, re.I)
                and not TOTAL.search(line) and not EXCLUDED.search(line)):
            receipt.merchant = line[:200]
            break
    candidates = []
    for index, line in enumerate(lines):
        label = TOTAL.search(line)
        if not label or EXCLUDED.search(line):
            continue
        tail = line[label.end():]
        values = MONEY.findall(tail)
        context = line
        if not values and index + 1 < len(lines):
            following = lines[index + 1]
            # A split-line total must be an amount/currency line, not the next item.
            if re.fullmatch(r'[\d\s.,:$៛]+|(?:USD|KHR|US\$|riels?)\s*[\d\s.,:$៛]+|[\d\s.,:$៛]+\s*(?:USD|KHR|riels?)', following, re.I):
                values = MONEY.findall(following)
                context += ' ' + following
        if values:
            if re.search(r'-\s*\d|\(\s*[\d$]', tail):
                continue
            rank = 1 if label.group().lower() == 'total' else 2
            detected = currencies(context)
            for value in values:
                amount = Decimal(value.replace(',', ''))
                if amount <= Decimal('1000000000000'):
                    candidates.append((rank, float(amount),
                                       next(iter(detected)) if len(detected) == 1 else None))
    if candidates:
        highest = max(candidate[0] for candidate in candidates)
        best = {(amount, currency) for rank, amount, currency in candidates if rank == highest}
        if len(best) == 1:
            receipt.amount, receipt.currency = best.pop()
    overall_currencies = currencies(text)
    if receipt.currency is None and len(overall_currencies) == 1:
        receipt.currency = next(iter(overall_currencies))
    if not receipt.merchant:
        receipt.warnings.append('Merchant was not recognized. Enter the store name.')
    if receipt.date is None:
        receipt.warnings.append('Date is missing or ambiguous. Confirm the transaction date.')
    if receipt.amount is None:
        receipt.warnings.append('Final total is missing or ambiguous. Check the receipt and enter the total.')
    if receipt.currency is None:
        receipt.warnings.append('Currency is missing or ambiguous. Select USD or KHR explicitly.')
    receipt.description = receipt.merchant
    receipt.items = parse_items(text, receipt.merchant, receipt.currency)
    return receipt


def suggest_category(model, text):
    """Reuse the app's trained model and its existing low-confidence threshold."""
    probabilities = model.predict_proba([text])[0]
    best = int(probabilities.argmax())
    label = str(model.classes_[best])
    return label if probabilities[best] >= 0.30 and label in CATEGORIES else 'Other'


def categorization_text(text):
    """Keep merchant/item words; receipt totals and payment metadata confuse the model."""
    descriptions = []
    for line in text.splitlines():
        if (TOTAL.search(line) or EXCLUDED.search(line)
                or re.search(r'\b(receipt|invoice|cashier|tel|phone|date|time|card|payment|thank)\b', line, re.I)):
            continue
        line = re.sub(r'\b(?:USD|KHR|riels?)\b', ' ', line, flags=re.I)
        words = ' '.join(re.findall(r'[A-Za-z]+', line))
        if words:
            descriptions.append(words)
    return ' '.join(descriptions[:20])


def expense_from_review(merchant, day, amount, currency, category, description='', items=None):
    merchant = merchant.strip()
    if not merchant or len(merchant) > 200:
        raise ReceiptError('Enter a merchant name of 1–200 characters.')
    if not isinstance(day, date):
        raise ReceiptError('Select the transaction date.')
    try:
        number = Decimal(str(amount))
    except InvalidOperation as exc:
        raise ReceiptError('Enter a valid positive total amount.') from exc
    if not number.is_finite() or number <= 0 or number > Decimal('1000000000000'):
        raise ReceiptError('Enter a positive total no larger than 1 trillion.')
    if number != number.quantize(Decimal('0.01')):
        raise ReceiptError('Use at most two decimal places for the amount.')
    if currency not in {'USD', 'KHR'} or category not in CATEGORIES:
        raise ReceiptError('Select a supported currency and category.')
    expense = {'Merchant': merchant, 'Date': day.isoformat(), 'Amount': float(number),
               'Currency': currency, 'Category': category,
               'Description': description.strip() or merchant}
    if items is not None:
        expense['Items'] = dump_items(items)
    return expense
