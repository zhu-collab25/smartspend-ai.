"""Small normalization layer shared by receipt field and item parsers."""
import re

KHMER_DIGITS = str.maketrans('០១២៣៤៥៦៧៨៩', '0123456789')
LABELS = {
    'ចំនួនទឹកប្រាក់សរុប': 'Grand Total',
    'សរុបរួម': 'Grand Total',
    'ត្រូវបង់': 'Grand Total',
    'សរុបរង': 'Subtotal',
    'ចំនួនសរុប': 'Total',
    'សរុប': 'Total',
    'បញ្ចុះតម្លៃ': 'Discount',
    'ប្រាក់អាប់': 'Change',
    'ប្រាក់ទទួល': 'Cash',
    'ប្រាកទទួល': 'Cash',
    'សាច់ប្រាក់': 'Cash',
    'ពន្ធ': 'Tax',
    'ថ្លៃសេវា': 'Service fee',
    'កាលបរិច្ឆេទ': 'Date',
    'វិក្កយបត្រលេខ': 'Receipt number',
    'វិក្កយបត្រ': 'Receipt',
    'អ្នកគិតលុយ': 'Cashier',
    'អាសយដ្ឋាន': 'Address',
    'ផ្លូវលេខ': 'Address',
    'ទូរស័ព្ទ': 'Phone',
    'ទូរសព្ទ': 'Phone',
    'សូមអរគុណ': 'Thank you',
    'មុខទំនិញ': 'Item',
    'បរិមាណ': 'Quantity',
    'តម្លៃ': 'Price',
    'រៀល': 'KHR',
    'ដុល្លារ': 'USD',
}
_LABEL = re.compile('|'.join(re.escape(label) for label in sorted(LABELS, key=len, reverse=True)))
_GROUPED = r'\d{1,3}(?:[ \u00a0\u202f]\d{3})+'
_CURRENCY = r'(?:KHR\b|USD\b|US\$|\$|៛)'


def normalize_receipt_text(text):
    text = text.translate(KHMER_DIGITS)
    text = _LABEL.sub(lambda match: LABELS[match.group()], text)
    # Collapse space-separated thousands only with an adjacent currency or in
    # a standalone amount/total. Do not merge item quantity/price columns.
    text = re.sub(rf'(?<![\w.,])({_GROUPED})(?=\s*{_CURRENCY})',
                  lambda match: re.sub(r'\s', '', match.group()), text, flags=re.I)
    text = re.sub(rf'({_CURRENCY}\s*)({_GROUPED})(?![\d.,])',
                  lambda match: match[1] + re.sub(r'\s', '', match[2]), text, flags=re.I)
    lines = []
    for line in text.splitlines():
        match = re.fullmatch(rf'((?:(?:grand\s+total|total|subtotal|amount\s+due)\s*[:៖]?\s*)?-?)({_GROUPED})\s*', line.strip(), re.I)
        if match:
            line = match[1] + re.sub(r'\s', '', match[2])
        lines.append(line)
    return '\n'.join(lines)


def has_letters(text):
    return sum(character.isalpha() for character in text) >= 2
