"""Receipt line items, independent of OCR, UI, and storage."""
from decimal import Decimal, InvalidOperation
import json
import math
import re

ITEM_COLUMNS = ['Item', 'Quantity', 'Unit Price', 'Line Total']
NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,3})?'
SUFFIX = re.compile(rf'(?<!\S)(?:{NUMBER}|[xX×@*])(?:\s+(?:{NUMBER}|[xX×@*]))*\s*$')
SUMMARY = re.compile(r'^(?:sub\s*total|grand\s+total|net\s+total|total\b|tax\b|vat\b|gst\b|'
                     r'discount\b|savings?\b|service\s+(?:charge|fee)|delivery\s+fee|tip\b|'
                     r'amount\s+(?:due|paid|payable)|balance\s+due|cash\b|change\b|tendered\b)', re.I)
ADJUSTMENT = re.compile(r'^(?:tax\b|vat\b|gst\b|discount\b|savings?\b|coupon\b|'
                        r'service\s+(?:charge|fee)|delivery\s+fee|tip\b)', re.I)
METADATA = re.compile(r'^(?:receipt\b|invoice\b|order\b|table\b|guest\b|customer\b|cashier\b|'
                      r'server\b|date\b|time\b|tel\b|phone\b|address\b|www\.|https?://|'
                      r'thank\b|welcome\b|payment\b|card\b|visa\b|mastercard\b|'
                      r'approval\b|authorization\b|transaction\b|terminal\b|register\b|'
                      r'vat\s*(?:no|id)|tax\s*(?:no|id))', re.I)
CURRENCY = re.compile(r'\b(?:USD|KHR|riels?)\b|US\s*\$|[$៛]', re.I)
DATE = re.compile(r'\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b')
HEADER = re.compile(r'\b(description|product|item|name|qty|quantity|unit\s*price|price|rate|amount|total)\b', re.I)


def _blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or value == ''


def validate_items(items):
    """Normalize edited/imported item rows. Unknown numeric values remain null."""
    if not isinstance(items, list) or len(items) > 500:
        raise ValueError('Receipt items must be a list of at most 500 rows.')
    result = []
    for row in items:
        if not isinstance(row, dict):
            raise ValueError('Each receipt item must be a row with an item name.')
        if all(_blank(row.get(column)) for column in ITEM_COLUMNS):
            continue  # An unused new row in the editor is harmless.
        name = row.get('Item')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 300:
            raise ValueError('Each item needs a name of 1–300 characters; remove unused rows.')
        normalized = {'Item': name.strip()}
        for column in ITEM_COLUMNS[1:]:
            value = row.get(column)
            if _blank(value):
                normalized[column] = None
                continue
            try:
                number = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(f'{column} must be a valid number or blank.') from exc
            maximum = Decimal('1000000') if column == 'Quantity' else Decimal('1000000000000')
            if not number.is_finite() or number < 0 or number > maximum or (column == 'Quantity' and number == 0):
                raise ValueError(f'{column} must be finite and nonnegative; quantities must be positive.')
            precision = Decimal('0.001') if column == 'Quantity' else Decimal('0.01')
            if number != number.quantize(precision):
                raise ValueError(f'{column} allows at most {3 if column == "Quantity" else 2} decimal places.')
            normalized[column] = float(number)
        result.append(normalized)
    return result


def load_items(value):
    """Items use a JSON cell so existing snapshots and CSV downloads round-trip."""
    if _blank(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError('Items must contain a JSON list of receipt item rows.') from exc
    return validate_items(value)


def dump_items(items):
    return json.dumps(validate_items(items), ensure_ascii=False, allow_nan=False)


def _header_layout(line):
    matches = list(HEADER.finditer(line))
    names = [match.group().lower() for match in matches]
    item_positions = [index for index, name in enumerate(names) if name in {'item', 'description', 'product', 'name'}]
    if not item_positions or len(names) < 2 or re.search(r'\d', line):
        return None
    pivot = item_positions[0]
    has_total = any(name in {'amount', 'total'} for name in names)
    def column(name):
        if name in {'qty', 'quantity'}:
            return 'Quantity'
        if name in {'price', 'unit price', 'unitprice', 'rate'}:
            return 'Unit Price' if has_total or name != 'price' else 'Line Total'
        if name in {'amount', 'total'}:
            return 'Line Total'
        return None
    return ([column(name) for name in names[:pivot] if column(name)],
            [column(name) for name in names[pivot + 1:] if column(name)])


def _decimal(value):
    return Decimal(value.replace(',', ''))


def _parse_row(line, layout):
    # A trailing tax code is not part of a product's printed price.
    line = re.sub(r'(?<=\d)\s+[A-Z](?:\d)?$', '', line)
    line = CURRENCY.sub(' ', line)
    line = re.sub(r'(?<=\d)\s*[x×@*]\s*(?=\d)', ' x ', line, flags=re.I)
    line = re.sub(r'\s+', ' ', line).strip()
    suffix = SUFFIX.search(line)
    if not suffix:
        return None
    name = line[:suffix.start()].strip(' :|')
    tokens = suffix.group().split()
    values = [_decimal(token) for token in tokens if re.fullmatch(NUMBER, token)]
    if not values:
        return None
    row = dict.fromkeys(ITEM_COLUMNS)
    row['Item'] = name
    # Explicit "2 x 3.00" or "2 @ 3.00" disambiguates quantity and unit price.
    explicit = re.fullmatch(rf'({NUMBER})\s*[xX×@*]\s*({NUMBER})(?:\s+({NUMBER}))?', suffix.group().strip())
    leading = re.match(rf'^({NUMBER})\s*[xX×*]\s*(.+)$', name)
    if explicit:
        quantity, unit, total = explicit.groups()
        row.update({'Quantity': float(_decimal(quantity)), 'Unit Price': float(_decimal(unit)),
                    'Line Total': float(_decimal(total)) if total else None})
    elif leading:
        row.update({'Item': leading[2], 'Quantity': float(_decimal(leading[1])), 'Line Total': float(values[-1])})
        if len(values) == 2:
            row['Unit Price'] = float(values[0])
    elif layout:
        before, after = layout
        leading_quantity = re.match(rf'^({NUMBER})\s+(.+)$', name) if before == ['Quantity'] else None
        if leading_quantity:
            row.update({'Item': leading_quantity[2], 'Quantity': float(_decimal(leading_quantity[1]))})
        if len(values) == len(after):
            row.update({column: float(value) for column, value in zip(after, values)})
        else:
            # OCR can omit a column. Keep the final printed price and leave the rest unknown.
            row['Line Total'] = float(values[-1])
            row['Item'] += (' ' + ' '.join(str(value) for value in values[:-1])) if len(values) > 1 else ''
    else:
        row['Line Total'] = float(values[-1])
        if len(values) == 3 and values[0] > 0 and abs(values[0] * values[1] - values[2]) <= Decimal('0.01'):
            row.update({'Quantity': float(values[0]), 'Unit Price': float(values[1])})
        elif len(values) == 2:
            prefix = re.match(rf'^({NUMBER})\s+(.+)$', name)
            if prefix and abs(_decimal(prefix[1]) * values[0] - values[1]) <= Decimal('0.01'):
                row.update({'Item': prefix[2], 'Quantity': float(_decimal(prefix[1])), 'Unit Price': float(values[0])})
            else:
                row['Item'] += ' ' + str(values[0])
        elif len(values) > 1:
            row['Item'] += ' ' + ' '.join(str(value) for value in values[:-1])
    return row


def parse_items(text, merchant='', currency=None):
    """Recognize common printed rows without treating receipt summaries as products."""
    items, pending = [], []
    layout = None
    active = False
    for raw in text.splitlines():
        line = re.sub(r'\s+', ' ', raw).strip().strip('*| ')
        if not line:
            continue
        header = _header_layout(line)
        if header:
            layout, active, pending = header, True, []
            continue
        if METADATA.match(line):
            pending = []
            continue
        if ADJUSTMENT.match(line):
            pending = []
            continue
        if SUMMARY.match(line):
            break  # Do not parse payment/footer amounts as purchased items.
        if (line.casefold() == merchant.casefold() or DATE.search(line)
                or re.search(r'\b(?:street|road|avenue|blvd|postal|zip)\b', line, re.I)
                or not re.search(r'[A-Za-z0-9]', line)):
            pending = []
            continue
        row = _parse_row(line, layout)
        if row is not None:
            if pending and (active or not row['Item']):
                row['Item'] = ' '.join(pending + [row['Item']]).strip()
            if not re.search(r'[A-Za-z]{2}', row['Item']):
                pending = []
                continue
            row['Item'] = row['Item'][:300]
            # Keep product names but leave amounts unknown if an explicit other currency appears.
            other = r'\bKHR\b|\briels?\b|៛' if currency == 'USD' else r'\bUSD\b|\$' if currency == 'KHR' else None
            if other and re.search(other, line, re.I):
                row.update(dict.fromkeys(ITEM_COLUMNS[1:]))
            try:
                items.extend(validate_items([row]))
            except ValueError:
                # Oversized/unclear numbers must not take down the receipt review.
                row.update(dict.fromkeys(ITEM_COLUMNS[1:]))
                if row['Item']:
                    items.extend(validate_items([row]))
            pending, active = [], True
        elif re.search(r'[A-Za-z]{2}', line) and (active or not re.search(r'\d', line)):
            pending = (pending + [line])[-2:]
        else:
            pending = []
        if len(items) >= 500:
            break
    return items


def item_warnings(items, receipt_total):
    """Tax/discount differences warn for review; the receipt total remains authoritative."""
    warnings = []
    if not items:
        return ['No purchased items were recognized. Add rows manually or check the recognized text.']
    if any(item['Line Total'] is None for item in items):
        warnings.append('Some item totals are missing. Check these against the receipt.')
    for item in items:
        quantity, unit, total = (item[column] for column in ITEM_COLUMNS[1:])
        if all(value is not None for value in (quantity, unit, total)) and abs(
                Decimal(str(quantity)) * Decimal(str(unit)) - Decimal(str(total))) > Decimal('0.01'):
            warnings.append(f'Quantity × unit price differs from the line total for {item["Item"]}. Check for discounts or OCR errors.')
    if receipt_total is not None and all(item['Line Total'] is not None for item in items):
        subtotal = sum(Decimal(str(item['Line Total'])) for item in items)
        difference = Decimal(str(receipt_total)) - subtotal
        if abs(difference) > Decimal('0.01'):
            warnings.append(f'Items add up to {subtotal:,.2f}; the receipt total differs by {difference:,.2f}. '
                            'Check for missing items, tax, discounts, or fees. Saving uses the receipt total.')
    return warnings
