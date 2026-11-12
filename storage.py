from pathlib import Path
import sqlite3
from contextlib import contextmanager
import json
import pandas as pd
from receipt_items import dump_items, load_items

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['Food & Drinks', 'Transportation', 'Shopping', 'Entertainment', 'Groceries', 'Housing', 'Utilities', 'Healthcare', 'Education', 'Insurance', 'Subscriptions', 'Travel', 'Other']

@contextmanager
def connect():
    db = sqlite3.connect(ROOT / 'smartspend.db')
    try:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, data TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS corrections (description TEXT PRIMARY KEY, category TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS receipt_submissions (token TEXT PRIMARY KEY, dataset TEXT NOT NULL)')
            yield db
    finally:
        db.close()

def validate(frame, allow_undated=False):
    if not {'Description', 'Amount'}.issubset(frame.columns):
        raise ValueError('CSV must contain Description and Amount columns.')
    if frame.empty:
        raise ValueError('No transactions found.')
    frame = frame.copy().reset_index(drop=True)
    frame['Description'] = frame['Description'].fillna('').astype(str).str.strip()
    frame['Amount'] = pd.to_numeric(frame['Amount'], errors='coerce')
    if frame['Amount'].isna().any() or not frame['Amount'].map(lambda n: 0 <= n < float('inf')).all():
        raise ValueError('Amounts must be finite, nonnegative numbers. Refunds are not supported.')
    if 'Date' in frame:
        dates = pd.to_datetime(frame['Date'], format='%Y-%m-%d', errors='coerce')
        missing = frame['Date'].isna() | frame['Date'].eq('')
        if (dates.isna() & ~(missing & allow_undated)).any():
            raise ValueError('Every Date must use YYYY-MM-DD.')
        frame['Date'] = dates.dt.strftime('%Y-%m-%d')
    if 'Currency' in frame and not frame['Currency'].isin(['USD', 'KHR']).all():
        raise ValueError('Currency must be USD or KHR. No currency conversion is performed.')
    if 'Category' in frame and not frame['Category'].isin(CATEGORIES).all():
        raise ValueError('Unsupported categories. Remove the Category column to predict again.')
    if 'Items' in frame:
        frame['Items'] = frame['Items'].map(lambda value: dump_items(load_items(value)))
    return frame


class DuplicateReceiptError(ValueError):
    pass


def _snapshot(data):
    return validate(pd.DataFrame(json.loads(data)), allow_undated=True)


def _same_frame(left, right):
    left, right = left.copy(), right.copy()
    left['Amount'] = left['Amount'].astype(float)
    right['Amount'] = right['Amount'].astype(float)
    return left.to_json(orient='records', date_format='iso') == right.to_json(orient='records', date_format='iso')


def matching_receipt(frame, expense):
    if frame.empty or 'Date' not in frame:
        return False
    merchant = frame.get('Merchant', frame['Description']).fillna(frame['Description'])
    currency = frame['Currency'] if 'Currency' in frame else pd.Series('USD', index=frame.index)
    same_merchant = merchant.astype(str).str.strip().str.casefold().eq(expense['Merchant'].strip().casefold())
    same_description = frame['Description'].str.strip().str.casefold().eq(expense['Description'].strip().casefold())
    return bool(((same_merchant | same_description) & frame['Date'].eq(expense['Date'])
                 & frame['Amount'].eq(expense['Amount']) & currency.eq(expense['Currency'])).any())


def save_receipt(name, expense, token, frame=None, baseline=None, allow_duplicate=False,
                 category_corrected=False):
    """Atomically append to the existing snapshot and record an idempotency token.

    frame/baseline let the UI include reviewed categories without overwriting a
    snapshot changed by another session. No receipt image or OCR text is stored.
    """
    if not name.strip() or not token:
        raise ValueError('Enter a dataset name before saving.')
    name = name.strip()
    row = validate(pd.DataFrame([expense]))
    if (row.loc[0, 'Amount'] <= 0 or 'Currency' not in row or 'Date' not in row
            or not str(expense.get('Merchant', '')).strip()):
        raise ValueError('Confirm merchant, date, positive amount, and currency before saving.')
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT dataset FROM receipt_submissions WHERE token=?', (token,)).fetchone()
        if previous:
            data = db.execute('SELECT data FROM snapshots WHERE name=?', (previous[0],)).fetchone()
            if not data:
                raise ValueError('This receipt was already saved. Open its original dataset.')
            return _snapshot(data[0]), False
        saved = db.execute('SELECT data FROM snapshots WHERE name=?', (name,)).fetchone()
        persisted = _snapshot(saved[0]) if saved else None
        if frame is not None:
            base = validate(frame, allow_undated=True)
            if persisted is not None and (baseline is None or not _same_frame(
                    persisted, validate(baseline, allow_undated=True))):
                raise ValueError('This dataset already exists or changed in another session. '
                                 'Open its latest version or choose a new dataset name.')
        else:
            base = persisted if persisted is not None else pd.DataFrame(columns=['Description', 'Amount', 'Category'])
        if matching_receipt(base, expense) and not allow_duplicate:
            raise DuplicateReceiptError('A matching merchant, date, amount, and currency already exists. '
                                        'Confirm a legitimate repeat purchase to save another expense.')
        base = base.copy()
        if 'Currency' not in base:
            base['Currency'] = 'USD'
        if 'Date' not in base:
            base['Date'] = None  # Legacy undated transactions stay undated.
        if 'Category' not in base:
            base['Category'] = 'Other'
        combined = validate(pd.concat([base, row], ignore_index=True), allow_undated=True)
        db.execute('INSERT OR REPLACE INTO snapshots (name, data) VALUES (?, ?)',
                   (name, combined.to_json(orient='records', date_format='iso')))
        db.execute('INSERT INTO receipt_submissions VALUES (?, ?)', (token, name))
        if frame is not None and baseline is not None and 'Category' in baseline:
            changed = frame['Category'].ne(baseline['Category']) & frame['Description'].ne('')
            db.executemany('INSERT OR REPLACE INTO corrections VALUES (?, ?)',
                           frame.loc[changed, ['Description', 'Category']].itertuples(index=False, name=None))
        if category_corrected:
            db.execute('INSERT OR REPLACE INTO corrections VALUES (?, ?)',
                       (expense['Description'], expense['Category']))
    return combined, True
