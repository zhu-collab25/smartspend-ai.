from pathlib import Path
import sqlite3
from contextlib import contextmanager
import pandas as pd

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['Food & Drinks', 'Transportation', 'Shopping', 'Entertainment', 'Groceries', 'Housing', 'Utilities', 'Healthcare', 'Education', 'Insurance', 'Subscriptions', 'Travel', 'Other']

@contextmanager
def connect():
    db = sqlite3.connect(ROOT / 'smartspend.db')
    try:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, data TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS corrections (description TEXT PRIMARY KEY, category TEXT NOT NULL)')
            yield db
    finally:
        db.close()

def validate(frame):
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
        if dates.isna().any():
            raise ValueError('Every Date must use YYYY-MM-DD.')
        frame['Date'] = dates.dt.strftime('%Y-%m-%d')
    if 'Category' in frame and not frame['Category'].isin(CATEGORIES).all():
        raise ValueError('Unsupported categories. Remove the Category column to predict again.')
    return frame
