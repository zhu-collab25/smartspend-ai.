# SmartSpend AI

A local, single-user Streamlit expense dashboard. Upload CSV transactions, review
categories, save datasets locally, filter a month, and compare expenses with your
income, budget, savings goal, and expected expenses reserve.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

CSV columns are case sensitive. `Description` and `Amount` are required. Optional
`Date` values must use `YYYY-MM-DD`. Optional `Category` values must match the
labels in `storage.py`; existing categories are preserved. Amounts are positive
USD expenses by default; optional `Currency` must be `USD` or `KHR`. Income and
refunds should not be included. With no Date column,
the entire upload is treated as one month. Separate datasets are not merged.

Save under a name to persist the full dataset, including corrections, in
`smartspend.db`. Saving the same name replaces it. Open a saved dataset to resume
after restarting. Downloads include all months, regardless of the dashboard filter.
This app has no login: anyone accessing the app can open saved datasets. Keep it
local. Financial settings are session-only. Back up the database if needed.

## Train and evaluate

```powershell
.\.venv\Scripts\python train_model.py
.\.venv\Scripts\python test_model.py
```

Saving records changed category labels as reviewed examples. Training evaluates
on a held-out split of the base CSV, then fits the final model on the complete
base dataset plus saved corrections. For repeated descriptions, the latest
reviewed label wins. The app reloads the model after retraining when it next
categorizes an upload. Existing datasets retain their reviewed categories.

The training set is small and manually authored. Evaluation scores do not imply
accuracy on bank statements. Predictions below a heuristic probability threshold
use Other; probabilities are not calibrated confidence scores. Savings estimates
depend on complete income/expense inputs and do not represent bank balances.
Bank integration, authentication, currency conversion, and refund accounting
are not implemented.

## AI Receipt Scanner

The scanner extends this same Streamlit app and SQLite dataset workflow. Enable
**Scan a receipt**, upload a JPG/JPEG/PNG image, check the preview, and click
**Scan Receipt**. Review or correct merchant, date, total, currency, category,
description, and the **Purchased items** table. Choose a dataset name, then click
**Save Expense**. No expense is
saved until you confirm. **Enter details manually** works without OCR, and
**Cancel / Reset** clears the review and upload. To scan again, click
**Scan Receipt**; this replaces the unsaved extraction. Reviews are session-only;
keep the session open until saving or canceling.

### Installation

Python 3.10+ is required. Install the existing requirements, which now explicitly
include Pillow for image validation and preprocessing:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
winget install --id UB-Mannheim.TesseractOCR --exact
```

Install Tesseract with its English (`eng`) language data. See the
[Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
for other platforms or the Windows installer. On Debian/Ubuntu, the equivalent
system packages are `tesseract-ocr` and `tesseract-ocr-eng`.
The app checks PATH and the usual Windows installation directory. If necessary,
set the full executable path in the shell before starting Streamlit:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
.\.venv\Scripts\python -m streamlit run app.py
```

If Tesseract or its language data is missing, the scanner explains the problem
and leaves the manual review form available. No paid API or additional ML model
is used.

### Recognition and saving

Images are limited to 10 MB and 20 million decoded pixels; extensions and decoded
JPEG/PNG contents are checked. Pillow corrects orientation, converts to grayscale,
and adjusts contrast. Tesseract reads the image through stdin and returns English
text through stdout, with a 45-second timeout. No receipt image or full OCR text
is written to disk, sent to a service, or logged by the scanner. The recognized
text can be inspected in the session before saving.

Parsing prioritizes labeled final totals over subtotal, tax, discounts, cash,
and change. Missing or ambiguous fields stay empty for review, including ambiguous
numeric dates such as `11/12/2026`. Common ISO, numeric, and English month-name
dates are supported. USD (`$`, `US$`, `USD`) and KHR (`KHR`, `riel`, `៛`) indicators
are recognized. Conflicting totals require manual confirmation. Clear, upright,
English printed receipts work best. Handwriting, poor photos, unusual layouts,
automatic allocation of tax/discounts to items, and Khmer OCR are outside this version. The OCR
service accepts a language parameter for future language support; the UI uses
English only.

The existing TF-IDF/Naive Bayes classifier in `classifier.joblib` suggests one of
the existing categories from merchant/item words, excluding totals and payment
metadata that can confuse this transaction-description model. Predictions below the existing 0.30
threshold use Other. This is a trained AI prediction, not a keyword rule, and
can be corrected. Classification failure leaves a manual category selector.
Corrected labels use the existing `corrections` table and retraining workflow.

Purchased items are extracted from common English receipt rows such as
`Bread 2.00`, `Coffee 2 x 3.00 6.00`, and tables with item, quantity, unit-price,
and amount headers. Wrapped names and prices on the following line are supported.
Tax, subtotal, payment, change, and discount rows are excluded from products.
The editable table lets you correct names/numbers, add missed items, or delete
incorrect rows. Unknown quantities/prices stay blank; the scanner does not assume
every purchase has quantity one. Click **Update item totals** to check your edits.
Differences from the receipt total or quantity × unit price show a warning; tax,
discounts, and fees can legitimately cause a difference. Check the image and the
recognized text because OCR cannot guarantee every item was detected. All item
prices use the selected receipt currency; an explicitly conflicting USD/KHR price
is left blank for correction. Up to 500 item rows can be saved per expense.

Confirmed receipts append ordinary expense rows to the existing `snapshots`
table in `smartspend.db`. They immediately appear in the category editor/history,
CSV download, monthly dashboard, category charts, budget, and savings estimates.
Additional row fields are `Merchant`, `Currency`, and `Items`; `Date`, `Description`,
`Amount`, and `Category` retain their existing meanings. The app selects the saved
receipt's month and currency. Older undated records remain under **Undated**
when a dated receipt is appended; their dates are never invented. CSV imports
with a Date column still require every date to be valid; a mixed dated/undated
download can be reopened as a saved dataset or have its dates completed before
CSV reimport.

Items are saved as a JSON list in the expense's `Items` cell, using the same
snapshot and database; CSV exports/imports preserve them. Expand **Purchased
items in saved expenses** below the transaction editor to inspect them later.
An item list is detail attached to a single expense: the dashboard uses the
confirmed receipt total once, so items never double-count spending. Older
transactions do not need item data, and no new database table is required.

USD and KHR are analyzed separately using **Dashboard currency**, with separate
session income, budget, goal, and reserve settings. Older amounts without a
Currency column stay USD. No exchange rate or implicit conversion is used.

The existing database initialization adds only a `receipt_submissions` table
for idempotency tokens. Saves atomically update the snapshot, corrections, and
token. Retrying the same submission cannot add a second row. A matching merchant
or description, date, amount, and currency in the target dataset prompts a
warning; tick the repeat-purchase confirmation only for a legitimate repeat.
Matching is limited to the target dataset. Concurrent changes to an open saved
dataset are rejected by the receipt save instead of overwriting another session.
There is still no login: the existing single-user/local access model applies.

### Verify the feature

```powershell
.\.venv\Scripts\python -m unittest test_app test_receipt test_receipt_items -v
.\.venv\Scripts\python test_model.py
```

Unit and Streamlit integration tests use synthetic images, mock OCR/model calls
where needed, and temporary databases; they do not touch personal expense data.
They cover upload validation, OCR failures, parsing, category suggestions,
manual edits, persistence/reopening, rollback, duplicate protection, and currency
separation, multi-item parsing, editable item rows, and item persistence through
reopening and CSV round-trips. `test_model.py` also refreshes the existing model evaluation CSV.
To include a real Tesseract smoke test against an in-memory synthetic receipt,
set `$env:RUN_RECEIPT_OCR_TESTS = '1'` before running the unittest command.

For a manual check, open an existing dataset (or start with none), enable the
scanner, upload a clear receipt, scan, correct a value, and save. Check the row in
the category editor, its month and currency in the dashboard, and reopen the saved
dataset. Scan the same receipt again to check the duplicate warning. Try a blank
or damaged image to confirm the error and manual-entry path. Reset should discard
the review without changing saved transactions.
