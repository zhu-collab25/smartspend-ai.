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
USD expenses; income and refunds should not be included. With no Date column,
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
