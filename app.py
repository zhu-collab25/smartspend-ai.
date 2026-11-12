from hashlib import sha256
from io import BytesIO, StringIO
import pandas as pd
import streamlit as st
from joblib import load
from storage import ROOT, CATEGORIES, connect, validate

st.set_page_config(page_title='SmartSpend AI', page_icon='💰', layout='wide')
st.title('💰 SmartSpend AI')
st.caption('Personal expense analysis and budgeting')
st.sidebar.title('Financial settings')
income = st.sidebar.number_input('Monthly income ($)', min_value=0.0, value=600.0)
budget = st.sidebar.number_input('Monthly spending budget ($)', min_value=0.0, value=500.0)
goal = st.sidebar.number_input('Monthly savings goal ($)', min_value=0.0, value=100.0)
st.sidebar.caption('Saved data is stored locally in smartspend.db. This is a single-user app: anyone accessing this app can open saved datasets.')

@st.cache_resource
def classifier(modified):
    return load(ROOT / 'classifier.joblib')

st.header('Upload or open transactions')
st.write('CSV columns: Description, Amount, and optionally Date (YYYY-MM-DD). Use positive expense amounts in USD.')
try:
    with connect() as db:
        names = [r[0] for r in db.execute('SELECT name FROM snapshots ORDER BY name')]
except Exception:
    names = []
    st.warning('Local storage is unavailable. CSV uploads and downloads still work.')
selected = st.selectbox('Saved datasets', ['Select a dataset'] + names)
if st.button('Open saved dataset', disabled=selected not in names):
    try:
        with connect() as db:
            data = db.execute('SELECT data FROM snapshots WHERE name=?', (selected,)).fetchone()[0]
        st.session_state.transactions = validate(pd.read_json(StringIO(data)))
        st.session_state.dataset_name = selected
        st.session_state.revision = st.session_state.get('revision', 0) + 1
    except Exception as exc:
        st.error(f'Could not open dataset: {exc}')

uploaded = st.file_uploader('Expense CSV', type=['csv'])
if uploaded is not None:
    signature = sha256(uploaded.getvalue()).hexdigest()
    if signature != st.session_state.get('signature'):
        try:
            frame = validate(pd.read_csv(BytesIO(uploaded.getvalue())))
            if 'Category' not in frame:
                frame['Category'] = 'Other'
                nonblank = frame['Description'].ne('')
                if nonblank.any():
                    try:
                        path = ROOT / 'classifier.joblib'
                        model = classifier(path.stat().st_mtime_ns)
                        probabilities = model.predict_proba(frame.loc[nonblank, 'Description'])
                        labels = model.classes_[probabilities.argmax(axis=1)].copy()
                        labels[probabilities.max(axis=1) < 0.30] = 'Other'
                        frame.loc[nonblank, 'Category'] = labels
                    except Exception:
                        st.warning('Classifier unavailable. Run python train_model.py to rebuild it; you can assign categories manually below.')
            st.session_state.transactions = frame
            st.session_state.dataset_name = uploaded.name
            st.session_state.signature = signature
            st.session_state.revision = st.session_state.get('revision', 0) + 1
        except Exception as exc:
            st.error(f'Could not import CSV: {exc}')
            st.stop()
if 'transactions' not in st.session_state:
    st.info('Upload a CSV or open a saved dataset to begin.')
    st.stop()

st.subheader('Review categories')
if st.session_state.pop('saved_notice', False):
    st.success('Saved. Run python train_model.py to incorporate reviewed corrections into the classifier.')
st.caption('Low confidence predictions and blank descriptions use Other. Review categories before saving.')
original = st.session_state.transactions
edited = st.data_editor(original, hide_index=True, width='stretch',
    disabled=[c for c in original.columns if c != 'Category'],
    column_config={'Category': st.column_config.SelectboxColumn(options=CATEGORIES, required=True)},
    key=f"editor_{st.session_state.get('revision', 0)}")
name = st.text_input('Save dataset as', value=st.session_state.get('dataset_name', 'My expenses'))
st.caption('Saving replaces the dataset with this name. Changed categories are recorded as reviewed training examples.')
if st.button('Save transactions and corrections'):
    if not name.strip():
        st.error('Enter a dataset name.')
    else:
        changed = edited['Category'].ne(original['Category']) & edited['Description'].ne('')
        corrections = list(edited.loc[changed, ['Description', 'Category']].itertuples(index=False, name=None))
        try:
            with connect() as db:
                db.execute('INSERT OR REPLACE INTO snapshots VALUES (?, ?)', (name.strip(), edited.to_json(orient='records', date_format='iso')))
                db.executemany('INSERT OR REPLACE INTO corrections VALUES (?, ?)', corrections)
            st.session_state.transactions = edited.copy()
            st.session_state.dataset_name = name.strip()
            st.session_state.revision = st.session_state.get('revision', 0) + 1
            st.session_state.saved_notice = True
            st.rerun()
        except Exception as exc:
            st.error(f'Could not save: {exc}')

df = edited.copy()
if 'Date' in df:
    month = st.selectbox('Month to analyze', sorted(df['Date'].str[:7].unique(), reverse=True))
    df = df[df['Date'].str.startswith(month)]
else:
    st.warning('No Date column: all rows are treated as one month. Upload one month of expenses for meaningful comparisons.')
st.header('Financial dashboard')
total = df['Amount'].sum()
remaining = income - total
a, b, c = st.columns(3)
a.metric('Monthly income', f'${income:,.2f}')
b.metric('Listed expenses', f'${total:,.2f}')
c.metric('Income less listed expenses', f'${remaining:,.2f}')
if budget:
    st.progress(min(total / budget, 1.0))
    st.write(f'Budget used: {total / budget:.1%}')
    if total > budget:
        st.error(f'Over budget by ${total - budget:,.2f}.')
    else:
        st.success(f'Budget remaining: ${budget - total:,.2f}.')
else:
    st.info('Set a spending budget to track progress.')
spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
st.subheader('Spending by category')
st.bar_chart(spending)
st.dataframe(spending.rename('Total ($)'), width='stretch')
if total > 0:
    category, amount = spending.idxmax(), spending.max()
    st.write(f'Largest category: **{category}**, ${amount:,.2f} ({amount / total:.1%} of expenses).')
    reduction = st.slider('Explore a reduction in this category (%)', 5, 30, 10, 5)
    st.info(f'A {reduction}% reduction could leave an additional ${amount * reduction / 100:,.2f}.')
st.subheader('Savings goal estimate')
reserve = st.number_input('Other expected expenses / money to set aside ($)', min_value=0.0, value=0.0)
available = max(0.0, remaining - reserve)
st.caption('Assumes income and expenses cover the selected month. This is an estimate, not a bank balance or confirmed savings.')
st.write(f'Estimated amount available after the reserve: **${available:,.2f}**')
if goal:
    st.progress(min(available / goal, 1.0))
    if available >= goal:
        st.success('Your estimated surplus covers the savings goal.')
    else:
        st.warning(f'Estimated savings shortfall: ${goal - available:,.2f}.')
else:
    st.info('Set a savings goal to track progress.')
st.download_button('Download all categorized transactions', edited.to_csv(index=False).encode('utf-8'), 'smartspend_results.csv', 'text/csv')
st.caption('Predictions can be wrong. Budgeting suggestions are educational estimates.')
