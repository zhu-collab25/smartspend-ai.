"""Receipt scanner UI using the application's existing Streamlit controls."""
from hashlib import sha256
from uuid import uuid4

import streamlit as st
import pandas as pd

from receipt import (Receipt, ReceiptError, categorization_text, decode_image, expense_from_review,
                     parse_receipt, recognize_text, suggest_category)
from storage import CATEGORIES, DuplicateReceiptError, ROOT, save_receipt
from receipt_items import ITEM_COLUMNS, item_warnings, load_items, validate_items


def _clear():
    for key in list(st.session_state):
        if key.startswith('receipt_') and key != 'receipt_generation':
            del st.session_state[key]
    st.session_state.receipt_generation = st.session_state.get('receipt_generation', 0) + 1


def render_scanner(classifier, frame=None):
    st.subheader('AI Receipt Scanner')
    st.caption('JPG, JPEG, or PNG · up to 10 MB · English printed receipts. '
               'Images stay in memory. Review every field before saving.')
    generation = st.session_state.get('receipt_generation', 0)
    uploaded = st.file_uploader('Receipt image', type=['jpg', 'jpeg', 'png'],
                                key=f'receipt_upload_{generation}', max_upload_size=10)
    image = None
    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            signature = sha256(data).hexdigest()
            if signature != st.session_state.get('receipt_signature'):
                for key in ('receipt_result', 'receipt_text', 'receipt_token'):
                    st.session_state.pop(key, None)
                st.session_state.receipt_signature = signature
                st.session_state.receipt_review_revision = st.session_state.get('receipt_review_revision', 0) + 1
            image = decode_image(data, uploaded.name)
            st.image(image, caption='Receipt preview', width=420)
        except ReceiptError as exc:
            st.error(str(exc))
    actions = st.columns(3)
    scan = actions[0].button('Scan Receipt', disabled=image is None)
    manual = actions[1].button('Enter details manually')
    cancel = actions[2].button('Cancel / Reset', key='receipt_cancel')
    if cancel:
        _clear()
        st.rerun()
    if scan or manual:
        st.session_state.pop('receipt_text', None)
        result = Receipt()
        if scan:
            try:
                with st.spinner('Reading receipt…'):
                    text = recognize_text(image)
                    result = parse_receipt(text)
                    st.session_state.receipt_text = text
                    try:
                        path = ROOT / 'classifier.joblib'
                        result.category = suggest_category(classifier(path.stat().st_mtime_ns), categorization_text(text))
                    except Exception:
                        result.warnings.append('The trained classifier is unavailable. Choose a category manually.')
            except ReceiptError as exc:
                result.warnings.append(str(exc))
            except Exception:
                result.warnings.append('This receipt could not be processed. Try another image or enter details manually.')
        st.session_state.receipt_result = result
        st.session_state.receipt_token = uuid4().hex
        st.session_state.receipt_review_revision = st.session_state.get('receipt_review_revision', 0) + 1
    result = st.session_state.get('receipt_result')
    if result is None:
        return
    st.info('Unsaved review: saving is explicit. Reset discards this review; leaving the session may lose it.')
    for warning in result.warnings:
        st.warning(warning)
    st.caption('Category suggestion uses the existing trained AI classifier. Low-confidence results use Other.')
    if st.session_state.get('receipt_text'):
        with st.expander('Recognized text (review only)'):
            st.text(st.session_state.receipt_text)
    revision = st.session_state.get('receipt_review_revision', 0)
    with st.form(f'receipt_review_{generation}_{revision}'):
        merchant = st.text_input('Merchant', value=result.merchant, max_chars=200)
        day = st.date_input('Transaction date', value=result.date, min_value=None, max_value=None)
        amount = st.number_input('Total Amount', min_value=0.0, value=result.amount, step=0.01,
                                 placeholder='Enter the final receipt total')
        currency = st.selectbox('Currency', ['USD', 'KHR'],
                                index=['USD', 'KHR'].index(result.currency) if result.currency else None,
                                placeholder='Confirm currency')
        category = st.selectbox('Expense category', CATEGORIES, index=CATEGORIES.index(result.category))
        description = st.text_input('Description (optional)', value=result.description)
        st.write('Purchased items')
        st.caption('Review item names, quantities, unit prices, and line totals. Add or remove rows as needed. '
                   'Blank values mean they were not recognized. All item prices use the selected receipt currency.')
        item_frame = pd.DataFrame(result.items, columns=ITEM_COLUMNS)
        for column in ITEM_COLUMNS[1:]:
            item_frame[column] = pd.to_numeric(item_frame[column], errors='coerce').astype(float)
        reviewed_items = st.data_editor(
            item_frame, hide_index=True, num_rows='dynamic', width='stretch',
            key=f'receipt_items_{generation}_{revision}',
            column_config={
                'Item': st.column_config.TextColumn('Item', required=True, max_chars=300),
                'Quantity': st.column_config.NumberColumn('Quantity', min_value=0.001, step=0.001),
                'Unit Price': st.column_config.NumberColumn('Unit price', min_value=0.0, step=0.01),
                'Line Total': st.column_config.NumberColumn('Line total', min_value=0.0, step=0.01),
            },
        )
        st.form_submit_button('Update item totals')
        try:
            item_rows = validate_items(reviewed_items.to_dict(orient='records'))
            for warning in item_warnings(item_rows, amount):
                st.warning(warning)
            if item_rows:
                subtotal = sum(item['Line Total'] or 0 for item in item_rows)
                st.caption(f'Listed item totals: {currency or "currency unconfirmed"} {subtotal:,.2f}. '
                           'Tax, discounts, and fees may explain differences. Dashboard spending uses the receipt total once.')
        except ValueError as exc:
            st.warning(str(exc))
        name = st.text_input('Receipt dataset', value=st.session_state.get('dataset_name', 'My expenses'))
        st.caption('The receipt is added to this dataset. Existing USD rows remain USD; currencies are never combined.')
        repeat = st.checkbox('Confirm this is a legitimate repeat purchase if a matching expense exists')
        save = st.form_submit_button('Save Expense')
    if save:
        try:
            expense = expense_from_review(merchant, day, amount, currency, category, description,
                                          items=reviewed_items.to_dict(orient='records'))
            combined, created = save_receipt(
                name, expense, st.session_state.receipt_token, frame=frame,
                baseline=st.session_state.get('transactions'), allow_duplicate=repeat,
                category_corrected=category != result.category,
            )
        except (ReceiptError, DuplicateReceiptError, ValueError) as exc:
            st.warning(str(exc))
        except Exception:
            st.error('Could not save the expense to local storage. Your review is retained; try again.')
        else:
            st.session_state.transactions = combined
            st.session_state.dataset_name = name.strip()
            st.session_state.revision = st.session_state.get('revision', 0) + 1
            st.session_state.dashboard_currency = expense['Currency']
            st.session_state.dashboard_month = expense['Date'][:7]
            notice = 'Expense saved. It is now in the transaction history and dashboard.' if created else 'This receipt was already saved; no duplicate was added.'
            _clear()
            st.session_state.receipt_saved_notice = notice
            st.rerun()


def render_saved_items(frame):
    if 'Items' not in frame:
        return
    available = {}
    for index, row in frame.iterrows():
        try:
            items = load_items(row['Items'])
        except ValueError:
            continue
        if items:
            available[index] = items
    if not available:
        return
    with st.expander('Purchased items in saved expenses'):
        selected = st.selectbox(
            'Expense items to view', list(available),
            format_func=lambda index: f'{index + 1}. {frame.loc[index, "Description"]} · '
                                      f'{frame.loc[index].get("Date", "Undated")} · '
                                      f'{frame.loc[index].get("Currency", "USD")} {frame.loc[index, "Amount"]:,.2f}',
        )
        currency = frame.loc[selected].get('Currency', 'USD')
        st.caption(f'Item prices are in {currency}. These details belong to one expense.')
        st.dataframe(pd.DataFrame(available[selected], columns=ITEM_COLUMNS), hide_index=True, width='stretch')
