import json
import os
from datetime import date
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from streamlit.testing.v1 import AppTest

import receipt
from receipt_items import dump_items, item_warnings, load_items, parse_items, validate_items
import storage


ITEM_RECEIPT = '''CORNER MARKET
Date: 2026-11-12
Item Qty Unit Price Amount
Milk 2 1.50 3.00
Bread 1 2.00 2.00
Subtotal USD 5.00
Tax USD 0.50
TOTAL USD 5.50
Cash 10.00
Change 4.50'''


class ItemParsingTests(unittest.TestCase):
    def test_header_columns_and_summary_exclusion(self):
        result = receipt.parse_receipt(ITEM_RECEIPT)
        self.assertEqual(result.items, [
            {'Item': 'Milk', 'Quantity': 2, 'Unit Price': 1.5, 'Line Total': 3},
            {'Item': 'Bread', 'Quantity': 1, 'Unit Price': 2, 'Line Total': 2},
        ])
        self.assertEqual(result.amount, 5.5)

    def test_quantity_and_price_layouts(self):
        for line in ['2 x Coffee 3.00 6.00', 'Coffee 2 x 3.00 6.00',
                     'Coffee 2 @ 3.00 6.00', 'Coffee 2x3.00 6.00',
                     'Coffee 2 3.00 6.00', '2 Coffee 3.00 6.00']:
            with self.subTest(line=line):
                items = parse_items('SHOP\n' + line + '\nTOTAL USD 6.00', 'SHOP', 'USD')
                self.assertEqual(items, [{'Item': 'Coffee', 'Quantity': 2, 'Unit Price': 3, 'Line Total': 6}])

    def test_quantity_before_description_header(self):
        items = parse_items('SHOP\nQty Description Price Amount\n2 Coffee 3.00 6.00\nTotal 6.00', 'SHOP')
        self.assertEqual(items, [{'Item': 'Coffee', 'Quantity': 2, 'Unit Price': 3, 'Line Total': 6}])

    def test_quantity_total_header_does_not_invent_unit_price(self):
        items = parse_items('SHOP\nDescription Qty Amount\nCoffee 2 6.00\nTotal 6.00', 'SHOP')
        self.assertEqual(items[0], {'Item': 'Coffee', 'Quantity': 2, 'Unit Price': None, 'Line Total': 6})

    def test_simple_price_does_not_invent_quantity(self):
        items = parse_items('SHOP\nPhone: 012 345 678\n2026-11-12\nBread USD 2.00 A\nTOTAL USD 2.00\nVISA 2.00', 'SHOP', 'USD')
        self.assertEqual(items, [{'Item': 'Bread', 'Quantity': None, 'Unit Price': None, 'Line Total': 2}])

    def test_wrapped_name_and_numeric_price_line(self):
        items = parse_items('SHOP\nItem Qty Unit Price Amount\nOrganic Whole\nMilk 2 1.50 3.00\nBread\n1 x 2.00 2.00\nTOTAL 5.00', 'SHOP')
        self.assertEqual([item['Item'] for item in items], ['Organic Whole Milk', 'Bread'])
        self.assertEqual(items[1]['Line Total'], 2)

    def test_khr_and_product_numbers(self):
        items = parse_items('SHOP\n7 Up 330ml KHR 2,000\nWater 500ml 1,000\nTotal KHR 3,000', 'SHOP', 'KHR')
        self.assertEqual([item['Item'] for item in items], ['7 Up 330ml', 'Water 500ml'])
        self.assertEqual([item['Line Total'] for item in items], [2000, 1000])

    def test_metadata_before_products_is_not_an_item_or_stop(self):
        items = parse_items('SHOP\nVAT No 1234\nTax ID 5678\nReceipt #123\nPhnom Penh\nBread 2.00\nSubtotal 2.00\nTax 0.20\nTotal 2.20', 'SHOP')
        self.assertEqual([item['Item'] for item in items], ['Bread'])

    def test_product_discounts_do_not_hide_following_purchases(self):
        items = parse_items('SHOP\nMilk 3.00\nDiscount -0.50\nBread 2.00\nCoupon -0.25\nEggs 4.00\nTOTAL 8.25', 'SHOP')
        self.assertEqual([item['Item'] for item in items], ['Milk', 'Bread', 'Eggs'])

    def test_mixed_currency_price_left_unknown(self):
        for line, currency in [('Milk KHR 12000', 'USD'), ('Milk $3.00', 'KHR')]:
            items = parse_items(f'SHOP\n{line}\nTOTAL 3.00', 'SHOP', currency)
            self.assertEqual(items[0]['Item'], 'Milk')
            self.assertIsNone(items[0]['Line Total'])

    def test_unknown_price_and_unreadable_values(self):
        items = parse_items('SHOP\nCoffee 2 x 3.00\nTOTAL 6.00', 'SHOP')
        self.assertIsNone(items[0]['Line Total'])
        items = parse_items('SHOP\nBread ' + '9' * 400 + '\nTOTAL 2.00', 'SHOP')
        self.assertEqual(items[0]['Item'], 'Bread')
        self.assertIsNone(items[0]['Line Total'])
        self.assertEqual(parse_items('SHOP\nTOTAL 2.00\nCash 3.00\nChange 1.00', 'SHOP'), [])

    def test_review_validation_and_partial_values(self):
        valid = {'Item': 'Milk', 'Quantity': 2, 'Unit Price': 1.5, 'Line Total': 3}
        for change in [{'Item': ''}, {'Quantity': 0}, {'Quantity': -1}, {'Line Total': -1},
                       {'Unit Price': float('inf')}, {'Quantity': True}, {'Line Total': 1.001}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_items([valid | change])
        self.assertEqual(validate_items([dict.fromkeys(valid)]), [])
        self.assertEqual(load_items(dump_items([valid])), [valid])
        self.assertIsNone(validate_items([valid | {'Quantity': float('nan')}])[0]['Quantity'])
        with self.assertRaises(ValueError):
            load_items('not JSON')

    def test_reconciliation_warns_without_changing_total(self):
        items = receipt.parse_receipt(ITEM_RECEIPT).items
        self.assertEqual(item_warnings(items, 5), [])
        self.assertIn('tax', item_warnings(items, 5.5)[0])
        self.assertIn('No purchased items', item_warnings([], 5)[0])
        self.assertIn('missing', item_warnings([items[0] | {'Line Total': None}], 5)[0])
        self.assertIn('Quantity', item_warnings([items[0] | {'Line Total': 4}], 4)[0])

    @unittest.skipUnless(os.environ.get('RUN_RECEIPT_OCR_TESTS') == '1', 'Set RUN_RECEIPT_OCR_TESTS=1 with Tesseract installed')
    def test_real_ocr_recognizes_multiple_items(self):
        image = Image.new('RGB', (1500, 1000), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=34)
        for index, line in enumerate(ITEM_RECEIPT.splitlines()):
            draw.text((40, 30 + index * 80), line, fill='black', font=font)
        parsed = receipt.parse_receipt(receipt.recognize_text(image))
        self.assertEqual([item['Item'] for item in parsed.items], ['Milk', 'Bread'])
        self.assertEqual([item['Line Total'] for item in parsed.items], [3, 2])


class ItemPersistenceAndUITests(unittest.TestCase):
    def setUp(self):
        self.original_root = storage.ROOT
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(storage, 'ROOT', Path(self.temp.name))
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def app(self):
        return AppTest.from_file(str(self.original_root / 'app.py'), default_timeout=15)

    def find(self, collection, label):
        return next(element for element in collection if element.label == label)

    def test_items_persist_and_round_trip_through_csv(self):
        result = receipt.parse_receipt(ITEM_RECEIPT)
        expense = receipt.expense_from_review(result.merchant, result.date, result.amount,
                                             result.currency, 'Groceries', items=result.items)
        frame, _ = storage.save_receipt('Items', expense, 'one')
        loaded_csv = storage.validate(pd.read_csv(StringIO(frame.to_csv(index=False))))
        self.assertEqual(load_items(loaded_csv.loc[0, 'Items']), result.items)
        with storage.connect() as db:
            saved = json.loads(db.execute('SELECT data FROM snapshots WHERE name=?', ('Items',)).fetchone()[0])
        self.assertEqual(load_items(saved[0]['Items']), result.items)
        frame, created = storage.save_receipt('Items', expense, 'one')
        self.assertFalse(created)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame['Amount'].sum(), 5.5)

    def test_manual_item_correction_addition_and_removal_save_and_reopen(self):
        at = self.app()
        at.session_state['scan_receipt_open'] = True
        at.session_state['receipt_result'] = receipt.parse_receipt(ITEM_RECEIPT)
        at.session_state['receipt_token'] = 'items'
        at.run()
        self.assertFalse(at.exception)
        # Keep Milk, remove Bread, and add a corrected purchase.
        at.session_state['receipt_items_0_0'] = {
            'edited_rows': {0: {'Item': 'Whole Milk', 'Quantity': 3.0, 'Line Total': 4.5}},
            'added_rows': [{'Item': 'Apple', 'Quantity': 1.0, 'Unit Price': 0.5, 'Line Total': 0.5}],
            'deleted_rows': [1],
        }
        self.find(at.button, 'Update item totals').click().run()
        self.assertTrue(any('Items add up to 5.00' in warning.value for warning in at.warning))
        self.find(at.selectbox, 'Expense category').select('Groceries')
        self.find(at.button, 'Save Expense').click().run()
        self.assertFalse(at.exception)
        items = load_items(at.session_state['transactions'].loc[0, 'Items'])
        self.assertEqual([item['Item'] for item in items], ['Whole Milk', 'Apple'])
        self.assertEqual(items[0]['Quantity'], 3)
        self.assertEqual(at.metric[1].value, '$5.50')
        self.assertTrue(any(expander.label == 'Purchased items in saved expenses' for expander in at.expander))
        reopened = self.app().run()
        self.find(reopened.selectbox, 'Saved datasets').select('My expenses').run()
        self.find(reopened.button, 'Open saved dataset').click().run()
        self.assertFalse(reopened.exception)
        self.assertEqual(load_items(reopened.session_state['transactions'].loc[0, 'Items']), items)
        self.assertEqual(reopened.metric[1].value, '$5.50')

    def test_items_are_preserved_when_saving_categories(self):
        at = self.app()
        result = receipt.parse_receipt(ITEM_RECEIPT)
        expense = receipt.expense_from_review(result.merchant, result.date, result.amount, result.currency,
                                             'Groceries', items=result.items)
        frame, _ = storage.save_receipt('Items', expense, 'one')
        at.session_state['transactions'] = frame
        at.session_state['dataset_name'] = 'Items'
        at.run()
        at.session_state['editor_0'] = {'edited_rows': {0: {'Category': 'Other'}}, 'added_rows': [], 'deleted_rows': []}
        at.run()
        self.find(at.button, 'Save transactions and corrections').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(load_items(at.session_state['transactions'].loc[0, 'Items']), result.items)
        self.assertEqual(at.session_state['transactions'].loc[0, 'Category'], 'Other')


if __name__ == '__main__':
    unittest.main()
