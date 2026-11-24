"""Chakkhu integration tests use synthetic OCR text and temporary expenses."""
from datetime import date
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from chakkhu_ocr import ChakkhuError, layout_text
import receipt
from receipt_items import load_items
from receipt_text import normalize_receipt_text
import storage

KHMER_RECEIPT = '''ផ្សារចក្ខុ
កាលបរិច្ឆេទ ១៧/១២/២០២៦
មុខទំនិញ តម្លៃ
ទឹកដោះគោ ៧ ២០០៛
នំប៉័ង ១២ ៥០០៛
សរុប ១៩ ៧០០៛
បញ្ចុះតម្លៃ ៧០០៛
ត្រូវបង់ ១៩ ០០០៛
ប្រាក់ទទួល ២០ ០០០៛
ប្រាក់អាប់ ១ ០០០៛'''


class KhmerParsingTests(unittest.TestCase):
    def test_khmer_fields_and_items(self):
        result = receipt.parse_receipt(KHMER_RECEIPT)
        self.assertEqual((result.merchant, result.date, result.amount, result.currency),
                         ('ផ្សារចក្ខុ', date(2026, 12, 17), 19000, 'KHR'))
        self.assertEqual([row['Item'] for row in result.items], ['ទឹកដោះគោ', 'នំប៉័ង'])
        self.assertEqual([row['Line Total'] for row in result.items], [7200, 12500])
        self.assertTrue(all(row['Quantity'] is None for row in result.items))

    def test_normalization_does_not_merge_quantity_and_price_columns(self):
        text = 'Milk 2 200 400\nTotal KHR ១២ ៥០០\nត្រូវបង់ ១២ ៥០០'
        self.assertEqual(normalize_receipt_text(text), 'Milk 2 200 400\nTotal KHR 12500\nGrand Total 12500')
        self.assertIsNone(receipt.parse_receipt('ផ្សារ\nត្រូវបង់ USD 12,34').amount)

    def test_conflicting_totals_and_dates_remain_for_review(self):
        result = receipt.parse_receipt('ផ្សារ\n១១/១២/២០២៦\nត្រូវបង់ ៥ ០០០៛\nត្រូវបង់ ៦ ០០០៛')
        self.assertIsNone(result.amount)
        self.assertIsNone(result.date)

    def test_region_assembly_keeps_labels_and_prices_on_the_same_row(self):
        regions = [
            {'box': [218, 437, 129, 30], 'text': 'អង្ករ'},
            {'box': [618, 426, 91, 22], 'text': '២៥ ០០០៛'},
            {'box': [221, 483, 97, 31], 'text': 'ទឹកត្រី'},
            {'box': [630, 472, 80, 20], 'text': '៩ ០០០៛'},
        ]
        self.assertEqual(layout_text(regions), 'អង្ករ ២៥ ០០០៛\nទឹកត្រី ៩ ០០០៛')


class ChakkhuAppTests(unittest.TestCase):
    def setUp(self):
        self.root = storage.ROOT
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(storage, 'ROOT', Path(self.temp.name))
        self.root_patch.start()
        self.image = BytesIO()
        Image.new('RGB', (200, 500), 'white').save(self.image, 'PNG')
        self.image.name = 'synthetic.png'

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def app(self):
        at = AppTest.from_file(str(self.root / 'app.py'), default_timeout=20)
        at.session_state['scan_receipt_open'] = True
        return at

    def find(self, elements, label):
        return next(element for element in elements if element.label == label)

    def uploader(self, original):
        def upload(label, *args, **kwargs):
            return self.image if label == 'Receipt image' else original(label, *args, **kwargs)
        return upload

    def test_scan_review_save_and_reopen_khmer_expense(self):
        import streamlit as st
        with patch('receipt_ui.st.file_uploader', side_effect=self.uploader(st.file_uploader)), \
                patch('receipt_ui.recognize_receipt', return_value=(KHMER_RECEIPT, [])) as scan:
            at = self.app().run()
            self.assertEqual(self.find(at.selectbox, 'OCR engine').value, 'Chakkhu (Khmer + English)')
            self.find(at.button, 'Scan Receipt').click().run()
            self.assertFalse(at.exception)
            scan.assert_called_once()
            self.assertEqual(self.find(at.number_input, 'Total Amount').value, 19000)
            self.assertEqual(self.find(at.selectbox, 'Currency').value, 'KHR')
            with storage.connect() as db:
                self.assertEqual(db.execute('SELECT count(*) FROM snapshots').fetchone()[0], 0)
            self.find(at.number_input, 'Total Amount').set_value(19500)
            self.find(at.selectbox, 'Expense category').select('Groceries')
            self.find(at.button, 'Save Expense').click().run()
            self.assertFalse(at.exception)
            frame = at.session_state['transactions']
            self.assertEqual((frame.loc[0, 'Amount'], frame.loc[0, 'Currency']), (19500, 'KHR'))
            self.assertEqual(load_items(frame.loc[0, 'Items'])[0]['Item'], 'ទឹកដោះគោ')
            self.assertEqual(at.session_state['dashboard_currency'], 'KHR')
            reopened = AppTest.from_file(str(self.root / 'app.py'), default_timeout=20).run()
            self.find(reopened.selectbox, 'Saved datasets').select('My expenses').run()
            self.find(reopened.button, 'Open saved dataset').click().run()
            self.assertFalse(reopened.exception)
            self.assertEqual(load_items(reopened.session_state['transactions'].loc[0, 'Items'])[1]['Line Total'], 12500)

    def test_chakkhu_failure_allows_manual_review_and_tesseract_retry(self):
        import streamlit as st
        with patch('receipt_ui.st.file_uploader', side_effect=self.uploader(st.file_uploader)), \
                patch('receipt_ui.recognize_receipt', side_effect=ChakkhuError('Chakkhu model files are missing.')), \
                patch('receipt_ui.recognize_text', return_value='SHOP\n2026-11-24\nTotal USD 5.00') as fallback:
            at = self.app().run()
            self.find(at.button, 'Scan Receipt').click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any('model files are missing' in warning.value for warning in at.warning))
            self.assertIsNone(self.find(at.number_input, 'Total Amount').value)
            self.find(at.selectbox, 'OCR engine').select('Tesseract (English)').run()
            self.find(at.button, 'Scan Receipt').click().run()
            self.assertFalse(at.exception)
            fallback.assert_called_once()
            self.assertEqual(self.find(at.number_input, 'Total Amount').value, 5)
            self.assertEqual(self.find(at.selectbox, 'Currency').value, 'USD')


if __name__ == '__main__':
    unittest.main()
