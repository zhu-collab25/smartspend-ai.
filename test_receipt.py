import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from streamlit.testing.v1 import AppTest

import receipt
import storage


TEXT = '''COFFEE HOUSE
2026-11-12
Cappuccino USD 4.00
Subtotal USD 4.00
Tax 0.40
TOTAL USD 4.40
Cash 10.00
Change 5.60'''


class ImageAndOCRTests(unittest.TestCase):
    def image_bytes(self, fmt='PNG'):
        output = BytesIO()
        Image.new('RGB', (120, 100), 'white').save(output, format=fmt)
        return output.getvalue()

    def test_valid_decoded_uploads(self):
        for extension, fmt in [('png', 'PNG'), ('jpg', 'JPEG'), ('jpeg', 'JPEG')]:
            self.assertEqual(receipt.decode_image(self.image_bytes(fmt), f'r.{extension}').size, (120, 100))

    def test_invalid_uploads(self):
        for data, name in [(b'hello', 'receipt.png'), (b'', 'empty.png'),
                           (self.image_bytes(), 'r.exe'), (self.image_bytes('GIF'), 'r.png'),
                           (b'x' * (receipt.MAX_UPLOAD_BYTES + 1), 'large.jpg'),
                           (self.image_bytes()[:30], 'broken.png')]:
            with self.subTest(name=name), self.assertRaises(receipt.ReceiptError):
                receipt.decode_image(data, name)

    def test_pixel_limit_before_decode(self):
        with patch.object(receipt, 'MAX_PIXELS', 100):
            with self.assertRaisesRegex(receipt.ReceiptError, '20 million'):
                receipt.decode_image(self.image_bytes(), 'r.png')

    @patch.dict(os.environ, {'TESSERACT_CMD': 'test-tesseract'})
    @patch('receipt.subprocess.run')
    def test_ocr_streams_image_without_temporary_files(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout=TEXT.encode(), stderr=b'')
        result = receipt.recognize_text(Image.new('RGB', (100, 100)))
        self.assertEqual(result, TEXT)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ['test-tesseract', 'stdin', 'stdout', '-l', 'eng', '--psm', '6'])
        self.assertTrue(kwargs['input'].startswith(b'\x89PNG'))
        self.assertEqual(kwargs['timeout'], 45)
        self.assertNotIn('shell', kwargs)

    @patch.dict(os.environ, {'TESSERACT_CMD': 'test-tesseract'})
    @patch('receipt.subprocess.run')
    def test_ocr_errors_are_safe(self, run):
        for failure in [subprocess.TimeoutExpired('tesseract', 45),
                        subprocess.CalledProcessError(1, 'tesseract', stderr=b'private text'), OSError('private path')]:
            run.side_effect = failure
            with self.assertRaises(receipt.ReceiptError) as error:
                receipt.recognize_text(Image.new('RGB', (50, 50)))
            self.assertNotIn('private', str(error.exception))
        run.side_effect = None
        run.return_value = subprocess.CompletedProcess([], 0, stdout=b'   ', stderr=b'')
        with self.assertRaisesRegex(receipt.ReceiptError, 'No readable text'):
            receipt.recognize_text(Image.new('RGB', (50, 50)))

    @patch.dict(os.environ, {'TESSERACT_CMD': ''})
    @patch('receipt.shutil.which', return_value=None)
    @patch('receipt.Path.is_file', return_value=False)
    def test_missing_engine(self, *_):
        with self.assertRaisesRegex(receipt.ReceiptError, 'not installed'):
            receipt.recognize_text(Image.new('RGB', (10, 10)))

    @unittest.skipUnless(os.environ.get('RUN_RECEIPT_OCR_TESTS') == '1', 'Set RUN_RECEIPT_OCR_TESTS=1 with Tesseract installed')
    def test_real_tesseract_synthetic_receipt(self):
        # Synthetic pixels only: this never reads or retains a personal receipt.
        image = Image.new('RGB', (1100, 900), 'white')
        drawing = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=36)
        for index, line in enumerate(TEXT.replace('COFFEE HOUSE', 'COFFEE CAFE').splitlines()):
            drawing.text((50, 40 + index * 90), line, font=font, fill='black')
        recognized = receipt.recognize_text(image)
        result = receipt.parse_receipt(recognized)
        self.assertEqual(result.merchant, 'COFFEE CAFE')
        self.assertEqual(result.date, date(2026, 11, 12))
        self.assertEqual(result.amount, 4.4)
        self.assertEqual(result.currency, 'USD')
        from joblib import load
        model = load(storage.ROOT / 'classifier.joblib')
        self.assertEqual(receipt.suggest_category(model, receipt.categorization_text(recognized)), 'Food & Drinks')


class ParserTests(unittest.TestCase):
    def test_core_fields_and_final_total(self):
        result = receipt.parse_receipt(TEXT)
        self.assertEqual((result.merchant, result.date, result.amount, result.currency),
                         ('COFFEE HOUSE', date(2026, 11, 12), 4.4, 'USD'))
        self.assertEqual(result.warnings, [])

    def test_khr_and_split_line_total(self):
        result = receipt.parse_receipt('MARKET\n13/11/2026\nSubtotal 10,000\nGRAND TOTAL\nKHR 12,000\nCash 20,000\nChange 8,000')
        self.assertEqual((result.amount, result.currency), (12000, 'KHR'))
        self.assertEqual(result.date, date(2026, 11, 13))

    def test_currency_indicators(self):
        for indicator, expected in [('$', 'USD'), ('US$', 'USD'), ('USD', 'USD'), ('៛', 'KHR'), ('riel', 'KHR'), ('KHR', 'KHR')]:
            with self.subTest(indicator=indicator):
                result = receipt.parse_receipt(f'SHOP\nTOTAL {indicator} 100.00')
                self.assertEqual(result.currency, expected)

    def test_dates(self):
        for text, expected in [('2026/11/13', date(2026, 11, 13)), ('13-11-26', date(2026, 11, 13)),
                               ('Nov 13, 2026', date(2026, 11, 13)), ('13 November 2026', date(2026, 11, 13)),
                               ('11/12/2026', None), ('2026-02-30', None), ('Date unknown', None),
                               ('2026-11-12\n2026-11-13', None)]:
            with self.subTest(text=text):
                self.assertEqual(receipt.parse_date(text), expected)

    def test_missing_values_are_not_fabricated(self):
        result = receipt.parse_receipt('Receipt\nTax 1.00\nSubtotal 10.00\nCash 20.00\nChange 9.00')
        self.assertEqual((result.merchant, result.date, result.amount, result.currency), ('', None, None, None))
        self.assertEqual(len(result.warnings), 4)

    def test_ambiguous_totals_and_currency(self):
        for text in ['TOTAL USD 5.00\nTOTAL USD 6.00', 'TOTAL USD 5.00\nTOTAL KHR 20000']:
            self.assertIsNone(receipt.parse_receipt(text).amount)
        self.assertIsNone(receipt.parse_receipt('SHOP\nTOTAL 5.00').currency)
        result = receipt.parse_receipt('SHOP\nTOTAL USD 5.00\nEquivalent KHR 20000')
        self.assertEqual((result.amount, result.currency), (5, 'USD'))

    def test_total_ranking_and_non_total_labels(self):
        result = receipt.parse_receipt('SHOP\nTotal 10.00\nGrand Total USD 11.00\nTotal savings 4.00\nTotal tax 1.00\nTotal items 8')
        self.assertEqual(result.amount, 11)
        self.assertIsNone(receipt.parse_receipt('SHOP\nTotal\nCoffee 9.00').amount)
        self.assertIsNone(receipt.parse_receipt('SHOP\nTotal USD -5.00').amount)
        self.assertIsNone(receipt.parse_receipt('SHOP\nTotal USD 12,34').amount)
        self.assertIsNone(receipt.parse_receipt('SHOP\nTotal USD ' + '9' * 400).amount)

    def test_trained_classifier_and_threshold(self):
        class Model:
            classes_ = np.array(['Groceries', 'Shopping'])
            def predict_proba(self, descriptions):
                self.descriptions = descriptions
                return np.array([[0.9, 0.1]])
        model = Model()
        self.assertEqual(receipt.suggest_category(model, 'market vegetables'), 'Groceries')
        self.assertEqual(model.descriptions, ['market vegetables'])
        with patch.object(model, 'predict_proba', return_value=np.array([[0.2, 0.1]])):
            self.assertEqual(receipt.suggest_category(model, 'unknown'), 'Other')
        model.classes_ = np.array(['Unknown', 'Shopping'])
        self.assertEqual(receipt.suggest_category(model, 'x'), 'Other')

    def test_category_input_uses_merchant_and_item_words(self):
        self.assertEqual(receipt.categorization_text(TEXT), 'COFFEE HOUSE Cappuccino')

    def test_review_validation(self):
        arguments = dict(merchant='Cafe', day=date(2026, 11, 12), amount=4.4, currency='USD', category='Food & Drinks')
        for key, invalid in [('merchant', ''), ('day', None), ('amount', 0), ('amount', 'nan'),
                             ('amount', 'inf'), ('amount', -1), ('amount', None), ('amount', 1.001),
                             ('currency', None), ('currency', 'EUR'), ('category', 'Invalid')]:
            with self.subTest(key=key, invalid=invalid), self.assertRaises(receipt.ReceiptError):
                receipt.expense_from_review(**(arguments | {key: invalid}))


class StorageAndUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_root = storage.ROOT
        self.root_patch = patch.object(storage, 'ROOT', Path(self.temp.name))
        self.root_patch.start()
        self.expense = receipt.expense_from_review('Cafe', date(2026, 11, 12), 4.4, 'USD', 'Food & Drinks')

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def app(self):
        return AppTest.from_file(str(self.original_root / 'app.py'), default_timeout=15)

    def find(self, collection, label):
        return next(element for element in collection if element.label == label)

    def test_save_and_idempotent_replay(self):
        frame, created = storage.save_receipt('Receipts', self.expense, 'token')
        self.assertTrue(created)
        for _ in range(3):
            frame, created = storage.save_receipt('Receipts', self.expense, 'token')
            self.assertFalse(created)
            self.assertEqual(len(frame), 1)
        with storage.connect() as db:
            saved = json.loads(db.execute('SELECT data FROM snapshots WHERE name=?', ('Receipts',)).fetchone()[0])
            self.assertEqual(saved[0]['Merchant'], 'Cafe')
            self.assertEqual(db.execute('SELECT count(*) FROM receipt_submissions').fetchone()[0], 1)

    def test_duplicate_warning_and_legitimate_repeat(self):
        storage.save_receipt('Receipts', self.expense, 'one')
        with self.assertRaises(storage.DuplicateReceiptError):
            storage.save_receipt('Receipts', self.expense, 'two')
        frame, created = storage.save_receipt('Receipts', self.expense, 'two', allow_duplicate=True)
        self.assertTrue(created)
        self.assertEqual(len(frame), 2)
        khr = self.expense | {'Currency': 'KHR'}
        frame, _ = storage.save_receipt('Receipts', khr, 'three')
        self.assertEqual(len(frame), 3)

    def test_concurrent_repeated_submission_is_saved_once(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: storage.save_receipt('Receipts', self.expense, 'same-token'), range(2)))
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertTrue(all(len(frame) == 1 for frame, _ in results))

    def test_legacy_undated_dataset_and_category_correction(self):
        original = pd.DataFrame({'Description': ['rent'], 'Amount': [100], 'Category': ['Housing']})
        frame, _ = storage.save_receipt('Mixed', self.expense, 'one', frame=original,
                                        baseline=original, category_corrected=True)
        self.assertTrue(pd.isna(frame.loc[0, 'Date']))
        self.assertEqual(frame.loc[0, 'Currency'], 'USD')
        self.assertEqual(frame.loc[1, 'Date'], '2026-11-12')
        with storage.connect() as db:
            self.assertEqual(db.execute('SELECT category FROM corrections WHERE description=?', ('Cafe',)).fetchone()[0], 'Food & Drinks')
        # Reopened snapshots support unknown legacy dates; CSV validation stays strict.
        storage.validate(frame, allow_undated=True)
        with self.assertRaises(ValueError):
            storage.validate(frame)

    def test_save_retains_reviewed_rows_and_rejects_stale_overwrite(self):
        original, _ = storage.save_receipt('Receipts', self.expense, 'one')
        edited = original.copy()
        edited.loc[0, 'Category'] = 'Other'
        extra = self.expense | {'Merchant': 'Market', 'Description': 'Market', 'Amount': 10}
        combined, _ = storage.save_receipt('Receipts', extra, 'two', frame=edited, baseline=original)
        self.assertEqual(combined.loc[0, 'Category'], 'Other')
        with self.assertRaisesRegex(ValueError, 'changed'):
            storage.save_receipt('Receipts', extra, 'three', frame=original, baseline=original)
        with storage.connect() as db:
            self.assertEqual(db.execute('SELECT category FROM corrections WHERE description=?', ('Cafe',)).fetchone()[0], 'Other')

    def test_database_failure_rolls_back_receipt(self):
        with storage.connect() as db:
            db.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON receipt_submissions BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with self.assertRaises(Exception):
            storage.save_receipt('Failed', self.expense, 'one')
        with storage.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM snapshots').fetchone()[0], 0)

    def test_manual_review_save_dashboard_and_reopen(self):
        at = self.app().run()
        self.assertFalse(at.exception)
        at.toggle[0].set_value(True).run()
        self.find(at.button, 'Enter details manually').click().run()
        self.find(at.text_input, 'Merchant').set_value('Corrected Market')
        at.date_input[0].set_value(date(2026, 11, 12))
        self.find(at.number_input, 'Total Amount').set_value(12000.0)
        self.find(at.selectbox, 'Currency').select('KHR')
        self.find(at.selectbox, 'Expense category').select('Groceries')
        self.find(at.text_input, 'Receipt dataset').set_value('Scanned')
        self.find(at.button, 'Save Expense').click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Expense saved' in message.value for message in at.success))
        self.assertEqual(at.metric[1].value, 'KHR 12,000.00')
        self.assertEqual(at.session_state['transactions'].loc[0, 'Merchant'], 'Corrected Market')
        reopened = self.app().run()
        self.find(reopened.selectbox, 'Saved datasets').select('Scanned').run()
        self.find(reopened.button, 'Open saved dataset').click().run()
        self.assertFalse(reopened.exception)
        self.assertEqual(reopened.metric[1].value, 'KHR 12,000.00')

    def test_usd_khr_totals_and_budgets_are_separate(self):
        at = self.app()
        at.session_state['transactions'] = pd.DataFrame([
            self.expense, self.expense | {'Currency': 'KHR', 'Amount': 12000}])
        at.run()
        self.find(at.selectbox, 'Dashboard currency').select('USD').run()
        self.assertEqual(at.metric[1].value, '$4.40')
        self.assertTrue(any('Budget used: 0.9%' in item.value for item in at.markdown))
        self.find(at.number_input, 'Monthly spending budget ($)').set_value(750.0).run()
        self.find(at.selectbox, 'Dashboard currency').select('KHR').run()
        self.assertEqual(at.metric[1].value, 'KHR 12,000.00')
        self.find(at.number_input, 'Monthly spending budget (KHR)').set_value(20000.0).run()
        self.assertTrue(any('Budget used: 60.0%' in item.value for item in at.markdown))
        self.find(at.selectbox, 'Dashboard currency').select('USD').run()
        self.assertEqual(self.find(at.number_input, 'Monthly spending budget ($)').value, 750)

    def test_failed_save_keeps_review_and_reset_discards(self):
        at = self.app()
        at.session_state['scan_receipt_open'] = True
        at.session_state['receipt_result'] = receipt.parse_receipt(TEXT)
        at.session_state['receipt_token'] = 'one'
        at.run()
        self.find(at.selectbox, 'Expense category').select('Food & Drinks')
        with patch('receipt_ui.save_receipt', side_effect=OSError('secret path')):
            self.find(at.button, 'Save Expense').click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Could not save' in error.value for error in at.error))
        self.assertNotIn('secret', str(at.error))
        self.assertEqual(self.find(at.text_input, 'Merchant').value, 'COFFEE HOUSE')
        self.find(at.button, 'Cancel / Reset').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.form), 0)

    def test_uploaded_receipt_scan_review_and_save(self):
        image = BytesIO()
        Image.new('RGB', (300, 600), 'white').save(image, 'PNG')
        image.name = 'synthetic.png'
        import streamlit as st
        original_uploader = st.file_uploader

        def upload(label, *args, **kwargs):
            return image if label == 'Receipt image' else original_uploader(label, *args, **kwargs)

        class Model:
            classes_ = np.array(['Food & Drinks', 'Other'])
            def predict_proba(self, descriptions):
                return np.array([[0.9, 0.1]])

        with patch('receipt_ui.st.file_uploader', side_effect=upload), \
                patch('receipt_ui.recognize_text', return_value=TEXT) as ocr, \
                patch('receipt_ui.ROOT', self.original_root), \
                patch('joblib.load', return_value=Model()):
            at = self.app()
            at.session_state['scan_receipt_open'] = True
            at.run()
            self.find(at.button, 'Scan Receipt').click().run()
            self.assertFalse(at.exception)
            ocr.assert_called_once()
            self.assertEqual(self.find(at.text_input, 'Merchant').value, 'COFFEE HOUSE')
            self.assertEqual(self.find(at.selectbox, 'Expense category').value, 'Food & Drinks')
            # Nothing is saved by scanning alone.
            with storage.connect() as db:
                self.assertEqual(db.execute('SELECT count(*) FROM snapshots').fetchone()[0], 0)
            self.find(at.number_input, 'Total Amount').set_value(5.5)
            self.find(at.button, 'Save Expense').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[1].value, '$5.50')
            self.assertEqual(at.session_state['transactions'].loc[0, 'Amount'], 5.5)

    def test_ocr_failure_allows_manual_review(self):
        image = BytesIO()
        Image.new('RGB', (100, 100), 'white').save(image, 'PNG')
        image.name = 'blank.png'
        import streamlit as st
        original_uploader = st.file_uploader
        def upload(label, *args, **kwargs):
            return image if label == 'Receipt image' else original_uploader(label, *args, **kwargs)
        with patch('receipt_ui.st.file_uploader', side_effect=upload), \
                patch('receipt_ui.recognize_text', side_effect=receipt.ReceiptError('No readable text found.')):
            at = self.app()
            at.session_state['scan_receipt_open'] = True
            at.run()
            self.find(at.button, 'Scan Receipt').click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any('No readable text' in item.value for item in at.warning))
            self.assertEqual(self.find(at.text_input, 'Merchant').value, '')
            self.assertIsNone(self.find(at.number_input, 'Total Amount').value)
            self.assertIsNone(self.find(at.selectbox, 'Currency').value)

    def test_undated_history_is_preserved_and_visible(self):
        original = pd.DataFrame({'Description': ['rent'], 'Amount': [100], 'Category': ['Housing']})
        frame, _ = storage.save_receipt('Mixed', self.expense, 'one', frame=original, baseline=original)
        at = self.app()
        at.session_state['transactions'] = frame
        at.run()
        self.find(at.selectbox, 'Month to analyze').select('Undated').run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[1].value, '$100.00')
        self.find(at.selectbox, 'Month to analyze').select('2026-11').run()
        self.assertEqual(at.metric[1].value, '$4.40')


if __name__ == '__main__':
    unittest.main()
