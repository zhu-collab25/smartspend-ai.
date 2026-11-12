import unittest
import tempfile
from pathlib import Path
from io import StringIO
import pandas as pd
import storage
from streamlit.testing.v1 import AppTest

class ExpenseTests(unittest.TestCase):
    def test_validation(self):
        for amount in ['bad', float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError):
                storage.validate(pd.DataFrame({'Description': ['coffee'], 'Amount': [amount]}))
        with self.assertRaises(ValueError):
            storage.validate(pd.DataFrame({'Description': ['coffee'], 'Amount': [1], 'Date': ['2026-02-30']}))
        with self.assertRaises(ValueError):
            storage.validate(pd.DataFrame(columns=['Description', 'Amount']))
        valid = storage.validate(pd.DataFrame({'Description': [None], 'Amount': ['0'], 'Date': ['2026-11-12']}))
        self.assertEqual(valid.loc[0, 'Description'], '')

    def test_dashboard_and_saved_corrections(self):
        root = storage.ROOT
        with tempfile.TemporaryDirectory(dir=root) as directory:
            storage.ROOT = Path(directory)
            try:
                frame = pd.DataFrame({'Description': ['rent', 'electricity'], 'Amount': [100.0, 50.0], 'Date': ['2026-10-01', '2026-11-01'], 'Category': ['Housing', 'Utilities']})
                at = AppTest.from_file(str(root / 'app.py'))
                at.session_state['transactions'] = frame
                at.run()
                self.assertFalse(at.exception)
                self.assertEqual(at.metric[1].value, '$50.00')
                at.selectbox[1].select('2026-10').run()
                self.assertEqual(at.metric[1].value, '$100.00')
                at.session_state['editor_0'] = {'edited_rows': {0: {'Category': 'Other'}}, 'added_rows': [], 'deleted_rows': []}
                at.run()
                at.button[1].click().run()
                self.assertFalse(at.exception)
                with storage.connect() as db:
                    data = db.execute('SELECT data FROM snapshots').fetchone()[0]
                    correction = db.execute('SELECT category FROM corrections WHERE description=?', ('rent',)).fetchone()
                self.assertEqual(correction[0], 'Other')
                loaded = storage.validate(pd.read_json(StringIO(data)))
                self.assertEqual(loaded.loc[0, 'Category'], 'Other')
                at.selectbox[0].select('My expenses').run()
                at.button[0].click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state['transactions'].loc[0, 'Category'], 'Other')
            finally:
                storage.ROOT = root

if __name__ == '__main__':
    unittest.main()
