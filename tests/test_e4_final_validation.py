"""Optional artifact integration checks for the completed E4 batch."""
import csv
from importlib.util import find_spec
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.validate_e4 import validate


FOLDER = Path('results/e4_trading_ablation')


@unittest.skipUnless((FOLDER / 'analysis/overall_results.csv').exists(), 'Final E4 outputs not available')
@unittest.skipUnless(find_spec('numpy'), 'Install requirements-analysis.txt to check the intervals')
class FinalE4Tests(unittest.TestCase):
    def copy(self, target):
        shutil.copytree(FOLDER, target)
        return target

    def test_validates_historical_results_without_playing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy(Path(tmp) / 'batch')
            with patch('scripts.validate_e4.source_hash', return_value='different checkout'), \
                 patch('scripts.validate_e4.run_case', side_effect=AssertionError('No replay requested')):
                result = validate(folder)
                with self.assertRaisesRegex(ValueError, 'Replay requires'):
                    validate(folder, replay_sample=True)
            self.assertTrue(result['valid'])
            self.assertEqual(result['games'], 1200)
            self.assertTrue(result['independent_bootstrap_verified'])
            self.assertFalse(result['live_package_matches_archive'])

    def test_changed_game_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy(Path(tmp) / 'batch')
            path = folder / 'games.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['turns'] = str(int(rows[0]['turns']) - 1)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'CSV/journal mismatch'):
                validate(folder)

    def test_changed_paired_effect_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy(Path(tmp) / 'batch')
            path = folder / 'analysis/paired_differences.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['difference'] = '99'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Pair difference mismatch'):
                validate(folder)
