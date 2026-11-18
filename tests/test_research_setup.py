import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from monopoly.e4 import load_config, policy_audit, prepare, run, schedule
from monopoly.e4_analysis import analyze, bootstrap
from scripts.freeze_e1 import verify


class ResearchSetupTests(unittest.TestCase):
    def test_final_schedule_and_policy_isolation(self):
        config = load_config('configs/e4_final.json')
        cases = schedule(config)
        self.assertEqual(len(cases), 1200)
        self.assertEqual(len({c['game_id'] for c in cases}), 1200)
        for strategy in config['strategies']:
            self.assertEqual(sum(c['strategy'] == strategy for c in cases), 200)
            for seat in (1, 2, 3, 4):
                self.assertEqual(sum(c['strategy'] == strategy and c['test_seat'] == seat for c in cases), 50)
        self.assertEqual(policy_audit()['checked_configurations'], 24)

    def test_prepare_does_not_play_and_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'smoke'
            with patch('monopoly.e4.run_case', side_effect=AssertionError('Prepare must not play')):
                prepare(Path('configs/e4_smoke.json'), output)
            run(output, max_games=2)
            self.assertEqual(json.loads((output / 'progress.json').read_text())['completed'], 2)
            from monopoly.e4 import run_case
            with patch('monopoly.e4.run_case', wraps=run_case) as play:
                run(output)
                self.assertEqual(play.call_count, 22)
            with patch('monopoly.e4.run_case', side_effect=AssertionError('Already completed')):
                run(output)
            analyze(output)
            self.assertTrue(json.loads((output / 'analysis/validation.json').read_text())['complete'])
            with self.assertRaises(ValueError):
                analyze(output)

    def test_resume_rejects_source_and_config_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'smoke'
            prepare(Path('configs/e4_smoke.json'), output)
            with patch('monopoly.e4.source_hash', return_value='changed'), self.assertRaises(ValueError):
                run(output)
            config = json.loads((output / 'config.json').read_text())
            config['seed_start'] += 1
            (output / 'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(output)

    def test_prepare_failure_can_be_retried_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'smoke'
            with patch('monopoly.e4.archive_source', side_effect=OSError('Disk failure')), \
                 self.assertRaisesRegex(OSError, 'Disk failure'):
                prepare(Path('configs/e4_smoke.json'), output)
            self.assertFalse(output.exists())
            prepare(Path('configs/e4_smoke.json'), output)
            self.assertTrue((output / 'manifest.json').exists())

    def test_journal_recovers_tail_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'smoke'
            prepare(Path('configs/e4_smoke.json'), output)
            run(output, max_games=1)
            journal = output / 'games.journal.jsonl'
            original = journal.read_bytes()
            journal.write_bytes(original + b'{partial')
            run(output, max_games=1)
            self.assertTrue((output / 'journal-interrupted-tail.bin').exists())
            with journal.open('ab') as stream:
                stream.write(original)
            with self.assertRaises(ValueError):
                run(output)

    def test_block_bootstrap_retains_pairs(self):
        from monopoly.experiments import STRATEGIES
        blocks = [{s: value for s in STRATEGIES} for value in (0, .25, 1)]
        result = bootstrap(blocks, 100, 123)
        self.assertEqual(result, bootstrap(blocks, 100, 123))
        self.assertTrue(all(delta == (0, 0) for rate, delta in result.values()))
        self.assertTrue(all(rate == result['no_trade'][0] for rate, delta in result.values()))

    @unittest.skipUnless(Path('results/e1_frozen').exists(), 'Local E1 evidence is optional in source-only checkouts')
    def test_frozen_e1_verified_and_tampering_rejected(self):
        report = verify(Path('results/e1_frozen'))
        self.assertEqual(report['games'], 3000)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / 'freeze.json').write_text(json.dumps({'files_sha256': {'games.csv': 'wrong'}}))
            (target / 'games.csv').write_text('changed')
            with self.assertRaises(AssertionError):
                verify(target)


if __name__ == '__main__':
    unittest.main()
