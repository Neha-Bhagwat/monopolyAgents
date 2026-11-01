import csv
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from monopoly.bots import ExpectedValueDecisions
from monopoly.experiments import PILOT_STRATEGIES, STRATEGIES, make_game, replay, run_case, run_pilot, summarize


class ExperimentTests(unittest.TestCase):
    def test_all_variants_have_the_same_buying_policy(self):
        for strategy in STRATEGIES:
            for seat in range(4):
                game = make_game(strategy, 0, seat)
                self.assertIs(type(game.bots[seat].decisions), ExpectedValueDecisions)
                self.assertEqual(len({id(bot) for bot in game.bots}), 4)

    def test_saved_pilot_balances_seats_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "pilot"
            summary = run_pilot(folder, lambda _: None)
            with (folder / "games.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 120)
            self.assertEqual(len({row["game_id"] for row in rows}), 120)
            self.assertEqual(Counter((r["strategy"], r["test_seat"]) for r in rows),
                             Counter({(s, str(p)): 10 for s in PILOT_STRATEGIES for p in range(1, 5)}))
            self.assertEqual(summary, summarize(rows))
            with zipfile.ZipFile(folder / "source.zip") as archive:
                self.assertEqual(archive.read("monopoly/experiments.py"),
                                 Path("monopoly/experiments.py").read_bytes())
            schedules = [{(r["seed"], r["test_seat"]) for r in rows if r["strategy"] == s}
                         for s in PILOT_STRATEGIES]
            self.assertTrue(all(schedule == schedules[0] for schedule in schedules))
            selected = next(r for r in rows if int(r["test_trades_accepted"]) > 0)
            messages = []
            replay(folder, selected["game_id"], messages.append)
            self.assertTrue(messages[-1].startswith("Replay matched"))
            self.assertTrue(all(int(r["test_trades_accepted"]) == 0 for r in rows
                                if r["strategy"] == "no_trade"))
            with self.assertRaises(FileExistsError):
                run_pilot(folder, lambda _: None)
            with self.assertRaises(ValueError):
                replay(folder, "unknown")
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["source_sha256"] = "changed"
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different code"):
                replay(folder, selected["game_id"])

    def test_failed_run_does_not_publish_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "pilot"
            with patch("monopoly.experiments.run_case", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    run_pilot(folder)
            self.assertFalse(folder.exists())

    def test_case_is_deterministic(self):
        for strategy in STRATEGIES:
            self.assertEqual(run_case(strategy, 8, 2), run_case(strategy, 8, 2))

    def test_custom_schedule_is_saved_and_invalid_schedules_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'custom'
            summary = run_pilot(folder, lambda _: None, seed_start=60, seed_count=1)
            manifest = json.loads((folder / 'manifest.json').read_text())
            self.assertEqual(manifest['seeds'], [60])
            self.assertEqual(manifest['games'], 12)
            self.assertTrue(all(row['games'] == 4 for row in summary))
            replay(folder, 'cooperator-seed60-seat1', lambda _: None)
            for start, count in ((-1, 1), (0, 0), (0, -1), (True, 1), (0, 1.5)):
                with self.assertRaises(ValueError):
                    run_pilot(Path(directory) / 'invalid', seed_start=start, seed_count=count)

    def test_all_six_variants_can_be_saved_and_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'all'
            summary = run_pilot(folder, lambda _: None, seed_start=110, seed_count=1, strategies=STRATEGIES)
            manifest = json.loads((folder / 'manifest.json').read_text())
            self.assertEqual(manifest['games'], 24)
            self.assertEqual(manifest['strategies'], list(STRATEGIES))
            self.assertEqual(len(summary), 6)
            for strategy in STRATEGIES:
                replay(folder, f'{strategy}-seed110-seat1', lambda _: None)
            for strategies in ((), ('unknown',), ('adaptive', 'adaptive')):
                with self.assertRaises(ValueError):
                    run_pilot(Path(directory) / 'invalid', strategies=strategies)
