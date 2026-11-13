"""Fast E1 structural tests. Place at tests/test_tournament.py.
Run: python -m unittest discover -s tests -p test_tournament.py
"""
import unittest

from monopoly.tournament import (
    FIVE_AGENTS, MCTS_IMPLEMENTED, _ci, make_schedule,
)


class TournamentScheduleTests(unittest.TestCase):
    def test_five_agent_schedule(self):
        games = make_schedule(
            include_mcts=False, seed_start=1000, seeds_per_matchup=50,
            depth=1, max_nodes=200,
        )
        self.assertEqual(len(games), 1000)
        self.assertEqual(len({g.game_id for g in games}), 1000)
        self.assertEqual(len({g.matchup_id for g in games}), 5)
        for name in FIVE_AGENTS:
            seats = [g.seats.index(name) for g in games if name in g.seats]
            self.assertEqual(len(seats), 800)
            self.assertEqual([seats.count(i) for i in range(4)], [200] * 4)

    @unittest.skipIf(MCTS_IMPLEMENTED, "Only applies before MCTS is completed")
    def test_mcts_refused_while_stubbed(self):
        with self.assertRaises(RuntimeError):
            make_schedule(include_mcts=True, seed_start=1000,
                          seeds_per_matchup=1, depth=1, max_nodes=200)

    def test_intervals(self):
        lo, hi = _ci(50, 200)
        self.assertLess(lo, 0.25)
        self.assertGreater(hi, 0.25)
        self.assertEqual(_ci(0, 10), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
