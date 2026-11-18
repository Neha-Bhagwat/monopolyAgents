import json
from pathlib import Path
import tempfile
import unittest

from analysis.e1_analysis import GameRow, _read_trade_results


class TradeAnalysisTests(unittest.TestCase):
    def test_tournament_and_flat_logs_use_their_own_seat_conventions(self):
        game = GameRow('game', 'lineup', 1, 0, ('a', 'b', 'c', 'd'), 'a', 1, 150,
                       'turn_limit', (100, 90, 80, 70), (0, 0, 0, 0), 2, 2, None)
        trades = [
            {'game_id': 'game', 'accepted_trade': {
                'initiated_by': 0, 'offer': {'proposer': 3, 'recipient': 0}}},
            {'game_id': 'game', 'initiated_by': 1, 'offer': {'proposer': 4, 'recipient': 1}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trades.jsonl'
            path.write_text(''.join(json.dumps(trade) + '\n' for trade in trades), encoding='utf-8')
            rows, warnings = _read_trade_results(path, [game])
        self.assertEqual(warnings, [])
        by_agent = {row['agent']: row for row in rows}
        self.assertEqual(by_agent['a']['accepted_trades_initiated'], 2)
        self.assertEqual(by_agent['a']['accepted_trades_as_final_recipient'], 2)
        self.assertEqual(by_agent['d']['accepted_trades_as_final_proposer'], 2)
        self.assertEqual(by_agent['b']['accepted_trades_involved'], 0)
