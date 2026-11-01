import unittest

from monopoly.board import BOARD, GROUPS, Kind
from monopoly.bots import (adaptive_bot, baseline_bot, cooperator_bot, hard_bargainer_bot,
                           reciprocator_bot, risk_averse_bot)
from monopoly.engine import Game
from monopoly.models import Offer, legal_offer


class AuditTests(unittest.TestCase):
    def test_bad_offers_are_rejected_without_crashing_or_mutating(self):
        game = Game([baseline_bot(), baseline_bot()])
        before = game.view()
        for offer in (None, "accept", Offer(0, 1, ([1],)), Offer(0, 1, None),
                      Offer(0, 1, (True,)), Offer(0, 1, give_cash=1.5),
                      Offer(-1, 1, give_cash=1)):
            with self.subTest(offer=offer):
                self.assertFalse(legal_offer(before, offer))
                self.assertFalse(game.negotiate(offer))
                self.assertEqual(before, game.view())

    def test_new_game_clears_reciprocator_memory_and_replays(self):
        bots = [cooperator_bot(), reciprocator_bot(), baseline_bot(), baseline_bot()]
        first = Game(bots, seed=6).play()
        bots[1].trading.last_offer_fair[3] = False
        second = Game(bots, seed=6)
        self.assertEqual(bots[1].trading.last_offer_fair, {})
        self.assertEqual(first, second.play())
        with self.assertRaises(ValueError):
            Game([bots[0], bots[0]])

    def test_complete_games_preserve_state_on_every_turn(self):
        class CheckedGame(Game):
            def _take_turn(inner, player):
                super()._take_turn(player)
                for seat, state in enumerate(inner.players):
                    self.assertGreaterEqual(state.cash, 0)
                    self.assertTrue(0 <= state.position < len(BOARD))
                    if state.bankrupt:
                        self.assertNotIn(seat, inner.owners)
                for square, owner in enumerate(inner.owners):
                    level = inner.buildings[square]
                    self.assertIn(level, range(6))
                    if owner is not None:
                        self.assertIs(BOARD[square].kind, Kind.PROPERTY)
                        self.assertFalse(inner.players[owner].bankrupt)
                    if level:
                        self.assertIsNotNone(owner)
                        group = GROUPS[BOARD[square].property.group]
                        self.assertTrue(all(inner.owners[i] == owner for i in group))
                        self.assertLessEqual(max(inner.buildings[i] for i in group) -
                                             min(inner.buildings[i] for i in group), 1)
                self.assertEqual(inner.trades_accepted, len(inner.trade_history))

        for seed in range(30):
            with self.subTest(seed=seed):
                result = CheckedGame(
                    [cooperator_bot(), reciprocator_bot(), baseline_bot(), baseline_bot()], seed,
                ).play()
                self.assertFalse(result.bankrupt[result.winner])
            with self.subTest(seed=seed, lineup='step6'):
                result = CheckedGame(
                    [hard_bargainer_bot(), adaptive_bot(), risk_averse_bot(), baseline_bot()], seed,
                ).play()
                self.assertFalse(result.bankrupt[result.winner])
