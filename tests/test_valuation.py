import unittest
from collections import Counter

from monopoly.__main__ import DEMO_OFFER, set_up_demo_board
from monopoly.board import BOARD, JAIL
from monopoly.bots import SimpleBot
from monopoly.engine import Game
from monopoly.models import Offer
from monopoly.valuation import IN_JAIL, SET_BONUS, landing_probabilities, movement_table, score_trade


class ValuationTests(unittest.TestCase):
    def test_movement_table_matches_dice_cards_and_jail(self):
        table = movement_table()
        self.assertEqual(len(table), 25)
        for row in table:
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertAlmostEqual(table[0][5], 4 / 36)
        self.assertAlmostEqual(table[0][8], (5 / 36) * (4 / 6))
        # From Community Chest at index 11: roll 7 to Go to Jail,
        # or roll 4 to Chance and draw its one Go to Jail card.
        self.assertAlmostEqual(table[11][IN_JAIL], (6 + 3 / 6) / 36)
        self.assertEqual(table[IN_JAIL][JAIL], 1.0)
        self.assertAlmostEqual(sum(landing_probabilities()), 1.0)

    def test_property_chances_match_engine_movement(self):
        game = Game([SimpleBot(), SimpleBot()], seed=123)
        visits = Counter()
        turns = 120_000
        for _ in range(turns):
            game._take_turn(0)
            visits[game.players[0].position] += 1
            # Keep a player solvent without changing the engine's dice or cards.
            game.players[0].cash = 1500
        for square, expected in enumerate(landing_probabilities()):
            if BOARD[square].property is not None:
                with self.subTest(square=BOARD[square].name):
                    self.assertAlmostEqual(visits[square] / turns, expected, delta=0.004)

    def test_demo_offer_totals_match_a_hand_calculation(self):
        game = Game([SimpleBot(), SimpleBot()])
        set_up_demo_board(game)
        game.turns = 109
        # No building budget: isolate the original cash/deed/current-rent math.
        for player in game.players:
            player.cash = 150
        before = game.view()
        proposer, recipient = score_trade(before, DEMO_OFFER)
        chances = landing_probabilities()
        brown_rent_change = 20 * (2 * chances[1] + 8 * chances[3] - 6 * chances[5])
        blue_rent_change = 20 * (
            12 * chances[5] + 6 * chances[7] + 8 * chances[9] - 4 * chances[3]
        )
        self.assertEqual(proposer.sets_gained, ("brown",))
        self.assertEqual(recipient.sets_gained, ("light blue",))
        self.assertAlmostEqual(proposer.total, -25 - 40 + brown_rent_change - blue_rent_change + SET_BONUS)
        self.assertAlmostEqual(recipient.total, 25 + 40 + blue_rent_change - brown_rent_change + SET_BONUS)
        self.assertAlmostEqual(proposer.rent_change + recipient.rent_change, 0)
        self.assertEqual((proposer.development_change, recipient.development_change), (0, 0))
        self.assertEqual(game.view(), before)
        self.assertEqual(game.trade_history, [])

    def test_breaking_a_set_loses_bonus_until_the_final_turn(self):
        game = Game([SimpleBot(), SimpleBot()])
        game.owners[1] = game.owners[3] = 0
        offer = Offer(0, 1, (3,), ask_cash=60)
        proposer, _ = score_trade(game.view(), offer)
        self.assertEqual(proposer.sets_lost, ("brown",))
        self.assertEqual(proposer.set_bonus_change, -SET_BONUS)
        game.turns = 149
        proposer, recipient = score_trade(game.view(), offer)
        self.assertEqual((proposer.rent_change, recipient.rent_change), (0.0, 0.0))
        self.assertEqual((proposer.development_change, recipient.development_change), (0.0, 0.0))
        self.assertEqual((proposer.total, recipient.total), (0.0, 0.0))

    def test_illegal_offer_cannot_be_scored(self):
        game = Game([SimpleBot(), SimpleBot()])
        offer = Offer(0, 1, (3,))
        self.assertFalse(game.legal_offer(offer))
        with self.assertRaisesRegex(ValueError, "legal offer"):
            score_trade(game.view(), offer)
