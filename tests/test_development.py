import unittest
from monopoly.bargaining import NashTrading, TitForTatTrading
from monopoly.bots import baseline_bot
from monopoly.engine import Game
from monopoly.models import Offer, Reply
from monopoly.valuation import _development_income, landing_probabilities, score_trade


class DevelopmentTests(unittest.TestCase):
    def brown_game(self):
        game = Game([baseline_bot(), baseline_bot()])
        game.owners[1] = game.owners[3] = 0
        game.players[0].cash = 300  # $250 reserve leaves exactly one $50 house.
        game.turns = 109
        return game

    def forecast(self, game):
        view = game.view()
        return _development_income(view, view.owners, tuple(p.cash for p in view.players))

    def test_one_affordable_house_matches_hand_calculation(self):
        game = self.brown_game()
        before = game.view()
        income = self.forecast(game)
        # Player 0 next builds in two individual turns. Baltic rent rises 8 -> 20.
        expected = landing_probabilities()[3] * 12 * 38 / 2
        self.assertAlmostEqual(income[0], expected)
        self.assertEqual(income[1], 0)
        self.assertEqual(game.view(), before)
        game.players[0].cash = 299
        self.assertEqual(self.forecast(game), (0, 0))

    def test_budget_even_building_and_one_upgrade_per_turn(self):
        game = self.brown_game()
        game.turns = 49  # Enough time for both houses to clear the 10% hurdle.
        game.players[0].cash = 350
        chances = landing_probabilities()
        # Baltic first, then Mediterranean; cannot add Baltic's second house yet.
        expected = chances[3] * 12 * 98 / 2 + chances[1] * 6 * 96 / 2
        self.assertAlmostEqual(self.forecast(game)[0], expected)
        game.buildings[3] = 1
        game.players[0].cash = 300
        self.assertAlmostEqual(self.forecast(game)[0], chances[1] * 6 * 98 / 2)

    def test_jail_turn_limit_and_hotel_cap(self):
        game = self.brown_game()
        game.players[0].jailed = True
        self.assertAlmostEqual(self.forecast(game)[0], landing_probabilities()[3] * 12 * 36 / 2)
        game.players[0].jailed = False
        game.turns = 148  # Only opponent's turn remains; owner cannot build.
        self.assertEqual(self.forecast(game), (0, 0))
        game.turns = 149
        self.assertEqual(self.forecast(game), (0, 0))
        game.turns = 109
        game.buildings[1] = game.buildings[3] = 5
        self.assertEqual(self.forecast(game), (0, 0))

    def test_eliminated_seats_are_skipped(self):
        game = Game([baseline_bot() for _ in range(4)])
        game.players[1].bankrupt = game.players[3].bankrupt = True
        game.owners[1] = game.owners[3] = 0
        game.players[0].cash = 300
        game.turns = 109
        self.assertAlmostEqual(self.forecast(game)[0], landing_probabilities()[3] * 12 * 38 / 2)
        self.assertEqual(self.forecast(game)[1:], (0, 0, 0))

    def test_cash_transfer_can_fund_an_existing_set(self):
        game = self.brown_game()
        game.players[0].cash = 250
        offer = Offer(1, 0, give_cash=50)
        giver, receiver = score_trade(game.view(), offer)
        self.assertGreater(receiver.development_change, 0)
        self.assertAlmostEqual(giver.development_change + receiver.development_change, 0)

    def test_dangerous_orange_sale_is_rejected_without_mutating_game(self):
        game = Game([baseline_bot() for _ in range(4)])
        game.turns, game.active_player = 59, 3
        game.owners[19] = 0
        game.owners[16] = game.owners[17] = 3
        game.players[0].cash, game.players[3].cash = 1290, 1731
        offer = Offer(3, 0, ask_properties=(19,), give_cash=300)
        before = game.view()
        buyer, seller = score_trade(before, offer)
        self.assertAlmostEqual(seller.total - seller.development_change, 3.509680339713927, places=5)
        self.assertLess(seller.total, 0)
        self.assertGreater(buyer.development_change, 0)
        for policy in (NashTrading(), TitForTatTrading()):
            self.assertIs(policy.respond(before, offer, 3), Reply.REJECT)
        self.assertEqual(game.view(), before)
        self.assertEqual(score_trade(before, offer), (buyer, seller))

    def test_opponent_set_completion_is_still_allowed_when_it_is_worthwhile(self):
        game = Game([baseline_bot() for _ in range(4)])
        game.owners[1] = 0
        game.owners[3] = 1
        game.players[1].cash = 200
        offer = Offer(1, 0, ask_properties=(1,), give_cash=85)
        buyer, seller = score_trade(game.view(), offer)
        self.assertEqual(buyer.sets_gained, ('brown',))
        self.assertEqual(seller.sets_gained, ())
        self.assertEqual(buyer.development_change, 0)
        self.assertGreater(buyer.total, 0)
        self.assertGreater(seller.total, 0)
        self.assertIs(NashTrading().respond(game.view(), offer, 3), Reply.ACCEPT)
        # With a rich buyer the exact same cheap deed sale is more dangerous.
        game.players[1].cash = 1500
        self.assertLess(score_trade(game.view(), offer)[1].total, seller.total)


if __name__ == '__main__':
    unittest.main()
