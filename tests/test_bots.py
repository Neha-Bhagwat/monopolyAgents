import unittest

from monopoly.bots import ExpectedValueDecisions, NoTrading, PolicyBot, SetCompletionTrading, baseline_bot
from monopoly.engine import Game
from monopoly.models import Offer, Reply
from monopoly.valuation import building_gain, landing_probabilities, purchase_gain


class BotTests(unittest.TestCase):
    def test_baselines_start_from_go_and_find_a_trade(self):
        game = Game([baseline_bot() for _ in range(4)], seed=42)
        self.assertTrue(all(owner is None for owner in game.owners))
        result = game.play()
        repeat = Game([baseline_bot() for _ in range(4)], seed=42)
        self.assertEqual(result, repeat.play())
        self.assertEqual(game.trade_history, repeat.trade_history)
        self.assertGreater(result.trades_accepted, 0)
        self.assertEqual(result.trades_accepted, len(game.trade_history))

    def test_baseline_bargains_from_live_ownership(self):
        game = Game([baseline_bot(), baseline_bot()])
        policy = SetCompletionTrading()
        self.assertIsNone(policy.propose(game.view(), 0))
        game.owners[1], game.owners[5] = 0, 1
        self.assertIsNone(policy.propose(game.view(), 0))  # Baltic still belongs to the bank.

        game.owners[3] = 1
        view = game.view()
        self.assertEqual(policy.propose(view, 0), Offer(0, 1, ask_properties=(3,), give_cash=60))
        self.assertIs(policy.respond(view, Offer(0, 1, ask_properties=(3,), give_cash=60), 1), Reply.REJECT)

        game.owners[5], game.owners[7], game.owners[9] = 0, 1, 1
        view = game.view()
        offer = Offer(0, 1, (5,), (3,), ask_cash=40)
        self.assertEqual(policy.propose(view, 0), offer)
        self.assertIs(policy.respond(view, offer, 1), Reply.ACCEPT)
        self.assertEqual(game.view(), view)

        game.players[1].cash = 189
        self.assertIs(policy.respond(game.view(), offer, 1), Reply.REJECT)
        self.assertNotEqual(policy.propose(game.view(), 0), offer)

    def test_baseline_rejects_breaking_one_set_to_finish_another(self):
        game = Game([baseline_bot(), baseline_bot()])
        game.owners[1] = game.owners[3] = game.owners[7] = game.owners[9] = 1
        game.owners[5] = 0
        offer = Offer(0, 1, (5,), (3,))
        self.assertTrue(game.legal_offer(offer))
        self.assertIs(SetCompletionTrading().respond(game.view(), offer, 1), Reply.REJECT)

    def test_expected_value_uses_rent_reserves_and_time_left(self):
        game = Game([baseline_bot() for _ in range(4)])
        decisions = ExpectedValueDecisions()
        chances = landing_probabilities()
        future = 149 * 3 / 4
        self.assertAlmostEqual(purchase_gain(game.view(), 0, 3), chances[3] * 4 * future - 6)
        self.assertTrue(decisions.buy(game.view(), 0, 3))
        game.players[0].cash = 209
        self.assertFalse(decisions.buy(game.view(), 0, 3))

        game.players[0].cash = 1500
        game.owners[1] = game.owners[3] = 0
        self.assertAlmostEqual(building_gain(game.view(), 3), chances[3] * (20 - 8) * future - 5)
        self.assertEqual(decisions.build(game.view(), 0, (1, 3)), 3)
        game.buildings[1] = game.buildings[3] = 1
        self.assertAlmostEqual(building_gain(game.view(), 3), chances[3] * (60 - 20) * future - 5)
        self.assertEqual(decisions.build(game.view(), 0, (1, 3)), 3)
        game.players[0].cash = 299
        self.assertIsNone(decisions.build(game.view(), 0, (1, 3)))
        game.players[0].cash = 1500
        game.turns = 149
        self.assertFalse(decisions.buy(game.view(), 0, 5))
        self.assertIsNone(decisions.build(game.view(), 0, (1, 3)))

    def test_swapping_trading_keeps_buy_and_build_decisions(self):
        game = Game([baseline_bot(), baseline_bot()])
        game.owners[1], game.owners[5] = 0, 0
        game.owners[3], game.owners[7], game.owners[9] = 1, 1, 1
        view = game.view()
        trading = PolicyBot(ExpectedValueDecisions(), SetCompletionTrading())
        no_trade = PolicyBot(ExpectedValueDecisions(), NoTrading())
        self.assertIsNotNone(trading.propose(view, 0))
        self.assertIsNone(no_trade.propose(view, 0))
        self.assertTrue(trading.buy(view, 0, 10))
        self.assertEqual(trading.buy(view, 0, 10), no_trade.buy(view, 0, 10))
        game.owners[1] = 1
        game.buildings[1] = game.buildings[3] = 1
        view = game.view()
        self.assertEqual(trading.build(view, 1, (1, 3)), 3)
        self.assertEqual(trading.build(view, 1, (1, 3)), no_trade.build(view, 1, (1, 3)))
