import unittest
from dataclasses import replace

from monopoly.__main__ import DEMO_OFFER, set_up_demo_board
from monopoly.bargaining import NashTrading, TitForTatTrading, candidate_offers
from monopoly.bots import baseline_bot, cooperator_bot, reciprocator_bot
from monopoly.engine import Game
from monopoly.models import Offer, Reply, legal_offer
from monopoly.valuation import score_trade


class BargainingTests(unittest.TestCase):
    def setUp(self):
        self.game = Game([baseline_bot() for _ in range(4)])
        set_up_demo_board(self.game)
        self.game.turns = 109
        for player in self.game.players:
            player.cash = 150

    def test_finder_checks_live_ownership_and_legality(self):
        empty = Game([baseline_bot(), baseline_bot()]).view()
        self.assertEqual(list(candidate_offers(empty, 0)), [])
        self.game.players[0].cash = 10
        view = self.game.view()
        offers = list(candidate_offers(view, 0))
        self.assertEqual(len(offers), len(set(offers)))
        self.assertTrue(all(offer.ask_properties == (3,) for offer in offers))
        self.assertTrue(all(legal_offer(view, offer) for offer in offers))
        self.assertIn(Offer(0, 1, (5,), (3,), ask_cash=40), offers)

    def test_cooperator_picks_largest_product_of_positive_gains(self):
        view = self.game.view()
        offer = NashTrading().propose(view, 0)
        ours, theirs = score_trade(view, offer)
        self.assertGreater(ours.total, 0)
        self.assertGreater(theirs.total, 0)
        for candidate in candidate_offers(view, 0):
            first, second = score_trade(view, candidate)
            if first.total > 0 and second.total > 0:
                self.assertGreaterEqual(ours.total * theirs.total, first.total * second.total)
        self.assertEqual(self.game.view(), view)

    def test_cooperator_counters_one_sided_offer_or_rejects_at_limit(self):
        self.game = Game([baseline_bot(), baseline_bot()])
        set_up_demo_board(self.game)
        self.game.turns = 109
        for player in self.game.players:
            player.cash = 150
        view = self.game.view()
        policy = NashTrading()
        counter = policy.respond(view, DEMO_OFFER, 1)
        self.assertIsInstance(counter, Offer)
        self.assertEqual((counter.proposer, counter.recipient), (1, 0))
        self.assertTrue(all(value.total > 0 for value in score_trade(view, counter)))
        self.assertIs(policy.respond(view, counter, 2), Reply.ACCEPT)
        self.assertIs(policy.respond(view, DEMO_OFFER, 3), Reply.REJECT)

    def test_reciprocator_uses_previous_offer_and_keeps_opponents_separate(self):
        game = Game([baseline_bot(), baseline_bot()])
        set_up_demo_board(game)
        game.turns = 109
        for player in game.players:
            player.cash = 150
        view = game.view()
        offer = replace(DEMO_OFFER, give_cash=0, ask_cash=90)
        unfair = replace(offer, ask_cash=150)
        neutral = TitForTatTrading()
        self.assertIs(neutral.respond(view, offer, 1), Reply.ACCEPT)

        policy = TitForTatTrading()
        policy.respond(view, unfair, 1)
        self.assertEqual(policy.minimum_gain(0), 25)
        self.assertEqual(policy.minimum_gain(2), 10)
        counter = policy.respond(view, offer, 1)
        self.assertEqual(counter, Offer(1, 0, (3,), (5,), give_cash=65))

        fair = replace(offer, ask_cash=40)
        self.assertIs(policy.respond(view, fair, 1), Reply.ACCEPT)
        self.assertEqual(policy.minimum_gain(0), 0)
        self.assertEqual(TitForTatTrading().minimum_gain(0), 10)

    def test_both_bots_complete_live_games_and_find_deals(self):
        for seed, proposer in ((0, 0), (6, 1)):
            with self.subTest(seed=seed):
                game = Game([cooperator_bot(), reciprocator_bot(), baseline_bot(), baseline_bot()], seed)
                self.assertTrue(all(owner is None for owner in game.owners))
                result = game.play()
                self.assertLessEqual(result.turns, 150)
                self.assertEqual(result.trades_accepted, len(game.trade_history))
                self.assertTrue(any(
                    record.offer.proposer == proposer for record in game.trade_history
                ))
