import unittest
from dataclasses import replace

from monopoly.__main__ import DEMO_OFFER, set_up_demo_board
from monopoly.advanced_bargaining import (BayesianTrading, MinimaxRegretTrading, ZeuthenTrading,
                                         acceptance_likelihood, conflict_risk, minimax_regret_choice)
from monopoly.bots import (AcceptingBot, SimpleBot, adaptive_bot, baseline_bot,
                           hard_bargainer_bot, risk_averse_bot)
from monopoly.engine import Game
from monopoly.models import Offer, Reply, legal_offer
from monopoly.valuation import score_trade


def prepared_game(bots=None):
    game = Game(bots or [SimpleBot(), SimpleBot()])
    set_up_demo_board(game)
    game.turns = 109
    for player in game.players:
        player.cash = 150
    return game


class NewPolicyTests(unittest.TestCase):
    def test_zeuthen_risks_match_hand_calculation(self):
        self.assertAlmostEqual(conflict_risk(80, 60), 0.25)
        self.assertEqual(conflict_risk(80, -20), 1)
        self.assertEqual(conflict_risk(80, 100), 0)
        self.assertEqual(conflict_risk(0, 0), 1)

    def test_zeuthen_makes_minimal_sufficient_cash_concession(self):
        view = prepared_game().view()
        policy = ZeuthenTrading()
        opening = policy.propose(view, 0)
        self.assertEqual(opening, Offer(0, 1, (5,), (3,), ask_cash=90))
        incoming = Offer(1, 0, (3,), (5,), give_cash=40)
        counter = policy.respond(view, incoming, 2)
        self.assertEqual(counter, Offer(0, 1, (5,), (3,), ask_cash=65))
        old_us, old_them = score_trade(view, opening)
        their_demand, their_offer_to_us = score_trade(view, incoming)
        new_us, new_them = score_trade(view, counter)
        self.assertLess(conflict_risk(old_us.total, their_offer_to_us.total),
                        conflict_risk(their_demand.total, old_them.total))
        self.assertGreater(conflict_risk(new_us.total, their_offer_to_us.total),
                           conflict_risk(their_demand.total, new_them.total))
        self.assertLess(new_us.total, old_us.total)

    def test_zeuthen_holds_when_its_risk_is_larger_and_accepts_at_deadline(self):
        view = prepared_game().view()
        policy = ZeuthenTrading()
        opening = policy.propose(view, 0)
        bad = Offer(1, 0, (3,), (5,))
        self.assertEqual(policy.respond(view, bad, 2), opening)
        good = replace(bad, give_cash=40)
        self.assertIs(policy.respond(view, good, 3), Reply.ACCEPT)
        self.assertIs(policy.respond(view, bad, 3), Reply.REJECT)

    def test_zeuthen_old_turn_and_old_negotiation_demands_are_not_reused(self):
        game = prepared_game()
        policy = ZeuthenTrading()
        policy.propose(game.view(), 0)
        game.turns += 1
        incoming = Offer(1, 0, (3,), (5,), give_cash=40)
        self.assertEqual(policy.respond(game.view(), incoming, 2),
                         ZeuthenTrading().respond(game.view(), incoming, 2))
        self.assertEqual(policy.respond(game.view(), incoming, 1),
                         ZeuthenTrading().respond(game.view(), incoming, 1))
        policy.reset()
        self.assertEqual(policy.last_offer, {})

    def test_bayesian_likelihood_and_update_match_hand_calculation(self):
        view = prepared_game().view()
        policy = BayesianTrading()
        offer = Offer(0, 1, (5,), (3,), ask_cash=90)  # Recipient gains $14.41.
        self.assertAlmostEqual(policy.acceptance_probability(1, 14), 1.1 / 3)
        self.assertEqual(acceptance_likelihood(0, 0), 0.1)
        self.assertEqual(acceptance_likelihood(25, 25), 0.9)
        policy.observe_response(view, offer, False)
        for actual, expected in zip(policy.beliefs_for(1), (1 / 19, 9 / 19, 9 / 19)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(policy.beliefs_for(2), (1 / 3,) * 3)
        self.assertEqual(policy.observations, {1: 1})
        accepting = BayesianTrading()
        accepting.observe_response(view, offer, True)
        self.assertAlmostEqual(accepting.beliefs_for(1)[0], 9 / 11)

    def test_real_rejections_change_adaptive_offer(self):
        game = prepared_game([adaptive_bot(), SimpleBot()])
        policy = game.bots[0].trading
        before = game.view()
        for _ in range(2):
            offer = policy.propose(game.view(), 0)
            self.assertEqual(offer.ask_cash, 65)
            self.assertFalse(game.negotiate(offer))
        self.assertEqual(policy.observations, {1: 2})
        self.assertAlmostEqual(policy.beliefs_for(1)[2], 81 / 83)
        self.assertEqual(policy.propose(game.view(), 0).ask_cash, 15)
        self.assertEqual(game.view(), before)
        # Calling respond alone is not evidence about replies to its own offers.
        policy.respond(before, Offer(1, 0, (3,), (5,), give_cash=40), 1)
        self.assertEqual(policy.observations, {1: 2})

    def test_real_acceptance_is_observed_before_deeds_move(self):
        game = prepared_game([adaptive_bot(), AcceptingBot()])
        policy = game.bots[0].trading
        offer = Offer(0, 1, (5,), (3,), ask_cash=90)
        self.assertTrue(game.negotiate(offer))
        self.assertEqual(policy.observations, {1: 1})
        self.assertAlmostEqual(policy.beliefs_for(1)[0], 9 / 11)
        self.assertEqual(game.owners[3], 0)
        self.assertEqual(game.owners[5], 1)

    def test_legal_counter_is_one_rejection_and_invalid_reply_gives_no_evidence(self):
        class CounterBot(SimpleBot):
            def respond(self, view, offer, round_number):
                return Offer(1, 0, (3,), (5,), give_cash=90)
        game = prepared_game([adaptive_bot(), CounterBot()])
        self.assertTrue(game.negotiate(Offer(0, 1, (5,), (3,), ask_cash=65)))
        self.assertEqual(game.bots[0].trading.observations, {1: 1})
        class InvalidCounter(SimpleBot):
            def respond(self, view, offer, round_number):
                return Offer(1, 0, (23,))  # Does not own it.
        game = prepared_game([adaptive_bot(), InvalidCounter()])
        self.assertFalse(game.negotiate(Offer(0, 1, (5,), (3,), ask_cash=65)))
        self.assertEqual(game.bots[0].trading.observations, {})
        self.assertFalse(game.negotiate(Offer(0, 1, (23,))))
        self.assertEqual(game.bots[0].trading.observations, {})

    def test_minimax_regret_is_not_maximin_and_includes_walking_away(self):
        payoffs = [(0, 0, 0), (100, 0, 0), (60, 60, 0), (40, 40, 40)]
        # Column bests 100,60,40; row worst regrets 100,60,40,60.
        self.assertEqual(minimax_regret_choice(payoffs), 2)
        self.assertEqual(minimax_regret_choice([(0, 0), (-10, -20)]), 0)
        self.assertEqual(minimax_regret_choice([(0, 0), (0, 0)]), 0)
        self.assertEqual(minimax_regret_choice([(10, 0), (5, 5)]), 1)  # Same regret, safer payoff.
        for table in ([], [()], [(1,), (1, 2)]):
            with self.assertRaises(ValueError):
                minimax_regret_choice(table)

    def test_minimax_compares_certain_acceptance_with_uncertain_counters(self):
        view = prepared_game().view()
        policy = MinimaxRegretTrading()
        incoming = Offer(1, 0, (3,), (5,), give_cash=65)
        self.assertIs(policy.respond(view, incoming, 1), Reply.ACCEPT)
        self.assertIs(policy.respond(view, incoming, 3), Reply.ACCEPT)
        # Recipient gains, but the proposer loses: all new policies require both positive.
        self.assertIs(policy.respond(view, DEMO_OFFER, 3), Reply.REJECT)
        bad_for_recipient = replace(DEMO_OFFER, give_cash=0, ask_cash=150)
        self.assertIs(policy.respond(view, bad_for_recipient, 3), Reply.REJECT)

    def test_all_new_policies_respect_legality_deadlines_and_state(self):
        game = prepared_game()
        before = game.view()
        for policy in (ZeuthenTrading(), BayesianTrading(), MinimaxRegretTrading()):
            self.assertIs(policy.respond(before, Offer(0, 1, (23,)), 1), Reply.REJECT)
            offer = policy.propose(before, 0)
            self.assertTrue(legal_offer(before, offer))
            self.assertTrue(all(value.total > 0 for value in score_trade(before, offer)))
            for round_number in (1, 2, 3):
                reply = policy.respond(before, DEMO_OFFER, round_number)
                if isinstance(reply, Offer):
                    self.assertLess(round_number, 3)
                    self.assertTrue(legal_offer(before, reply))
                    self.assertEqual((reply.proposer, reply.recipient), (1, 0))
            self.assertEqual(game.view(), before)
            empty = Game([SimpleBot(), SimpleBot()]).view()
            self.assertIsNone(policy.propose(empty, 0))

    def test_all_new_policies_reject_the_dangerous_orange_sale(self):
        game = Game([baseline_bot() for _ in range(4)])
        game.turns, game.active_player = 59, 3
        game.owners[19] = 0
        game.owners[16] = game.owners[17] = 3
        game.players[0].cash, game.players[3].cash = 1290, 1731
        offer = Offer(3, 0, ask_properties=(19,), give_cash=300)
        for policy in (ZeuthenTrading(), BayesianTrading(), MinimaxRegretTrading()):
            self.assertIs(policy.respond(game.view(), offer, 3), Reply.REJECT)

    def test_new_games_clear_beliefs_and_demands_and_replay(self):
        bots = [hard_bargainer_bot(), adaptive_bot(), risk_averse_bot(), baseline_bot()]
        first = Game(bots, seed=42).play()
        self.assertTrue(bots[1].trading.observations)
        bots[1].trading.beliefs[99] = (1, 0, 0)
        second = Game(bots, seed=42)
        self.assertEqual(bots[1].trading.beliefs, {})
        self.assertEqual(bots[1].trading.observations, {})
        self.assertEqual(bots[0].trading.last_offer, {})
        self.assertEqual(first, second.play())


if __name__ == '__main__':
    unittest.main()
