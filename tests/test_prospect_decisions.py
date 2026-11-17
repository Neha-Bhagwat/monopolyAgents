"""Prospect-policy checks and regressions for integrating the two branches."""

import subprocess
import sys
import unittest

from monopoly.advanced_decisions import ExpectimaxDecisions, ProspectTheoryDecisions
from monopoly.bots import ExpectedValueDecisions, baseline_bot, hard_bargainer_bot, risk_averse_bot
from monopoly.engine import Game
from monopoly.experiments import STRATEGIES, make_game
from monopoly.valuation import landing_probabilities


class ProspectIntegrationTests(unittest.TestCase):
    def test_modules_import_in_either_order_in_a_fresh_interpreter(self):
        for first, second in (('bots', 'advanced_decisions'), ('advanced_decisions', 'bots')):
            code = (
                f'import monopoly.{first}; import monopoly.{second}; '
                'from monopoly.bots import hard_bargainer_bot, risk_averse_bot; '
                'hard_bargainer_bot(); risk_averse_bot()'
            )
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_ordinary_games_use_both_advanced_policies_and_ablation_uses_ev(self):
        self.assertIsInstance(hard_bargainer_bot().decisions, ExpectimaxDecisions)
        self.assertIsInstance(risk_averse_bot().decisions, ProspectTheoryDecisions)
        self.assertIs(type(risk_averse_bot(use_prospect_theory=False).decisions), ExpectedValueDecisions)
        for strategy in STRATEGIES:
            for seat in range(4):
                self.assertIs(type(make_game(strategy, 0, seat).bots[seat].decisions), ExpectedValueDecisions)

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({'loss_aversion': .5}, {'alpha': 0}, {'alpha': 1.1},
                       {'probability_gamma': 0}, {'probability_gamma': 1.1}, {'capital_hurdle': -1}):
            with self.assertRaises(ValueError):
                ProspectTheoryDecisions(**kwargs)

    def test_value_probability_and_prospect_match_hand_calculation(self):
        policy = ProspectTheoryDecisions(loss_aversion=2, alpha=1, probability_gamma=1)
        self.assertEqual(policy._value(10), 10)
        self.assertEqual(policy._value(-10), -20)
        for p in (0, .25, .5, 1):
            self.assertAlmostEqual(policy._weight(p), p)
        # Expected gain 20 already excludes a $10 hurdle: success pays 110,
        # failure loses 10. Utility = .25*110 + .75*(-20) = 12.5.
        self.assertAlmostEqual(policy._prospect_score(20, 100, .25), 12.5)
        more_averse = ProspectTheoryDecisions(loss_aversion=4, alpha=1, probability_gamma=1)
        self.assertAlmostEqual(more_averse._prospect_score(20, 100, .25), -2.5)
        self.assertEqual(policy._prospect_score(0, 100, .25), float('-inf'))
        self.assertEqual(policy._prospect_score(20, 100, 0), float('-inf'))

    def test_rent_probability_accounts_for_set_completion_and_time_limit(self):
        game = Game([baseline_bot(), baseline_bot()])
        policy = ProspectTheoryDecisions()
        chances = landing_probabilities()
        self.assertAlmostEqual(policy._rent_probability(game.view(), 0, 3, buying=True),
                               1 - (1 - chances[3]) ** (149 / 2))
        game.owners[1] = 0
        self.assertAlmostEqual(policy._rent_probability(game.view(), 0, 3, buying=True),
                               1 - (1 - chances[1] - chances[3]) ** (149 / 2))
        game.turns = 149
        self.assertEqual(policy._rent_probability(game.view(), 0, 3, buying=True), 0)
        game.turns = 0
        game.players[1].bankrupt = True
        self.assertEqual(policy._rent_probability(game.view(), 0, 3, buying=True), 0)

    def test_neutral_prospect_matches_ev_and_preserves_state_and_reserves(self):
        game = Game([baseline_bot(), baseline_bot()])
        game.owners[1] = game.owners[3] = 0
        neutral = ProspectTheoryDecisions(loss_aversion=1, alpha=1, probability_gamma=1)
        expected_value = ExpectedValueDecisions()
        for turns in (0, 80, 149):
            for cash in (100, 249, 250, 299, 300, 1500):
                game.turns, game.players[0].cash = turns, cash
                before = game.view()
                legal = game._build_options(0)
                self.assertEqual(neutral.buy(before, 0, 5), expected_value.buy(before, 0, 5))
                self.assertEqual(neutral.build(before, 0, legal), expected_value.build(before, 0, legal))
                self.assertEqual(game.view(), before)


if __name__ == '__main__':
    unittest.main()
