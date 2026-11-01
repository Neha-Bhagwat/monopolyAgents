import unittest
from dataclasses import replace
from unittest.mock import patch

from monopoly.advanced_decisions import ExpectimaxDecisions, _assets
from monopoly.board import BOARD, CHANCE, CHEST, JAIL, CardMove
from monopoly.bots import ExpectedValueDecisions, baseline_bot, hard_bargainer_bot
from monopoly.engine import Game
from monopoly.experiments import make_game


class ExpectimaxTests(unittest.TestCase):
    def game(self, count=2):
        return Game([baseline_bot() for _ in range(count)])

    def test_configuration_rejects_unbounded_or_invalid_search(self):
        for depth in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                ExpectimaxDecisions(depth=depth)
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ExpectimaxDecisions(max_nodes=budget)

    def test_chance_probabilities_and_cash_match_hand_calculation(self):
        game = self.game()
        policy = ExpectimaxDecisions(depth=1)
        outcomes = list(policy._outcomes(game.view()))
        self.assertAlmostEqual(sum(p for p, _, _ in outcomes), 1)
        # From Go: tax on 4 (3/36), Chest on 2 or 11 (3/36), Chance on 8 (5/36).
        chest_cash = (200 + 0 + 100 - 50 + 25 + 50) / 6
        chance_cash = (200 + 0 + 50 - 50 + 100 - 25) / 6
        expected = 1500 - 200 * 3 / 36 + chest_cash * 3 / 36 + chance_cash * 5 / 36
        with patch.object(policy, '_after_landing', side_effect=lambda v, p, d, b, s: v.players[0].cash):
            self.assertAlmostEqual(policy._search(game.view(), 0, 1, [0]), expected)
        jailed = sum(p for p, v, _ in outcomes if v.players[0].jailed)
        self.assertAlmostEqual(jailed, (3 + 5) / (36 * 6))

    def test_chance_states_match_engine_rules_across_board(self):
        # Disable purchases and builds so this compares movement/landing alone.
        class Passive:
            def buy(self, *args): return False
            def build(self, *args): return None
            def propose(self, *args): return None
        for start in range(len(BOARD)):
            source = Game([Passive(), Passive()])
            source.players[0].position = start
            source.players[0].cash = 100
            source.owners[1] = source.owners[3] = 1
            source.buildings[1] = source.buildings[3] = 2
            actual = {}
            for probability, state, skipped in ExpectimaxDecisions._outcomes(source.view()):
                actual[state] = actual.get(state, 0) + probability
            expected = {}
            for first in range(1, 7):
                for second in range(1, 7):
                    kind = BOARD[(start + first + second) % len(BOARD)].kind.name
                    cards = CHANCE if kind == 'CHANCE' else CHEST if kind == 'CHEST' else (None,)
                    for card in cards:
                        game = Game([Passive(), Passive()])
                        game.players[0].position = start
                        game.players[0].cash = 100
                        game.owners[:] = source.owners
                        game.buildings[:] = source.buildings
                        if kind == 'CHANCE': game.chance = [card]
                        if kind == 'CHEST': game.chest = [card]
                        with patch.object(game.random, 'randint', side_effect=[first, second]):
                            game._take_turn(0)
                        state = game.view()
                        expected[state] = expected.get(state, 0) + 1 / (36 * len(cards))
            self.assertEqual(actual.keys(), expected.keys())
            for state in actual:
                self.assertAlmostEqual(actual[state], expected[state])

    def test_jail_skips_actions_and_eliminated_seats_are_skipped(self):
        game = self.game(3)
        game.players[0].jailed = True
        game.players[0].position = JAIL
        [(probability, state, skipped)] = list(ExpectimaxDecisions._outcomes(game.view()))
        self.assertEqual(probability, 1)
        self.assertTrue(skipped)
        self.assertFalse(state.players[0].jailed)
        self.assertEqual(state.players[0].position, JAIL)
        game.players[1].bankrupt = True
        policy = ExpectimaxDecisions()
        with patch.object(policy, '_search', return_value=0) as search:
            policy._next_turn(game.view(), 0, 2, [0])
            self.assertEqual(search.call_args.args[0].active_player, 2)
            self.assertEqual(search.call_args.args[0].turns, 1)

    def test_actions_are_legal_repeatable_and_do_not_touch_live_state(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        view, rng = game.view(), game.random.getstate()
        decks = (list(game.chance), list(game.chest))
        policy = ExpectimaxDecisions()
        legal = game._build_options(0)
        action = policy.build(view, 0, legal)
        self.assertIn(action, (None,) + legal)
        self.assertEqual(action, policy.build(view, 0, legal))
        self.assertIsInstance(policy.buy(view, 0, 5), bool)
        self.assertEqual(game.view(), view)
        self.assertEqual(game.random.getstate(), rng)
        self.assertEqual((game.chance, game.chest), decks)
        self.assertFalse(policy.buy(view, 0, 1))  # Already owned.
        self.assertFalse(policy.buy(view, 0, 0))  # Not a property.
        self.assertIsNone(policy.build(view, 0, ()))
        game.players[0].cash = 149
        self.assertFalse(policy.buy(game.view(), 0, 5))
        self.assertIsNone(policy.build(game.view(), 0, legal))

    def test_final_turn_does_not_spend_on_unrealizable_rent(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        game.turns = game.turn_limit - 1
        policy = ExpectimaxDecisions()
        self.assertFalse(policy.buy(game.view(), 0, 5))
        self.assertIsNone(policy.build(game.view(), 0, game._build_options(0)))

    def test_search_budget_and_bankruptcy_stop_expansion(self):
        game = self.game()
        policy = ExpectimaxDecisions(depth=3, max_nodes=2)
        budget = [0]
        policy._search(game.view(), 0, 3, budget)
        self.assertEqual(budget[0], 2)
        game.players[0].bankrupt = True
        with patch.object(policy, '_outcomes', side_effect=AssertionError('Should not expand')):
            self.assertEqual(policy._search(game.view(), 0, 3, [0]), -100_000)

    def test_root_max_selects_best_action_and_ties_skip(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        policy = ExpectimaxDecisions()
        with patch.object(policy, '_next_turn', side_effect=lambda v, p, d, b: v.buildings[3] * 10):
            self.assertEqual(policy.build(game.view(), 0, (1, 3)), 3)
        with patch.object(policy, '_finish_build', return_value=0):
            self.assertFalse(policy.buy(game.view(), 0, 5))
        with patch.object(policy, '_finish_build', side_effect=lambda v, p, d, b: int(v.owners[5] == p)):
            self.assertTrue(policy.buy(game.view(), 0, 5))

    def test_tournament_uses_search_but_ablation_keeps_expected_value(self):
        self.assertIsInstance(hard_bargainer_bot().decisions, ExpectimaxDecisions)
        for seat in range(4):
            self.assertIs(type(make_game('hard_bargainer', 0, seat).bots[seat].decisions), ExpectedValueDecisions)

    def test_cached_assets_refresh_for_ownership_and_building_changes(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        game.buildings[1] = game.buildings[3] = 1
        first = _assets(tuple(game.owners), tuple(game.buildings), 2)
        self.assertEqual(first[0][0], game.net_worth(0) - game.players[0].cash)
        self.assertEqual(first[2], (1, 0))
        self.assertEqual(first[3][0], game._build_options(0))
        game.buildings[3] += 1
        changed = _assets(tuple(game.owners), tuple(game.buildings), 2)
        self.assertEqual(changed[0][0], game.net_worth(0) - game.players[0].cash)
        self.assertGreater(changed[1][0], first[1][0])
        self.assertEqual(changed[3][0], game._build_options(0))
        game.owners[1] = game.owners[3] = 1
        transferred = _assets(tuple(game.owners), tuple(game.buildings), 2)
        self.assertEqual(transferred[0], (0, changed[0][0]))
        self.assertEqual(transferred[2], (0, 1))
        self.assertEqual(first[2], (1, 0))  # Cached tuples were not mutated.

    def test_transposition_hit_preserves_logical_cost_and_tight_budget(self):
        policy = ExpectimaxDecisions()
        view = self.game().view()
        initial_budget = [0]
        value = policy._search(view, 0, 2, initial_budget)
        with patch.object(policy, '_outcomes', side_effect=AssertionError('Cache hit should not expand')):
            repeat_budget = [0]
            self.assertEqual(policy._search(view, 0, 2, repeat_budget), value)
            self.assertEqual(repeat_budget, initial_budget)
        # A fully cached result must not bypass a smaller remaining budget.
        tight, reference_budget = [999], [999]
        actual = policy._search(view, 0, 2, tight)
        expected = ExpectimaxDecisions()._search(view, 0, 2, reference_budget)
        self.assertEqual(actual, expected)
        self.assertEqual(tight, reference_budget)

    def test_search_cache_distinguishes_perspective_depth_and_state(self):
        policy = ExpectimaxDecisions()
        view = self.game().view()
        policy._search(view, 0, 2, [0])
        moved = replace(view.players[1], position=15, cash=200)
        variations = [
            (view, 1, 2), (view, 0, 1),
            (replace(view, active_player=1), 0, 2),
            (replace(view, turns=149), 0, 2),
            (replace(view, turn_limit=2), 0, 2),
            (replace(view, players=(view.players[0], moved)), 0, 2),
        ]
        for state, player, depth in variations:
            actual_budget, expected_budget = [0], [0]
            actual = policy._search(state, player, depth, actual_budget)
            expected = ExpectimaxDecisions()._search(state, player, depth, expected_budget)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_budget, expected_budget)

    def test_cached_and_uncached_search_match_even_with_cutoffs(self):
        class NoTable(dict):
            def get(self, key): return None
            def __setitem__(self, key, value): pass
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        game.players[1].position = 18
        game.turns = 140
        for depth, limit in ((1, 1000), (2, 1), (2, 5), (2, 1000), (3, 20), (3, 1000)):
            cached = ExpectimaxDecisions(depth, limit)
            uncached = ExpectimaxDecisions(depth, limit)
            uncached._transpositions = NoTable()
            left, right = [0], [0]
            self.assertEqual(cached._search(game.view(), 0, depth, left),
                             uncached._search(game.view(), 0, depth, right))
            self.assertEqual(left, right)


if __name__ == '__main__':
    unittest.main()
