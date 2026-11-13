"""MCTS search, engine equivalence, isolation, and experiment integration."""

import unittest
from random import Random
from unittest.mock import patch

from monopoly.advanced_decisions import MCTSDecisions, _MCTSNode
from monopoly.board import BOARD, CHANCE, CHEST, JAIL, Kind
from monopoly.bots import (ExpectedValueDecisions, SimpleBot, adaptive_bot,
                           baseline_bot, hard_bargainer_bot, risk_averse_bot)
from monopoly.engine import Game
from monopoly.experiments import STRATEGIES, make_game


class MCTSTests(unittest.TestCase):
    def game(self, count=2):
        return Game([baseline_bot() for _ in range(count)])

    def test_configuration_is_bounded_and_validated(self):
        for kwargs in ({'iterations': 0}, {'iterations': True}, {'iterations': 1.5},
                       {'depth': 0}, {'depth': 151}, {'depth': True}, {'depth': 1.5},
                       {'exploration': -1}, {'exploration': float('nan')},
                       {'exploration': float('inf')}, {'exploration': True},
                       {'seed': True}, {'seed': 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MCTSDecisions(**kwargs)

    def test_sampled_movement_matches_engine_for_every_dice_pair_and_card(self):
        class Passive(SimpleBot):
            def buy(self, *args): return False
            def build(self, *args): return None

        for start in range(len(BOARD)):
            for first in range(1, 7):
                for second in range(1, 7):
                    kind = BOARD[(start + first + second) % len(BOARD)].kind
                    cards = CHANCE if kind is Kind.CHANCE else CHEST if kind is Kind.CHEST else (None,)
                    for card in cards:
                        game = Game([Passive(), Passive()])
                        game.players[0].position = start
                        game.players[0].cash = 15
                        game.owners[1] = game.owners[3] = 0
                        game.owners[16] = game.owners[17] = game.owners[19] = 1
                        game.buildings[16] = game.buildings[17] = game.buildings[19] = 2
                        rng = Random(0)
                        before = game.view()
                        with patch.object(rng, 'randint', side_effect=(first, second)), \
                             patch.object(rng, 'choice', return_value=card):
                            actual, skipped = MCTSDecisions._sample_move(before, rng)
                        if kind is Kind.CHANCE:
                            game.chance = [card]
                        elif kind is Kind.CHEST:
                            game.chest = [card]
                        with patch.object(game.random, 'randint', side_effect=(first, second)):
                            game._take_turn(0)
                        self.assertEqual(actual, game.view(), (start, first, second, card))
                        self.assertEqual(skipped, actual.players[0].jailed or actual.players[0].bankrupt)

    def test_jail_skips_actions_and_bankrupt_seats_do_not_consume_turns(self):
        game = self.game(4)
        game.players[0].jailed = True
        game.players[0].position = JAIL
        game.players[1].bankrupt = game.players[3].bankrupt = True
        policy = MCTSDecisions(depth=1)
        rng = Random(0)
        with patch.object(rng, 'randint', side_effect=AssertionError('No roll in jail')):
            view, phase, remaining = policy._step(game.view(), 'move', 1, None, rng)
        self.assertFalse(view.players[0].jailed)
        self.assertEqual(view.players[0].position, JAIL)
        self.assertEqual((view.active_player, view.turns, phase, remaining), (2, 1, 'move', 0))

    def test_purchase_then_single_even_build_and_hotel_limit(self):
        game = self.game()
        game.owners[1] = 0
        policy, rng = MCTSDecisions(), Random(0)
        view, phase, remaining = policy._step(game.view(), 'buy', 1, 3, rng)
        self.assertEqual((view.players[0].cash, phase, remaining), (1440, 'build', 1))
        self.assertEqual(policy._actions(view, phase, 0), (None, 1, 3))
        view, phase, remaining = policy._step(view, phase, remaining, 3, rng)
        self.assertEqual((view.players[0].cash, view.buildings[3]), (1390, 1))
        self.assertEqual((phase, view.active_player, view.turns), ('move', 1, 1))
        self.assertEqual(policy._actions(view, 'build', 0), (None,))  # Opponent owns no set.
        game.owners[3] = 0
        game.buildings[1] = game.buildings[3] = 5
        self.assertIsNone(policy.build(game.view(), 0, (1, 3)))

    def test_decisions_preserve_live_state_rng_decks_and_trading_memory(self):
        game = Game([adaptive_bot(), baseline_bot()])
        game.owners[1] = game.owners[3] = 0
        game.bots[0].trading.beliefs[1] = (.1, .2, .7)
        before, dice = game.view(), game.random.getstate()
        decks = (list(game.chance), list(game.chest))
        beliefs = dict(game.bots[0].trading.beliefs)
        policy = game.bots[0].decisions
        legal = game._build_options(0)
        first = policy.build(before, 0, legal)
        self.assertIn(first, (None,) + legal)
        self.assertEqual(first, policy.build(before, 0, legal))
        purchase = policy.buy(before, 0, 5)
        self.assertIsInstance(purchase, bool)
        self.assertEqual(purchase, policy.buy(before, 0, 5))
        self.assertEqual(game.view(), before)
        self.assertEqual(game.random.getstate(), dice)
        self.assertEqual((game.chance, game.chest), decks)
        self.assertEqual(game.bots[0].trading.beliefs, beliefs)

    def test_legality_reserves_empty_options_and_eliminated_player(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        policy = MCTSDecisions()
        self.assertFalse(policy.buy(game.view(), 0, 0))
        self.assertFalse(policy.buy(game.view(), 0, 1))
        self.assertIsNone(policy.build(game.view(), 0, ()))
        self.assertIsNone(policy.build(game.view(), 0, (5,)))
        game.players[0].cash = 249
        self.assertFalse(policy.buy(game.view(), 0, 5))
        game.players[0].cash = 299
        self.assertIsNone(policy.build(game.view(), 0, (1, 3)))
        game.players[0].bankrupt = True
        self.assertFalse(policy.buy(game.view(), 0, 5))
        self.assertIsNone(policy.build(game.view(), 0, (1, 3)))

    def test_final_turn_ties_skip_and_terminal_states_do_not_simulate(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        game.turns = 149
        policy = MCTSDecisions()
        with patch.object(policy, '_sample_move', side_effect=AssertionError('Past turn cap')):
            self.assertFalse(policy.buy(game.view(), 0, 5))
            self.assertIsNone(policy.build(game.view(), 0, (1, 3)))
        game.turns = 150
        with patch.object(policy, '_choose', side_effect=AssertionError('Terminal')):
            self.assertFalse(policy.buy(game.view(), 0, 5))
            self.assertIsNone(policy.build(game.view(), 0, (1, 3)))

    def test_uct_expands_tree_aggregates_chance_and_respects_budget(self):
        game = self.game()
        policy = MCTSDecisions(iterations=80, depth=2)
        created = []
        def record(*args):
            node = _MCTSNode(*args)
            created.append(node)
            return node
        with patch('monopoly.advanced_decisions._MCTSNode', side_effect=record), \
             patch.object(policy, '_sample_move', wraps=policy._sample_move) as moves, \
             patch.object(policy, '_rollout', wraps=policy._rollout) as rollouts:
            policy.buy(game.view(), 0, 5)
        root = created[0]
        self.assertEqual(root.visits, 80)
        self.assertEqual(sum(e.visits for e in root.edges.values()), 80)
        self.assertTrue(all(e.visits > 0 for e in root.edges.values()))
        self.assertLessEqual(len(created), 81)  # At most one new node per iteration.
        self.assertEqual(rollouts.call_count, 80)
        self.assertLessEqual(moves.call_count, 80 * 2)
        self.assertTrue(any(len(e.outcomes) > 1 for n in created for e in n.edges.values()))
        self.assertTrue(all(-e.visits <= e.total <= e.visits for n in created for e in n.edges.values()))

    def test_search_can_select_buy_and_best_build_from_backed_up_rewards(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        game.turns = 149  # Deterministic terminal payoffs isolate UCT action choice.
        policy = MCTSDecisions()
        with patch.object(policy, '_reward', side_effect=lambda v, p: int(v.owners[5] == p)):
            self.assertTrue(policy.buy(game.view(), 0, 5))
        with patch.object(policy, '_reward', side_effect=lambda v, p: v.buildings[3]):
            self.assertEqual(policy.build(game.view(), 0, (1, 3)), 3)

    def test_horizon_finishes_last_sampled_turn_before_evaluation(self):
        game = self.game()
        game.owners[1] = game.owners[3] = 0
        policy = MCTSDecisions(depth=1)
        self.assertFalse(policy._stopped(game.view(), 'buy', 0, 0))
        self.assertFalse(policy._stopped(game.view(), 'build', 0, 0))
        self.assertTrue(policy._stopped(game.view(), 'move', 0, 0))
        with patch.object(policy, '_sample_move', side_effect=AssertionError('Horizon reached')):
            policy._rollout(game.view(), 'build', 0, 0, Random(0))

    def test_ordinary_games_use_mcts_and_controlled_games_keep_ev(self):
        self.assertIsInstance(adaptive_bot().decisions, MCTSDecisions)
        self.assertIs(type(adaptive_bot(use_mcts=False).decisions), ExpectedValueDecisions)
        for strategy in STRATEGIES:
            for seat in range(4):
                self.assertIs(type(make_game(strategy, 0, seat).bots[seat].decisions), ExpectedValueDecisions)

    def test_full_search_lineup_replays_with_reused_bots(self):
        bots = [hard_bargainer_bot(), adaptive_bot(), risk_averse_bot(), baseline_bot()]
        game = Game(bots, seed=42)
        result = game.play()
        repeat = Game(bots, seed=42)
        self.assertEqual(result, repeat.play())
        self.assertEqual(game.trade_history, repeat.trade_history)
        self.assertFalse(result.bankrupt[result.winner])


if __name__ == '__main__':
    unittest.main()
