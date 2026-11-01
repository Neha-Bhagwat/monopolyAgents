import unittest
from unittest.mock import patch

from monopoly.__main__ import TradeDemoBot
from monopoly.board import BOARD, GROUPS, JAIL, CardMove
from monopoly.bots import AcceptingBot, SimpleBot
from monopoly.engine import Game
from monopoly.models import Offer, Reply


class CounterBot(SimpleBot):
    def __init__(self):
        self.rounds = []

    def respond(self, view, offer, round_number):
        self.rounds.append(round_number)
        return Offer(
            offer.recipient, offer.proposer,
            give_properties=offer.ask_properties,
            ask_properties=offer.give_properties,
        )


class EngineTests(unittest.TestCase):
    def test_board_and_seeded_games(self):
        self.assertEqual(len(BOARD), 24)
        self.assertEqual(len(GROUPS), 5)
        self.assertEqual(sum(map(len, GROUPS.values())), 14)
        for count in (2, 3, 4):
            bots = [SimpleBot() for _ in range(count)]
            for seed in range(10):
                with self.subTest(players=count, seed=seed):
                    game = Game(bots, seed)
                    result = game.play()
                    self.assertEqual(result, Game(bots, seed).play())
                    self.assertLessEqual(result.turns, 150)
                    self.assertFalse(result.bankrupt[result.winner])
                    self.assertTrue(all(p.cash >= 0 for p in game.players))

    def test_pass_go_buy_and_collect_rent(self):
        game = Game([SimpleBot(), SimpleBot()])
        game.players[0].position = 23
        with patch.object(game.random, "randint", side_effect=(1, 1)):
            game._take_turn(0)
        self.assertEqual(game.players[0].position, 1)
        self.assertEqual(game.owners[1], 0)
        self.assertEqual(game.players[0].cash, 1640)

        game.players[1].position = 1
        game._land(1)
        self.assertEqual(game.players[1].cash, 1498)
        self.assertEqual(game.players[0].cash, 1642)

    def test_cards_and_jail(self):
        game = Game([SimpleBot(), SimpleBot()])
        game.players[0].position = 8
        game.chance = [CardMove.GO_TO_JAIL, CardMove.ADVANCE_TO_GO]
        game._land(0)
        self.assertTrue(game.players[0].jailed)
        self.assertEqual(game.players[0].position, JAIL)
        game._take_turn(0)
        self.assertFalse(game.players[0].jailed)
        self.assertEqual(game.players[0].position, JAIL)
        game.players[0].position = 8
        game._land(0)
        self.assertEqual(game.players[0].position, 0)
        self.assertEqual(game.players[0].cash, 1700)

    def test_building_requires_an_even_complete_group(self):
        game = Game([SimpleBot(), SimpleBot()])
        game.owners[1] = 0
        self.assertEqual(game._build_options(0), ())
        game.owners[3] = 0
        self.assertEqual(game._build_options(0), (1, 3))
        game.buildings[1] = 1
        self.assertEqual(game._build_options(0), (3,))
        game.buildings[1] = game.buildings[3] = 5
        self.assertEqual(game._build_options(0), ())

    def test_bankruptcy_to_a_player_and_to_the_bank(self):
        for creditor in (None, 1):
            with self.subTest(creditor=creditor):
                game = Game([SimpleBot(), SimpleBot()])
                game.owners[1] = 0
                game.owners[3] = 0
                game.buildings[1] = 1
                game.players[0].cash = 5
                game._pay(0, 10, creditor)
                self.assertTrue(game.players[0].bankrupt)
                self.assertEqual(game.players[0].cash, 0)
                self.assertEqual(game.players[1].cash, 1505 if creditor == 1 else 1500)
                self.assertEqual((game.owners[1], game.owners[3]), (creditor, creditor))
                self.assertEqual(game.buildings[1], 0)
                self.assertEqual(game.play().end_reason, "last_player")

    def test_turn_cap_uses_net_worth(self):
        game = Game([SimpleBot(), SimpleBot()], turn_limit=1)
        game.players[1].cash += 1000
        result = game.play()
        self.assertEqual((result.winner, result.turns, result.end_reason), (1, 1, "turn_limit"))

    def test_trade_moves_deeds_and_cash(self):
        game = Game([SimpleBot(), AcceptingBot()])
        game.owners[1] = 0
        game.owners[3] = 1
        accepted = game.negotiate(Offer(0, 1, (1,), (3,), give_cash=25))
        self.assertTrue(accepted)
        self.assertEqual((game.owners[1], game.owners[3]), (1, 0))
        self.assertEqual(tuple(p.cash for p in game.players), (1475, 1525))
        self.assertEqual((game.trades_proposed, game.trades_accepted), (1, 1))

    def test_trade_happens_during_play_and_next_turn_runs(self):
        for accept in (True, False):
            with self.subTest(accept=accept):
                messages = []
                other = AcceptingBot() if accept else SimpleBot()
                game = Game([TradeDemoBot(), other], turn_limit=2, trace=messages.append)
                game.players[0].position = 5
                game.owners[5], game.owners[3] = 0, 1
                with patch.object(game.random, "randint", side_effect=(3, 4, 3, 3)):
                    result = game.play()

                self.assertEqual(result.turns, 2)
                self.assertEqual(result.trades_proposed, 1)
                self.assertEqual(result.trades_accepted, int(accept))
                self.assertIn("Turn 1: Player 1 rolls 7 and lands on Free Parking.", messages[0])
                self.assertIn("offers Oriental Avenue and $25 for Baltic Avenue", messages[1])
                self.assertTrue(messages[-1].startswith("Turn 2: Player 2 rolls 6"))
                if accept:
                    self.assertEqual((game.owners[5], game.owners[3]), (1, 0))
                    self.assertEqual(tuple(p.cash for p in game.players), (1475, 1525))
                    self.assertEqual(len(game.trade_history), 1)
                    record = game.trade_history[0]
                    self.assertEqual(record.turn, 1)
                    self.assertEqual(record.offer, Offer(0, 1, (5,), (3,), give_cash=25))
                else:
                    self.assertEqual((game.owners[5], game.owners[3]), (0, 1))
                    self.assertEqual(tuple(p.cash for p in game.players), (1500, 1500))
                    self.assertEqual(game.trade_history, [])

    def test_illegal_offers_leave_state_unchanged(self):
        game = Game([SimpleBot(), AcceptingBot()])
        game.owners[1] = game.owners[3] = 0
        game.buildings[3] = 1
        game.owners[5] = 0
        before = game.view()
        offers = (
            Offer(0, 1, give_properties=(1,)),
            Offer(0, 1, give_properties=(7,)),
            Offer(0, 1, give_properties=(5, 5)),
            Offer(0, 1, give_cash=1501),
            Offer(0, 1, ask_cash=-5),
            Offer(0, 0, give_cash=1),
            Offer(0, 4, give_cash=1),
        )
        for offer in offers:
            with self.subTest(offer=offer):
                self.assertFalse(game.negotiate(offer))
                self.assertEqual(game.view(), before)
        self.assertEqual(game.trades_proposed, 0)

    def test_counter_can_be_accepted_or_expire(self):
        second = CounterBot()
        game = Game([AcceptingBot(), second])
        game.owners[1] = 0
        game.owners[3] = 1
        self.assertTrue(game.negotiate(Offer(0, 1, (1,), (3,), give_cash=50)))
        self.assertEqual(second.rounds, [1])
        self.assertEqual(tuple(p.cash for p in game.players), (1500, 1500))
        self.assertEqual((game.owners[1], game.owners[3]), (1, 0))

        first, second = CounterBot(), CounterBot()
        game = Game([first, second])
        game.owners[1] = 0
        game.owners[3] = 1
        self.assertFalse(game.negotiate(Offer(0, 1, (1,), (3,))))
        self.assertEqual(first.rounds, [2])
        self.assertEqual(second.rounds, [1, 3])
        self.assertEqual((game.owners[1], game.owners[3]), (0, 1))

    def test_rejected_offer_does_not_move_cash(self):
        game = Game([SimpleBot(), SimpleBot()])
        self.assertFalse(game.negotiate(Offer(0, 1, give_cash=25)))
        self.assertEqual(tuple(p.cash for p in game.players), (1500, 1500))
        self.assertEqual((game.trades_proposed, game.trades_accepted), (1, 0))
