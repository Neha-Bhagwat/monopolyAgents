import argparse

from .board import BOARD
from .bots import AcceptingBot, SimpleBot
from .engine import Game
from .models import GameView, Offer
from .valuation import landing_probabilities, score_trade


def print_game(seed: int) -> None:
    game = Game([SimpleBot() for _ in range(4)], seed=seed)
    result = game.play()
    reasons = {"last_player": "last player standing", "turn_limit": "turn limit"}
    print(f"Seed {seed}: Player {result.winner + 1} won after {result.turns} turns.")
    print(f"End reason: {reasons[result.end_reason]}")
    for i, worth in enumerate(result.net_worth):
        status = " (bankrupt)" if result.bankrupt[i] else ""
        print(f"Player {i + 1}: ${worth:,}{status}")


DEMO_OFFER = Offer(0, 1, give_properties=(5,), ask_properties=(3,), give_cash=25)


class TradeDemoBot(SimpleBot):
    """Makes the demo offer when the engine asks on the first turn."""

    def propose(self, view: GameView, player: int) -> Offer | None:
        if view.turns == 0 and player == DEMO_OFFER.proposer:
            return DEMO_OFFER
        return None


def set_up_demo_board(game: Game) -> None:
    game.players[0].position = 5
    game.owners[1] = 0
    game.owners[5] = 0
    for square in (3, 7, 9):
        game.owners[square] = 1


def print_trade() -> None:
    game = Game([TradeDemoBot(), AcceptingBot()], turn_limit=2, trace=print)
    set_up_demo_board(game)
    print("Trade demo: two turns from a prepared board.")
    result = game.play()
    if result.trades_accepted != 1:
        raise RuntimeError("The demo expected exactly one accepted trade.")

    print("After two turns:")
    for player, state in enumerate(game.players):
        properties = [BOARD[i].name for i, owner in enumerate(game.owners) if owner == player]
        print(f"Player {player + 1}: ${state.cash:,} and {', '.join(properties)}")


def print_valuation() -> None:
    game = Game([SimpleBot(), SimpleBot()])
    set_up_demo_board(game)
    game.turns = 109
    print("Valuation demo: same offer as the trade demo, with 40 future turns in total.")
    print("Player 1 offers Oriental Avenue and $25 for Baltic Avenue.")
    print("\nEstimated chance of ending a turn on each property:")
    for square, probability in zip(BOARD, landing_probabilities()):
        if square.property is not None:
            print(f"  {square.name:23} {probability:6.2%}")

    for value in score_trade(game.view(), DEMO_OFFER):
        print(f"\nPlayer {value.player + 1}")
        if value.sets_gained:
            print(f"  Completes: {', '.join(value.sets_gained)}")
        print(f"  Cash change:            {value.cash_change:+.2f}")
        print(f"  Printed deed value:     {value.deed_change:+.2f}")
        print(f"  Estimated rent change:  {value.rent_change:+.2f}")
        print(f"  Color-set bonus change: {value.set_bonus_change:+.2f}")
        print(f"  Estimated total ($):    {value.total:+.2f}")
    print("\nThe offer was scored only. No money or deeds moved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simplified Monopoly demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the full game.")
    demos = parser.add_mutually_exclusive_group()
    demos.add_argument("--trade-demo", action="store_true", help="Watch a scripted trade during play.")
    demos.add_argument("--valuation-demo", action="store_true", help="Score the scripted trade.")
    args = parser.parse_args()
    if args.trade_demo:
        print_trade()
    elif args.valuation_demo:
        print_valuation()
    else:
        print_game(args.seed)


if __name__ == "__main__":
    main()
