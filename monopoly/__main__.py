import argparse

from .board import BOARD
from .bots import AcceptingBot, SimpleBot
from .engine import Game
from .models import Offer


def print_game(seed: int) -> None:
    game = Game([SimpleBot() for _ in range(4)], seed=seed)
    result = game.play()
    reasons = {"last_player": "last player standing", "turn_limit": "turn limit"}
    print(f"Seed {seed}: Player {result.winner + 1} won after {result.turns} turns.")
    print(f"End reason: {reasons[result.end_reason]}")
    for i, worth in enumerate(result.net_worth):
        status = " (bankrupt)" if result.bankrupt[i] else ""
        print(f"Player {i + 1}: ${worth:,}{status}")


def print_trade() -> None:
    game = Game([SimpleBot(), AcceptingBot()])
    game.owners[1] = 0
    game.owners[5] = 0
    for square in (3, 7, 9):
        game.owners[square] = 1

    offer = Offer(0, 1, give_properties=(5,), ask_properties=(3,), give_cash=25)
    print("Player 1 offers Oriental Avenue and $25 for Baltic Avenue.")
    if not game.negotiate(offer):
        raise RuntimeError("The scripted trade was unexpectedly rejected.")

    print("Player 2 accepts.")
    for player, state in enumerate(game.players):
        properties = [BOARD[i].name for i, owner in enumerate(game.owners) if owner == player]
        print(f"Player {player + 1}: ${state.cash:,} and {', '.join(properties)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simplified Monopoly demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the full game.")
    parser.add_argument("--trade-demo", action="store_true", help="Show a scripted trade.")
    args = parser.parse_args()
    if args.trade_demo:
        print_trade()
    else:
        print_game(args.seed)


if __name__ == "__main__":
    main()
