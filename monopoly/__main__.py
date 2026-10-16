import argparse

from .board import BOARD
from .bots import AcceptingBot, SimpleBot
from .engine import Game
from .models import GameView, Offer


def print_game(seed: int) -> None:
    game = Game([SimpleBot() for _ in range(4)], seed=seed)
    result = game.play()
    reasons = {"last_player": "last player standing", "turn_limit": "turn limit"}
    print(f"Seed {seed}: Player {result.winner + 1} won after {result.turns} turns.")
    print(f"End reason: {reasons[result.end_reason]}")
    for i, worth in enumerate(result.net_worth):
        status = " (bankrupt)" if result.bankrupt[i] else ""
        print(f"Player {i + 1}: ${worth:,}{status}")


class TradeDemoBot(SimpleBot):
    """Makes the demo offer when the engine asks on the first turn."""

    def propose(self, view: GameView, player: int) -> Offer | None:
        if view.turns == 0:
            return Offer(player, 1, give_properties=(5,), ask_properties=(3,), give_cash=25)
        return None


def print_trade() -> None:
    game = Game([TradeDemoBot(), AcceptingBot()], turn_limit=2, trace=print)
    game.players[0].position = 5
    game.owners[1] = 0
    game.owners[5] = 0
    for square in (3, 7, 9):
        game.owners[square] = 1

    print("Trade demo: two turns from a prepared board.")
    result = game.play()
    if result.trades_accepted != 1:
        raise RuntimeError("The demo expected exactly one accepted trade.")

    print("After two turns:")
    for player, state in enumerate(game.players):
        properties = [BOARD[i].name for i, owner in enumerate(game.owners) if owner == player]
        print(f"Player {player + 1}: ${state.cash:,} and {', '.join(properties)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simplified Monopoly demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the full game.")
    parser.add_argument("--trade-demo", action="store_true", help="Watch a scripted trade during play.")
    args = parser.parse_args()
    if args.trade_demo:
        print_trade()
    else:
        print_game(args.seed)


if __name__ == "__main__":
    main()
