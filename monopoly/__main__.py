import argparse
from dataclasses import replace
from pathlib import Path

from .bargaining import NashTrading, TitForTatTrading
from .board import BOARD
from .bots import AcceptingBot, SimpleBot, baseline_bot, cooperator_bot, reciprocator_bot
from .engine import Game
from .experiments import print_summary, replay, run_pilot
from .models import GameView, Offer, Reply
from .valuation import landing_probabilities, score_trade


def print_game(game: Game, show_trades: bool = False) -> None:
    result = game.play()
    reasons = {"last_player": "last player standing", "turn_limit": "turn limit"}
    print(f"Seed {result.seed}: Player {result.winner + 1} won after {result.turns} turns.")
    print(f"End reason: {reasons[result.end_reason]}")
    for i, worth in enumerate(result.net_worth):
        status = " (bankrupt)" if result.bankrupt[i] else ""
        print(f"Player {i + 1}: ${worth:,}{status}")
    if show_trades:
        print(f"Trades proposed: {result.trades_proposed}. Trades accepted: {result.trades_accepted}.")
        for record in game.trade_history:
            print(f"Turn {record.turn} accepted trade:")
            print_offer(record.offer)


def print_offer(offer: Offer) -> None:
    def describe(properties: tuple[int, ...], cash: int) -> str:
        items = [BOARD[square].name for square in properties]
        if cash:
            items.append(f"${cash:,}")
        return " and ".join(items) or "nothing"

    for player, properties, cash in (
        (offer.proposer, offer.give_properties, offer.give_cash),
        (offer.recipient, offer.ask_properties, offer.ask_cash),
    ):
        print(f"  Player {player + 1} gives {describe(properties, cash)}.")


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
        print(f"  Estimated net rent:     {value.rent_change:+.2f}")
        print(f"  Color-set bonus change: {value.set_bonus_change:+.2f}")
        print(f"  Estimated total ($):    {value.total:+.2f}")
    print("\nThe offer was scored only. No money or deeds moved.")


def print_bargaining_demo() -> None:
    game = Game([SimpleBot(), SimpleBot()])
    set_up_demo_board(game)
    game.turns = 109
    view = game.view()
    offer = replace(DEMO_OFFER, give_cash=0, ask_cash=90)
    print("Bargaining demo: Player 2 is deciding on this offer.")
    print_offer(offer)
    proposer, recipient = score_trade(view, offer)
    print(f"Player 1's estimated gain is ${proposer.total:.2f}.")
    print(f"Player 2's estimated gain is ${recipient.total:.2f}.")

    reciprocator = TitForTatTrading()
    for name, strategy in (("Cooperator", NashTrading()), ("Reciprocator", reciprocator)):
        reply = strategy.respond(view, offer, 1)
        if reply is Reply.ACCEPT:
            print(f"{name} accepts.")

    print("\nNow suppose Player 1 previously sent an unfair offer.")
    reciprocator = TitForTatTrading()
    unfair = replace(DEMO_OFFER, give_cash=0, ask_cash=150)
    reciprocator.respond(view, unfair, 1)
    print(f"Reciprocator now wants at least ${reciprocator.minimum_gain(0)} in estimated gain.")
    reply = reciprocator.respond(view, offer, 1)
    if isinstance(reply, Offer):
        print("Reciprocator counters with:")
        print_offer(reply)
    elif reply is Reply.REJECT:
        print("Reciprocator rejects.")
    else:
        print("Reciprocator accepts.")
    print("This demo compares decisions. No cash or deeds move.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simplified Monopoly demo.")
    parser.add_argument("--seed", type=int, help="Seed for a single full game (default 42).")
    demos = parser.add_mutually_exclusive_group()
    demos.add_argument("--trade-demo", action="store_true", help="Watch a scripted trade during play.")
    demos.add_argument("--valuation-demo", action="store_true", help="Score the scripted trade.")
    demos.add_argument("--baseline-game", action="store_true", help="Play four Baselines from Go.")
    demos.add_argument("--bargaining-game", action="store_true", help="Play both negotiators and two Baselines.")
    demos.add_argument("--bargaining-demo", action="store_true", help="Compare replies to an example offer.")
    demos.add_argument("--pilot", action="store_true", help="Run the 120-game controlled comparison.")
    demos.add_argument("--replay", type=Path, metavar="FOLDER", help="Replay a game from a pilot folder.")
    parser.add_argument("--output", type=Path, help="New pilot folder (default results/step5).")
    parser.add_argument("--game-id", help="Saved game ID to replay.")
    args = parser.parse_args()
    if args.replay and not args.game_id:
        parser.error("--replay requires --game-id")
    if args.game_id and not args.replay:
        parser.error("--game-id requires --replay")
    if args.output and not args.pilot:
        parser.error("--output requires --pilot")
    if args.seed is not None and (args.pilot or args.replay or args.trade_demo or
                                  args.valuation_demo or args.bargaining_demo):
        parser.error("--seed applies to single full games. The pilot uses seeds 0 through 9.")
    seed = 42 if args.seed is None else args.seed
    if args.pilot:
        output = args.output or Path("results/step5")
        try:
            print_summary(run_pilot(output))
        except (OSError, ValueError) as error:
            parser.exit(2, f"{error}\n")
        print(f"Saved results in {output.resolve()}")
    elif args.replay:
        try:
            replay(args.replay, args.game_id)
        except (OSError, ValueError) as error:
            parser.exit(2, f"{error}\n")
    elif args.trade_demo:
        print_trade()
    elif args.valuation_demo:
        print_valuation()
    elif args.bargaining_demo:
        print_bargaining_demo()
    elif args.bargaining_game:
        print("Players 1 to 4: Cooperator, Reciprocator, Baseline, Baseline.")
        bots = [cooperator_bot(), reciprocator_bot(), baseline_bot(), baseline_bot()]
        print_game(Game(bots, seed=seed), show_trades=True)
    elif args.baseline_game:
        print_game(Game([baseline_bot() for _ in range(4)], seed=seed), show_trades=True)
    else:
        print_game(Game([SimpleBot() for _ in range(4)], seed=seed))


if __name__ == "__main__":
    main()
