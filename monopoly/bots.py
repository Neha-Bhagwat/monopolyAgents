from typing import Protocol

from .advanced_bargaining import BayesianTrading, MinimaxRegretTrading, ZeuthenTrading
from .bargaining import NashTrading, TitForTatTrading
from .board import GROUPS
from .models import GameView, Offer, Reply, completed_sets, legal_offer, ownership_after_trade
from .valuation import BUILD_RESERVE, building_gain, purchase_gain


BUY_RESERVE = 150


class BuyBuildPolicy(Protocol):
    def buy(self, view: GameView, player: int, square: int) -> bool: ...

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None: ...


class TradePolicy(Protocol):
    def propose(self, view: GameView, player: int) -> Offer | None: ...

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer: ...


class PolicyBot:
    """A game player assembled from one buy/build policy and one trading policy."""

    def __init__(self, decisions: BuyBuildPolicy, trading: TradePolicy):
        self.decisions = decisions
        self.trading = trading

    def reset(self) -> None:
        reset = getattr(self.trading, "reset", None)
        if reset is not None:
            reset()

    def buy(self, view: GameView, player: int, square: int) -> bool:
        return self.decisions.buy(view, player, square)

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        return self.decisions.build(view, player, legal)

    def propose(self, view: GameView, player: int) -> Offer | None:
        return self.trading.propose(view, player)

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        return self.trading.respond(view, offer, round_number)

    def observe_response(self, view: GameView, offer: Offer, accepted: bool) -> None:
        observe = getattr(self.trading, "observe_response", None)
        if observe is not None:
            observe(view, offer, accepted)


class SimpleDecisions:
    def buy(self, view: GameView, player: int, square: int) -> bool:
        prop = view.board[square].property
        return prop is not None and view.players[player].cash >= prop.price + BUY_RESERVE

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        for square in legal:
            prop = view.board[square].property
            if prop is not None and view.players[player].cash >= prop.build_cost + BUILD_RESERVE:
                return square
        return None


class ExpectedValueDecisions(SimpleDecisions):
    def buy(self, view: GameView, player: int, square: int) -> bool:
        return super().buy(view, player, square) and purchase_gain(view, player, square) > 0

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        best_square = None
        best_gain = 0.0
        for square in legal:
            prop = view.board[square].property
            assert prop is not None
            if view.players[player].cash < prop.build_cost + BUILD_RESERVE:
                continue
            gain = building_gain(view, square)
            if gain > best_gain:
                best_square = square
                best_gain = gain
        return best_square


class NoTrading:
    def propose(self, view: GameView, player: int) -> Offer | None:
        return None

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        return Reply.REJECT


class SetCompletionTrading:
    """Look one deed away from a set and favor swaps that help both sides."""

    def propose(self, view: GameView, player: int) -> Offer | None:
        cash_offer = None
        for group in GROUPS.values():
            missing = [square for square in group if view.owners[square] != player]
            if len(missing) != 1:
                continue
            target = missing[0]
            seller = view.owners[target]
            if seller is None:
                continue
            target_prop = view.board[target].property
            assert target_prop is not None

            for deed, owner in enumerate(view.owners):
                if owner != player:
                    continue
                prop = view.board[deed].property
                assert prop is not None
                difference = target_prop.price - prop.price
                swap = Offer(
                    player, seller, (deed,), (target,),
                    give_cash=max(0, difference), ask_cash=max(0, -difference),
                )
                if legal_offer(view, swap) and (
                    self._helps_complete_set(view, swap, player, -difference)
                    and self._helps_complete_set(view, swap, seller, difference)
                ):
                    return swap

            offer = Offer(player, seller, ask_properties=(target,), give_cash=target_prop.price)
            if cash_offer is None and legal_offer(view, offer) and (
                self._helps_complete_set(view, offer, player, -target_prop.price)
            ):
                cash_offer = offer
        return cash_offer

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply:
        cash_change = offer.give_cash - offer.ask_cash
        if legal_offer(view, offer) and self._helps_complete_set(
            view, offer, offer.recipient, cash_change,
        ):
            return Reply.ACCEPT
        return Reply.REJECT

    def _helps_complete_set(
        self, view: GameView, offer: Offer, player: int, cash_change: int,
    ) -> bool:
        if cash_change < 0 and view.players[player].cash + cash_change < BUY_RESERVE:
            return False
        before = set(completed_sets(view.owners, player))
        after = set(completed_sets(ownership_after_trade(view, offer), player))
        return before < after


class SimpleBot(SimpleDecisions, NoTrading):
    """The original starter bot used by the engine and demo checks."""


def baseline_bot() -> PolicyBot:
    return PolicyBot(SimpleDecisions(), SetCompletionTrading())


def cooperator_bot() -> PolicyBot:
    return PolicyBot(ExpectedValueDecisions(), NashTrading())


def reciprocator_bot() -> PolicyBot:
    return PolicyBot(ExpectedValueDecisions(), TitForTatTrading())


def hard_bargainer_bot(*, use_expectimax: bool = True) -> PolicyBot:
    """Use search in ordinary games; retain EV for trading-only experiments."""
    # Advanced policies also reuse SimpleDecisions; defer this import until the
    # base policies above are defined, including when either module loads first.
    from .advanced_decisions import ExpectimaxDecisions

    decisions = ExpectimaxDecisions() if use_expectimax else ExpectedValueDecisions()
    return PolicyBot(decisions, ZeuthenTrading())


def adaptive_bot(*, use_mcts: bool = True) -> PolicyBot:
    """Use MCTS in ordinary games; retain EV for trading-only experiments."""
    from .advanced_decisions import MCTSDecisions

    decisions = MCTSDecisions() if use_mcts else ExpectedValueDecisions()
    return PolicyBot(decisions, BayesianTrading())


def risk_averse_bot(*, use_prospect_theory: bool = True) -> PolicyBot:
    from .advanced_decisions import ProspectTheoryDecisions

    decisions = ProspectTheoryDecisions() if use_prospect_theory else ExpectedValueDecisions()
    return PolicyBot(decisions, MinimaxRegretTrading())


class AcceptingBot(SimpleBot):
    """Accepts any offer the engine has already checked, for the trade demo."""

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        return Reply.ACCEPT
