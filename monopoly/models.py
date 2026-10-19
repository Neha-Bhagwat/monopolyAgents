from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from .board import GROUPS, Square


@dataclass(frozen=True)
class PlayerView:
    cash: int
    position: int
    jailed: bool
    bankrupt: bool


@dataclass(frozen=True)
class GameView:
    board: tuple[Square, ...]
    players: tuple[PlayerView, ...]
    owners: tuple[int | None, ...]
    buildings: tuple[int, ...]
    active_player: int
    turns: int
    turn_limit: int


@dataclass(frozen=True)
class Offer:
    proposer: int
    recipient: int
    give_properties: tuple[int, ...] = ()
    ask_properties: tuple[int, ...] = ()
    give_cash: int = 0
    ask_cash: int = 0


def ownership_after_trade(view: GameView, offer: Offer) -> tuple[int | None, ...]:
    """Preview a legal offer using a copy of deed ownership."""
    owners = list(view.owners)
    for square in offer.give_properties:
        owners[square] = offer.recipient
    for square in offer.ask_properties:
        owners[square] = offer.proposer
    return tuple(owners)


def completed_sets(owners: tuple[int | None, ...], player: int) -> tuple[str, ...]:
    return tuple(
        group for group, squares in GROUPS.items()
        if all(owners[square] == player for square in squares)
    )


def legal_offer(view: GameView, offer: Offer) -> bool:
    """Check the same trade rules for the engine and the calculator."""
    if offer.proposer == offer.recipient:
        return False
    if not any((offer.give_properties, offer.ask_properties, offer.give_cash, offer.ask_cash)):
        return False

    sides = (
        (offer.proposer, offer.give_properties, offer.give_cash),
        (offer.recipient, offer.ask_properties, offer.ask_cash),
    )
    for player, properties, cash in sides:
        if type(player) is not int or not 0 <= player < len(view.players):
            return False
        state = view.players[player]
        if state.bankrupt or type(cash) is not int or not 0 <= cash <= state.cash:
            return False
        if len(set(properties)) != len(properties):
            return False
        for square in properties:
            if type(square) is not int or not 0 <= square < len(view.board):
                return False
            prop = view.board[square].property
            if prop is None or view.owners[square] != player:
                return False
            if any(view.buildings[i] for i in GROUPS[prop.group]):
                return False
    return True


class Reply(Enum):
    ACCEPT = auto()
    REJECT = auto()


@dataclass(frozen=True)
class TradeRecord:
    turn: int
    offer: Offer


class Bot(Protocol):
    def buy(self, view: GameView, player: int, square: int) -> bool: ...

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None: ...

    def propose(self, view: GameView, player: int) -> Offer | None: ...

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer: ...


@dataclass(frozen=True)
class GameResult:
    seed: int
    winner: int
    turns: int
    end_reason: str
    net_worth: tuple[int, ...]
    bankrupt: tuple[bool, ...]
    trades_proposed: int
    trades_accepted: int
