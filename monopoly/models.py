from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from .board import Square


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


class Reply(Enum):
    ACCEPT = auto()
    REJECT = auto()


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
