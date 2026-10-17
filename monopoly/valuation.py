"""Estimate property income and compare both sides of a proposed trade."""

from collections import Counter
from dataclasses import dataclass
from functools import cache

from .board import BOARD, CHANCE, CHEST, GROUPS, JAIL, CardMove, Kind
from .models import GameView, Offer, legal_offer


# Visiting the Jail square and being held there lead to different next turns.
IN_JAIL = len(BOARD)
DICE_TOTALS = Counter(first + second for first in range(1, 7) for second in range(1, 7))
SET_BONUS = 50


def _destinations(square: int) -> list[int]:
    kind = BOARD[square].kind
    if kind is Kind.GO_TO_JAIL:
        return [IN_JAIL]
    if kind in (Kind.CHANCE, Kind.CHEST):
        cards = CHANCE if kind is Kind.CHANCE else CHEST
        destinations = []
        for card in cards:
            if card is CardMove.GO_TO_JAIL:
                destinations.append(IN_JAIL)
            elif card is CardMove.ADVANCE_TO_GO:
                destinations.append(0)
            else:
                destinations.append(square)
        return destinations
    return [square]


def movement_table() -> list[list[float]]:
    """Each row gives next-turn destination chances from one state."""
    table = [[0.0] * (len(BOARD) + 1) for _ in range(len(BOARD) + 1)]
    for start in range(len(BOARD)):
        for total, ways in DICE_TOTALS.items():
            square = (start + total) % len(BOARD)
            destinations = _destinations(square)
            for end in destinations:
                table[start][end] += ways / (36 * len(destinations))
    table[IN_JAIL][JAIL] = 1.0
    return table


@cache
def landing_probabilities() -> tuple[float, ...]:
    """Long-run chance of ending an individual turn at each board square."""
    table = movement_table()
    chances = [1.0] + [0.0] * len(BOARD)
    for _ in range(1_000):
        next_chances = [0.0] * len(chances)
        for start, row in enumerate(table):
            for end, chance in enumerate(row):
                next_chances[end] += chances[start] * chance
        if max(abs(before - after) for before, after in zip(chances, next_chances)) < 1e-12:
            # On the board, the jail state and the normal Jail square are shown together.
            next_chances[JAIL] += next_chances[IN_JAIL]
            return tuple(next_chances[:len(BOARD)])
        chances = next_chances
    raise RuntimeError("The board's landing chances did not settle.")


@dataclass(frozen=True)
class TradeValue:
    player: int
    cash_change: int
    deed_change: int
    rent_change: float
    set_bonus_change: int
    sets_gained: tuple[str, ...]
    sets_lost: tuple[str, ...]

    @property
    def total(self) -> float:
        return self.cash_change + self.deed_change + self.rent_change + self.set_bonus_change


def _completed_sets(owners: tuple[int | None, ...], player: int) -> tuple[str, ...]:
    return tuple(
        group for group, squares in GROUPS.items()
        if all(owners[square] == player for square in squares)
    )


def _income_per_opponent_turn(
    view: GameView, owners: tuple[int | None, ...], player: int, sets: tuple[str, ...],
) -> float:
    total = 0.0
    for square, chance in enumerate(landing_probabilities()):
        if owners[square] == player:
            prop = view.board[square].property
            assert prop is not None
            level = view.buildings[square]
            rent = prop.rents[level]
            if level == 0 and prop.group in sets:
                rent *= 2
            total += chance * rent
    return total


def _deed_value(view: GameView, squares: tuple[int, ...]) -> int:
    value = 0
    for square in squares:
        prop = view.board[square].property
        assert prop is not None
        value += prop.price
    return value


def score_trade(view: GameView, offer: Offer) -> tuple[TradeValue, TradeValue]:
    """Score a legal offer at the end of a turn, without applying it."""
    if not legal_offer(view, offer):
        raise ValueError("Only a legal offer can be scored.")

    new_owners = list(view.owners)
    for square in offer.give_properties:
        new_owners[square] = offer.recipient
    for square in offer.ask_properties:
        new_owners[square] = offer.proposer
    after = tuple(new_owners)

    # The turn cap counts everybody's turns. This turn has already moved.
    active_players = sum(not p.bankrupt for p in view.players)
    remaining_turns = max(0, view.turn_limit - view.turns - 1)
    opponent_turns = remaining_turns * (active_players - 1) / active_players
    set_bonus = SET_BONUS if remaining_turns else 0
    cash_change = offer.ask_cash - offer.give_cash
    deed_change = _deed_value(view, offer.ask_properties) - _deed_value(view, offer.give_properties)

    def compare(player: int, cash: int, deeds: int) -> TradeValue:
        old_sets = _completed_sets(view.owners, player)
        new_sets = _completed_sets(after, player)
        before_rent = _income_per_opponent_turn(view, view.owners, player, old_sets)
        after_rent = _income_per_opponent_turn(view, after, player, new_sets)
        return TradeValue(
            player, cash, deeds, (after_rent - before_rent) * opponent_turns,
            (len(new_sets) - len(old_sets)) * set_bonus,
            tuple(group for group in new_sets if group not in old_sets),
            tuple(group for group in old_sets if group not in new_sets),
        )

    return (
        compare(offer.proposer, cash_change, deed_change),
        compare(offer.recipient, -cash_change, -deed_change),
    )
