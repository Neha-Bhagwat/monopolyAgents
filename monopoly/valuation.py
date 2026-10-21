"""Estimate property income and compare both sides of a proposed trade."""

from collections import Counter
from dataclasses import dataclass
from functools import cache

from .board import BOARD, CHANCE, CHEST, JAIL, CardMove, Kind
from .models import GameView, Offer, completed_sets, legal_offer, ownership_after_trade


# Visiting the Jail square and being held there lead to different next turns.
IN_JAIL = len(BOARD)
DICE_TOTALS = Counter(first + second for first in range(1, 7) for second in range(1, 7))
SET_BONUS = 50
CAPITAL_HURDLE = 0.10


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


def _future_opponent_turns(view: GameView) -> float:
    # Buying, building, and trading all happen after the current dice roll.
    active_players = sum(not p.bankrupt for p in view.players)
    remaining_turns = max(0, view.turn_limit - view.turns - 1)
    if active_players < 2:
        return 0.0
    return remaining_turns * (active_players - 1) / active_players


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


def purchase_gain(view: GameView, player: int, square: int) -> float:
    """Expected gain after a liquidity charge; deeds retain face value at the cap."""
    prop = view.board[square].property
    assert prop is not None
    new_owners = list(view.owners)
    new_owners[square] = player
    owners = tuple(new_owners)
    before = completed_sets(view.owners, player)
    after = completed_sets(owners, player)
    future_turns = _future_opponent_turns(view)
    rent_before = _income_per_opponent_turn(view, view.owners, player, before)
    rent_after = _income_per_opponent_turn(view, owners, player, after)
    bonus = (len(after) - len(before)) * SET_BONUS if future_turns else 0
    return (rent_after - rent_before) * future_turns + bonus - prop.price * CAPITAL_HURDLE


def building_gain(view: GameView, square: int) -> float:
    """Extra rent minus the same liquidity charge used for purchases."""
    prop = view.board[square].property
    assert prop is not None
    level = view.buildings[square]
    current_rent = 2 * prop.rents[0] if level == 0 else prop.rents[level]
    additional_rent = prop.rents[level + 1] - current_rent
    return landing_probabilities()[square] * additional_rent * _future_opponent_turns(view) - prop.build_cost * CAPITAL_HURDLE


def score_trade(view: GameView, offer: Offer) -> tuple[TradeValue, TradeValue]:
    """Score a legal offer at the end of a turn, without applying it."""
    if not legal_offer(view, offer):
        raise ValueError("Only a legal offer can be scored.")

    after = ownership_after_trade(view, offer)
    opponent_turns = _future_opponent_turns(view)
    set_bonus = SET_BONUS if opponent_turns else 0
    cash_change = offer.ask_cash - offer.give_cash
    deed_change = _deed_value(view, offer.ask_properties) - _deed_value(view, offer.give_properties)

    def compare(player: int, cash: int, deeds: int) -> TradeValue:
        old_sets = completed_sets(view.owners, player)
        new_sets = completed_sets(after, player)
        before_rent = _income_per_opponent_turn(view, view.owners, player, old_sets)
        after_rent = _income_per_opponent_turn(view, after, player, new_sets)
        expense_change = 0.0
        opponents = sum(not p.bankrupt for p in view.players) - 1
        for other, state in enumerate(view.players):
            if other == player or state.bankrupt:
                continue
            old_income = _income_per_opponent_turn(view, view.owners, other, completed_sets(view.owners, other))
            new_income = _income_per_opponent_turn(view, after, other, completed_sets(after, other))
            expense_change += (new_income - old_income) * opponent_turns / opponents
        return TradeValue(
            player, cash, deeds, (after_rent - before_rent) * opponent_turns - expense_change,
            (len(new_sets) - len(old_sets)) * set_bonus,
            tuple(group for group in new_sets if group not in old_sets),
            tuple(group for group in old_sets if group not in new_sets),
        )

    return (
        compare(offer.proposer, cash_change, deed_change),
        compare(offer.recipient, -cash_change, -deed_change),
    )
