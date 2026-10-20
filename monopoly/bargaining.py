"""Offer generation and the first two valuation-based trading policies."""

from collections.abc import Iterator
from dataclasses import replace

from .board import GROUPS
from .models import GameView, Offer, Reply, legal_offer
from .valuation import TradeValue, score_trade


CASH_OFFSETS = (-100, -50, -25, 0, 25, 50, 100)


def _with_cash(offer: Offer, payment: int) -> Offer:
    """Positive payment means the proposer pays the recipient."""
    return replace(offer, give_cash=max(0, payment), ask_cash=max(0, -payment))


def _cash_options(view: GameView, offer: Offer, center: int) -> Iterator[Offer]:
    payments = {center + offset for offset in CASH_OFFSETS}
    payments.add(0)
    for payment in sorted(payments):
        candidate = _with_cash(offer, payment)
        if legal_offer(view, candidate):
            yield candidate


def candidate_offers(view: GameView, player: int) -> Iterator[Offer]:
    """Try cash and one-deed swaps for properties that would finish a set."""
    owned = [square for square, owner in enumerate(view.owners) if owner == player]
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
        for offset in CASH_OFFSETS:
            payment = target_prop.price + offset
            if payment > 0:
                offer = Offer(player, seller, ask_properties=(target,), give_cash=payment)
                if legal_offer(view, offer):
                    yield offer
        for deed in owned:
            prop = view.board[deed].property
            assert prop is not None
            swap = Offer(player, seller, give_properties=(deed,), ask_properties=(target,))
            yield from _cash_options(view, swap, target_prop.price - prop.price)


def _counteroffers(view: GameView, offer: Offer) -> Iterator[Offer]:
    if not offer.give_properties and not offer.ask_properties:
        return
    reversed_offer = Offer(offer.recipient, offer.proposer, offer.ask_properties, offer.give_properties)
    original_payment = offer.ask_cash - offer.give_cash
    for counter in _cash_options(view, reversed_offer, original_payment):
        if counter.give_cash - counter.ask_cash != original_payment:
            yield counter


def _both_gain(ours: TradeValue, theirs: TradeValue) -> bool:
    return ours.total > 0 and theirs.total > 0


class NashTrading:
    """Choose the candidate with the biggest product of positive gains."""

    def propose(self, view: GameView, player: int) -> Offer | None:
        return self._best_offer(view, candidate_offers(view, player))

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        if not legal_offer(view, offer):
            return Reply.REJECT
        if _both_gain(*score_trade(view, offer)):
            return Reply.ACCEPT
        if round_number < 3:
            counter = self._best_offer(view, _counteroffers(view, offer))
            if counter is not None:
                return counter
        return Reply.REJECT

    def _best_offer(self, view: GameView, offers: Iterator[Offer]) -> Offer | None:
        best = None
        best_product = 0.0
        for offer in offers:
            ours, theirs = score_trade(view, offer)
            product = ours.total * theirs.total
            if _both_gain(ours, theirs) and product > best_product:
                best = offer
                best_product = product
        return best


class TitForTatTrading:
    """Use each opponent's most recent incoming offer to set our next terms."""

    def __init__(self):
        self.last_offer_fair: dict[int, bool] = {}

    def minimum_gain(self, opponent: int) -> int:
        if opponent not in self.last_offer_fair:
            return 10
        return 0 if self.last_offer_fair[opponent] else 25

    def propose(self, view: GameView, player: int) -> Offer | None:
        best = None
        best_gain = 0.0
        for offer in candidate_offers(view, player):
            ours, theirs = score_trade(view, offer)
            minimum = self.minimum_gain(offer.recipient)
            if _both_gain(ours, theirs) and ours.total >= minimum and ours.total > best_gain:
                best = offer
                best_gain = ours.total
        return best

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        if not legal_offer(view, offer):
            return Reply.REJECT
        theirs, ours = score_trade(view, offer)
        minimum = self.minimum_gain(offer.proposer)
        self.last_offer_fair[offer.proposer] = (
            _both_gain(ours, theirs) and ours.total >= theirs.total / 2
        )
        if _both_gain(ours, theirs) and ours.total >= minimum:
            return Reply.ACCEPT
        if round_number < 3:
            original_payment = offer.ask_cash - offer.give_cash
            counters = sorted(
                _counteroffers(view, offer),
                key=lambda terms: abs(terms.give_cash - terms.ask_cash - original_payment),
            )
            for counter in counters:
                ours, theirs = score_trade(view, counter)
                if _both_gain(ours, theirs) and ours.total >= minimum:
                    return counter
        return Reply.REJECT
