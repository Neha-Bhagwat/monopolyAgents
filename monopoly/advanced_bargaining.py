"""Three small bargaining policies sharing the existing trade calculator."""

from collections.abc import Iterable

from .bargaining import _both_gain, _counteroffers, candidate_offers
from .models import GameView, Offer, Reply, legal_offer
from .valuation import score_trade


# Hypothetical minimum gains wanted by easy, moderate, and demanding recipients.
OPPONENT_THRESHOLDS = (0, 25, 75)


def conflict_risk(demand: float, alternative: float) -> float:
    """Fraction of our demanded gain lost by taking the other's deal; no deal = 0."""
    if demand <= 0:
        return 1.0
    return max(0.0, (demand - max(0.0, alternative)) / demand)


class ZeuthenTrading:
    """Discrete Zeuthen-style concessions, with a positive-gain deadline fallback."""

    def __init__(self):
        self.last_offer: dict[int, tuple[int, Offer]] = {}

    def reset(self) -> None:
        self.last_offer.clear()

    def _remember(self, view: GameView, offer: Offer | None) -> Offer | None:
        if offer is not None:
            self.last_offer[offer.recipient] = (view.turns, offer)
        return offer

    def propose(self, view: GameView, player: int) -> Offer | None:
        best, best_gain = None, 0.0
        for offer in candidate_offers(view, player):
            ours, theirs = score_trade(view, offer)
            if _both_gain(ours, theirs) and ours.total > best_gain:
                best, best_gain = offer, ours.total
        return self._remember(view, best)

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        if not legal_offer(view, offer):
            return Reply.REJECT
        theirs, ours = score_trade(view, offer)
        if round_number >= 3:
            return Reply.ACCEPT if _both_gain(ours, theirs) else Reply.REJECT
        options = []
        for counter in _counteroffers(view, offer):
            us, them = score_trade(view, counter)
            if _both_gain(us, them):
                options.append((counter, us.total, them.total))
        previous = self.last_offer.get(offer.proposer)
        if previous is not None and previous[0] == view.turns and round_number > 1:
            last = previous[1]
            same_deeds = (last.give_properties == offer.ask_properties
                          and last.ask_properties == offer.give_properties)
            if same_deeds and legal_offer(view, last):
                us, them = score_trade(view, last)
                if _both_gain(us, them):
                    demand = (last, us.total, them.total)
                else:
                    demand = None
            else:
                demand = None
        else:
            demand = None
        if demand is None:
            demand = max(options, key=lambda item: item[1], default=None)
        if demand is None or ours.total >= demand[1]:
            return Reply.ACCEPT if _both_gain(ours, theirs) else Reply.REJECT

        own_risk = conflict_risk(demand[1], ours.total)
        other_risk = conflict_risk(theirs.total, demand[2])
        if own_risk <= other_risk:
            sufficient = []
            for candidate, our_gain, their_gain in options:
                if our_gain >= demand[1] or their_gain < demand[2]:
                    continue
                new_own_risk = conflict_risk(our_gain, ours.total)
                new_other_risk = conflict_risk(theirs.total, their_gain)
                if new_own_risk > new_other_risk:
                    sufficient.append((candidate, our_gain, their_gain))
            # Highest retained own gain = smallest sufficient concession on this grid.
            demand = max(sufficient, key=lambda item: item[1], default=demand)
        return self._remember(view, demand[0])


def acceptance_likelihood(gain: float, threshold: int) -> float:
    """Allow noisy replies so one surprising observation cannot rule a type out."""
    return 0.9 if gain > 0 and gain >= threshold else 0.1


class BayesianTrading:
    """Learn three recipient-threshold probabilities from actual outgoing replies."""

    def __init__(self):
        self.beliefs: dict[int, tuple[float, ...]] = {}
        self.observations: dict[int, int] = {}

    def reset(self) -> None:
        self.beliefs.clear()
        self.observations.clear()

    def beliefs_for(self, opponent: int) -> tuple[float, ...]:
        return self.beliefs.get(opponent, (1 / 3, 1 / 3, 1 / 3))

    def acceptance_probability(self, opponent: int, gain: float) -> float:
        return sum(weight * acceptance_likelihood(gain, threshold)
                   for weight, threshold in zip(self.beliefs_for(opponent), OPPONENT_THRESHOLDS))

    def observe_response(self, view: GameView, offer: Offer, accepted: bool) -> None:
        _, recipient = score_trade(view, offer)
        weights = []
        for prior, threshold in zip(self.beliefs_for(offer.recipient), OPPONENT_THRESHOLDS):
            likelihood = acceptance_likelihood(recipient.total, threshold)
            weights.append(prior * (likelihood if accepted else 1 - likelihood))
        total = sum(weights)
        self.beliefs[offer.recipient] = tuple(weight / total for weight in weights)
        self.observations[offer.recipient] = self.observations.get(offer.recipient, 0) + 1

    def _best_offer(self, view: GameView, offers: Iterable[Offer]) -> tuple[Offer | None, float]:
        best, best_expected = None, 0.0
        for offer in offers:
            ours, theirs = score_trade(view, offer)
            expected = ours.total * self.acceptance_probability(offer.recipient, theirs.total)
            if _both_gain(ours, theirs) and expected > best_expected:
                best, best_expected = offer, expected
        return best, best_expected

    def propose(self, view: GameView, player: int) -> Offer | None:
        return self._best_offer(view, candidate_offers(view, player))[0]

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        if not legal_offer(view, offer):
            return Reply.REJECT
        theirs, ours = score_trade(view, offer)
        counter, expected = (None, 0.0)
        if round_number < 3:
            counter, expected = self._best_offer(view, _counteroffers(view, offer))
        # An incoming offer can be accepted now; a counter might be refused.
        if _both_gain(ours, theirs) and ours.total >= expected:
            return Reply.ACCEPT
        return counter if counter is not None else Reply.REJECT


def minimax_regret_choice(payoffs: list[tuple[float, ...]]) -> int:
    """Choose lowest worst regret; break ties by worst payoff, then mean payoff."""
    if not payoffs or not payoffs[0] or any(len(row) != len(payoffs[0]) for row in payoffs):
        raise ValueError("Provide a nonempty rectangular payoff table.")
    best = [max(row[column] for row in payoffs) for column in range(len(payoffs[0]))]
    best_index, best_key = 0, None
    for index, row in enumerate(payoffs):
        worst_regret = max(value - actual for value, actual in zip(best, row))
        # Smaller regret wins. Negative signs prefer larger payoffs on ties.
        key = (worst_regret, -min(row), -sum(row))
        if best_key is None or key < best_key:
            best_index, best_key = index, key
    return best_index


class MinimaxRegretTrading:
    """Minimize missed gain under three possible recipient acceptance thresholds."""

    def _choose(
        self, view: GameView, offers: Iterable[Offer], incoming: Offer | None = None,
    ) -> Reply | Offer | None:
        actions: list[Reply | Offer | None] = [None]  # Walking away always pays zero.
        payoffs = [(0.0, 0.0, 0.0)]
        if incoming is not None:
            theirs, ours = score_trade(view, incoming)
            if _both_gain(ours, theirs):
                actions.append(Reply.ACCEPT)
                payoffs.append((ours.total,) * len(OPPONENT_THRESHOLDS))
        for offer in offers:
            ours, theirs = score_trade(view, offer)
            if _both_gain(ours, theirs):
                actions.append(offer)
                payoffs.append(tuple(ours.total if theirs.total >= threshold else 0.0
                                     for threshold in OPPONENT_THRESHOLDS))
        return actions[minimax_regret_choice(payoffs)]

    def propose(self, view: GameView, player: int) -> Offer | None:
        return self._choose(view, candidate_offers(view, player))

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        if not legal_offer(view, offer):
            return Reply.REJECT
        counters = _counteroffers(view, offer) if round_number < 3 else ()
        return self._choose(view, counters, offer) or Reply.REJECT
