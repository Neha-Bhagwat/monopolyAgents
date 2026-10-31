from .models import GameView
from .bots import SimpleDecisions
from .valuation import (
    BUILD_RESERVE,
    building_gain,
    landing_probabilities,
    purchase_gain,
)



class ExpectimaxDecisions:
    def buy(self, view: GameView, player: int, square: int) -> bool:
        # Compare expected outcomes of buying vs skipping
        pass

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        # Evaluate legal builds using lookahead
        pass


class MCTSDecisions:
    def buy(self, view: GameView, player: int, square: int) -> bool:
        # Compare buying vs skipping using simulations
        pass

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        # Run simulations to choose a building
        pass    

class ProspectTheoryDecisions(SimpleDecisions):
    """Buying/building using a simplified Prospect Theory model."""

    # Initialization of loss aversion probabilites
    def __init__(
        self,
        loss_aversion: float = 2.25,
        alpha: float = 0.88,
        probability_gamma: float = 0.65,
        capital_hurdle: float = 0.10,
    ) -> None:
        if loss_aversion < 1 or not 0 < alpha <= 1 or not 0 < probability_gamma <= 1:
            raise ValueError("Invalid Prospect Theory parameters.")
        if capital_hurdle < 0:
            raise ValueError("capital_hurdle must be nonnegative.")

        self.loss_aversion = loss_aversion
        self.alpha = alpha
        self.probability_gamma = probability_gamma
        self.capital_hurdle = capital_hurdle

    def _value(self, outcome: float) -> float:
        """Convert a monetary gain or loss into subjective utility."""
        if outcome >= 0:
            return outcome ** self.alpha
        return -self.loss_aversion * (-outcome) ** self.alpha

    def _weight(self, probability: float) -> float:
        """Apply nonlinear probability weighting."""
        p = min(1.0, max(0.0, probability))

        if p == 0.0 or p == 1.0:
            return p

        gamma = self.probability_gamma
        return p ** gamma / (
            p ** gamma + (1.0 - p) ** gamma
        ) ** (1.0 / gamma)

    def _rent_probability(
        self,
        view: GameView,
        player: int,
        square: int,
        buying: bool,
    ) -> float:
        # Approximate the chance of future relevant rent collection.
        live_count = sum(not p.bankrupt for p in view.players)

        if live_count <= 1:
            return 0.0

        remaining_turns = max(0, view.turn_limit - view.turns - 1)
        opponent_turns = remaining_turns * (live_count - 1) / live_count

        relevant_squares = [square]
        prop = view.board[square].property
        assert prop is not None

        # Buying the last deed in a group increases rent on the whole group.
        if buying:
            group = [
                i for i, tile in enumerate(view.board)
                if tile.property is not None
                and tile.property.group == prop.group
            ]

            if group and all(
                i == square or view.owners[i] == player
                for i in group
            ):
                relevant_squares = group

        landing = landing_probabilities()
        per_turn_probability = min(
            1.0,
            sum(landing[i] for i in relevant_squares),
        )

        # Approximate repeated opponent turns as independent landings.
        return 1.0 - (1.0 - per_turn_probability) ** opponent_turns

    def _prospect_score(
        self,
        expected_gain: float,
        cost: int,
        rent_probability: float,
    ) -> float:
        """Score a simplified rent-success / no-rent prospect."""
        if expected_gain <= 0 or rent_probability <= 0:
            return float("-inf")

        # The existing expected-gain helper already deducts
        # a 10% capital-use hurdle.
        hurdle = self.capital_hurdle * cost
        gross_expected_benefit = expected_gain + hurdle

        # Approximate the gain conditional on earning rent.
        gain_if_rent = gross_expected_benefit / rent_probability - hurdle
        loss_if_no_rent = -hurdle

        return (
            self._weight(rent_probability) * self._value(gain_if_rent)
            + self._weight(1.0 - rent_probability) * self._value(loss_if_no_rent)
        )

    def buy(
        self,
        view: GameView,
        player: int,
        square: int,
    ) -> bool:
        # Preserve SimpleDecisions' minimum cash reserve.
        if not super().buy(view, player, square):
            return False

        prop = view.board[square].property
        assert prop is not None

        score = self._prospect_score(
            purchase_gain(view, player, square),
            prop.price,
            self._rent_probability(view, player, square, buying=True),
        )

        return score > 0

    def build(
        self,
        view: GameView,
        player: int,
        legal: tuple[int, ...],
    ) -> int | None:
        best_square = None
        best_score = 0.0

        for square in legal:
            prop = view.board[square].property
            assert prop is not None

            if view.players[player].cash < prop.build_cost + BUILD_RESERVE:
                continue

            score = self._prospect_score(
                building_gain(view, square),
                prop.build_cost,
                self._rent_probability(view, player, square, buying=False),
            )

            if score > best_score:
                best_square = square
                best_score = score

        return best_square