from .models import GameView

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


class ProspectTheoryDecisions:
    def buy(self, view: GameView, player: int, square: int) -> bool:
        # Evaluate potential gains and losses
        pass

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        # Choose a building based on risk-sensitive utility
        pass