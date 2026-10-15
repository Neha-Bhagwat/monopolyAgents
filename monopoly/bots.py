from .models import GameView, Offer, Reply


class SimpleBot:
    """A starter policy for exercising the engine."""

    def buy(self, view: GameView, player: int, square: int) -> bool:
        prop = view.board[square].property
        return prop is not None and view.players[player].cash >= prop.price + 150

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        for square in legal:
            prop = view.board[square].property
            if prop is not None and view.players[player].cash >= prop.build_cost + 250:
                return square
        return None

    def propose(self, view: GameView, player: int) -> Offer | None:
        return None

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        return Reply.REJECT


class AcceptingBot(SimpleBot):
    """Accepts any offer the engine has already checked, for the trade demo."""

    def respond(self, view: GameView, offer: Offer, round_number: int) -> Reply | Offer:
        return Reply.ACCEPT
