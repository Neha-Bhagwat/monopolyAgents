from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random

from .board import BOARD, CHANCE, CHEST, GROUPS, JAIL, CardMove, Kind
from .models import Bot, GameResult, GameView, Offer, PlayerView, Reply, TradeRecord, legal_offer
from .valuation import score_trade


@dataclass
class Player:
    cash: int = 1500
    position: int = 0
    jailed: bool = False
    bankrupt: bool = False


class Game:
    def __init__(
        self, bots: Sequence[Bot], seed: int = 0, turn_limit: int = 150,
        *, trace: Callable[[str], None] | None = None,
    ):
        if not 2 <= len(bots) <= 4:
            raise ValueError("A game needs two to four bots.")
        if type(turn_limit) is not int or turn_limit < 1:
            raise ValueError("The turn limit must be positive.")
        if len({id(bot) for bot in bots}) != len(bots):
            raise ValueError("Each seat needs a separate bot instance.")

        self.bots = tuple(bots)
        for bot in self.bots:
            reset = getattr(bot, "reset", None)
            if reset is not None:
                reset()
        self.players = [Player() for _ in bots]
        self.owners: list[int | None] = [None] * len(BOARD)
        self.buildings = [0] * len(BOARD)
        self.seed = seed
        self.random = Random(seed)
        self.chance = list(CHANCE)
        self.chest = list(CHEST)
        self.random.shuffle(self.chance)
        self.random.shuffle(self.chest)
        self.turn_limit = turn_limit
        self.active_player = 0
        self.turns = 0
        self.trades_proposed = 0
        self.trades_accepted = 0
        self.trade_history: list[TradeRecord] = []
        self.trace = trace

    def view(self) -> GameView:
        players = tuple(
            PlayerView(p.cash, p.position, p.jailed, p.bankrupt) for p in self.players
        )
        return GameView(
            BOARD, players, tuple(self.owners), tuple(self.buildings),
            self.active_player, self.turns, self.turn_limit,
        )

    def play(self) -> GameResult:
        while self.turns < self.turn_limit and len(self._survivors()) > 1:
            self._take_turn(self.active_player)
            self.turns += 1
            self.active_player = (self.active_player + 1) % len(self.players)
            while self.players[self.active_player].bankrupt:
                self.active_player = (self.active_player + 1) % len(self.players)

        survivors = self._survivors()
        net_worth = tuple(self.net_worth(i) for i in range(len(self.players)))
        highest = max(net_worth[i] for i in survivors)
        tied = [i for i in survivors if net_worth[i] == highest]
        return GameResult(
            self.seed, tied[0] if len(tied) == 1 else Random(self.seed).choice(tied),
            self.turns, "last_player" if len(survivors) == 1 else "turn_limit",
            net_worth, tuple(p.bankrupt for p in self.players),
            self.trades_proposed, self.trades_accepted,
        )

    def net_worth(self, player: int) -> int:
        total = self.players[player].cash
        for square, owner in enumerate(self.owners):
            if owner == player:
                prop = BOARD[square].property
                assert prop is not None
                total += prop.price + self.buildings[square] * prop.build_cost
        return total

    def _survivors(self) -> list[int]:
        return [i for i, player in enumerate(self.players) if not player.bankrupt]

    def _take_turn(self, player: int) -> None:
        state = self.players[player]
        if state.jailed:
            if self.trace is not None:
                self.trace(f"Turn {self.turns + 1}: Player {player + 1} skips a turn in jail.")
            state.jailed = False
            return

        dice = self.random.randint(1, 6) + self.random.randint(1, 6)
        position = state.position + dice
        if position >= len(BOARD):
            state.cash += 200
            if self.trace is not None:
                self.trace(f"Player {player + 1} passes Go and receives $200.")
        state.position = position % len(BOARD)
        if self.trace is not None:
            square = BOARD[state.position].name
            self.trace(f"Turn {self.turns + 1}: Player {player + 1} rolls {dice} and lands on {square}.")
        self._land(player)
        if state.bankrupt or state.jailed or len(self._survivors()) == 1:
            return

        legal = self._build_options(player)
        if legal:
            square = self.bots[player].build(self.view(), player, legal)
            if type(square) is int and square in legal:
                prop = BOARD[square].property
                assert prop is not None
                state.cash -= prop.build_cost
                self.buildings[square] += 1
                if self.trace is not None:
                    self.trace(f"Player {player + 1} builds on {BOARD[square].name} for ${prop.build_cost:,}.")

        offer = self.bots[player].propose(self.view(), player)
        if isinstance(offer, Offer) and offer.proposer == player:
            self.negotiate(offer)

    def _land(self, player: int) -> None:
        position = self.players[player].position
        square = BOARD[position]
        if square.kind is Kind.PROPERTY:
            self._land_on_property(player, position)
        elif square.kind is Kind.TAX:
            if self.trace is not None:
                self.trace(f"Player {player + 1} owes ${square.tax} in tax.")
            self._pay(player, square.tax)
        elif square.kind is Kind.GO_TO_JAIL:
            self._send_to_jail(player)
        elif square.kind is Kind.CHANCE:
            self._draw_card(player, self.chance)
        elif square.kind is Kind.CHEST:
            self._draw_card(player, self.chest)

    def _land_on_property(self, player: int, square: int) -> None:
        prop = BOARD[square].property
        assert prop is not None
        owner = self.owners[square]
        if owner is None:
            if self.players[player].cash >= prop.price and self.bots[player].buy(self.view(), player, square):
                self.players[player].cash -= prop.price
                self.owners[square] = player
                if self.trace is not None:
                    self.trace(f"Player {player + 1} buys {BOARD[square].name} for ${prop.price}.")
        elif owner != player:
            level = self.buildings[square]
            rent = prop.rents[level]
            if level == 0 and all(self.owners[i] == owner for i in GROUPS[prop.group]):
                rent *= 2
            if self.trace is not None:
                self.trace(f"Player {player + 1} owes Player {owner + 1} ${rent} in rent.")
            self._pay(player, rent, owner)

    def _send_to_jail(self, player: int) -> None:
        self.players[player].position = JAIL
        self.players[player].jailed = True
        if self.trace is not None:
            self.trace(f"Player {player + 1} is sent to Jail.")

    def _draw_card(self, player: int, deck: list[int | CardMove]) -> None:
        card = deck.pop(0)
        deck.append(card)
        if card is CardMove.ADVANCE_TO_GO:
            self.players[player].position = 0
            self.players[player].cash += 200
            if self.trace is not None:
                self.trace(f"Player {player + 1} draws Advance to Go and receives $200.")
        elif card is CardMove.GO_TO_JAIL:
            self._send_to_jail(player)
        elif isinstance(card, int):
            if self.trace is not None:
                self.trace(f"Player {player + 1} draws a cash card worth {card:+d} dollars.")
            if card >= 0:
                self.players[player].cash += card
            else:
                self._pay(player, -card)

    def _pay(self, player: int, amount: int, creditor: int | None = None) -> None:
        payer = self.players[player]
        paid = min(amount, payer.cash)
        payer.cash -= paid
        if creditor is not None:
            self.players[creditor].cash += paid
        if paid < amount:
            payer.bankrupt = True
            if self.trace is not None:
                self.trace(f"Player {player + 1} pays ${paid} and is bankrupt.")
            for square, owner in enumerate(self.owners):
                if owner == player:
                    self.owners[square] = creditor
                    self.buildings[square] = 0

    def _build_options(self, player: int) -> tuple[int, ...]:
        legal = []
        for group in GROUPS.values():
            if all(self.owners[i] == player for i in group):
                lowest = min(self.buildings[i] for i in group)
                for i in group:
                    prop = BOARD[i].property
                    assert prop is not None
                    if self.buildings[i] == lowest < 5 and self.players[player].cash >= prop.build_cost:
                        legal.append(i)
        return tuple(legal)

    def negotiate(self, offer: Offer) -> bool:
        """Run one negotiation. A counter swaps the proposer and recipient."""
        if not self.legal_offer(offer):
            if self.trace is not None:
                self.trace("The engine rejects an illegal offer.")
            return False
        self.trades_proposed += 1
        initiated_by = offer.proposer
        if self.trace is not None:
            self._trace_offer(offer, "offers")
        for round_number in range(1, 4):
            reply = self.bots[offer.recipient].respond(self.view(), offer, round_number)
            if reply is Reply.ACCEPT:
                values = score_trade(self.view(), offer)
                self._apply_trade(offer)
                self.trades_accepted += 1
                self.trade_history.append(TradeRecord(
                    self.turns + 1, offer, initiated_by, values[0].total, values[1].total,
                ))
                if self.trace is not None:
                    self.trace(f"Player {offer.recipient + 1} accepts. The trade goes through.")
                    for seat in (offer.proposer, offer.recipient):
                        self.trace(f"Player {seat + 1} now has ${self.players[seat].cash:,} in cash.")
                return True
            if not isinstance(reply, Offer):
                if self.trace is not None:
                    self.trace(f"Player {offer.recipient + 1} rejects the offer.")
                return False
            if (reply.proposer, reply.recipient) != (offer.recipient, offer.proposer):
                if self.trace is not None:
                    self.trace("The counteroffer names the wrong players. Negotiation ends.")
                return False
            if not self.legal_offer(reply):
                if self.trace is not None:
                    self.trace("The counteroffer is illegal. Negotiation ends.")
                return False
            offer = reply
            if self.trace is not None:
                self._trace_offer(offer, "counters with")
        if self.trace is not None:
            self.trace("The negotiation ends without a trade.")
        return False

    def _trace_offer(self, offer: Offer, action: str) -> None:
        assert self.trace is not None

        def describe(properties: tuple[int, ...], cash: int) -> str:
            items = [BOARD[square].name for square in properties]
            if cash:
                items.append(f"${cash:,}")
            return " and ".join(items) or "nothing"

        given = describe(offer.give_properties, offer.give_cash)
        requested = describe(offer.ask_properties, offer.ask_cash)
        self.trace(
            f"Player {offer.proposer + 1} {action} {given} for {requested} "
            f"to Player {offer.recipient + 1}."
        )

    def legal_offer(self, offer: Offer) -> bool:
        return legal_offer(self.view(), offer)

    def _apply_trade(self, offer: Offer) -> None:
        self.players[offer.proposer].cash += offer.ask_cash - offer.give_cash
        self.players[offer.recipient].cash += offer.give_cash - offer.ask_cash
        for square in offer.give_properties:
            self.owners[square] = offer.recipient
        for square in offer.ask_properties:
            self.owners[square] = offer.proposer
