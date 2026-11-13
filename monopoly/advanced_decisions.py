"""Bounded Expectimax, Monte Carlo tree search, and prospect-theory decisions."""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import isfinite, log, sqrt, tanh
from random import Random

from .board import BOARD, CHANCE, CHEST, GROUPS, Kind
from .engine import Game, Player
from .models import GameView
from .bots import SimpleDecisions
from .valuation import (BUILD_RESERVE, CAPITAL_HURDLE, DICE_TOTALS, SET_BONUS,
                        building_gain, landing_probabilities, purchase_gain)


@lru_cache(maxsize=4096)
def _assets(owners: tuple[int | None, ...], buildings: tuple[int, ...], players: int):
    """Immutable board-only data shared by leaves with different dice/cash states.

    The engine uses the fixed BOARD. Cash, turn, position, and the searcher's
    identity do not affect these quantities and must not fragment the cache.
    """
    capital = [0] * players
    incomes = [0.0] * players
    sets = [0] * players
    builds = [[] for _ in range(players)]
    completed = set()
    for group, squares in GROUPS.items():
        owner = owners[squares[0]]
        if owner is None or not all(owners[square] == owner for square in squares):
            continue
        completed.add(group)
        sets[owner] += 1
        lowest = min(buildings[square] for square in squares)
        if lowest < 5:
            builds[owner].extend(square for square in squares if buildings[square] == lowest)
    for square, chance in enumerate(landing_probabilities()):
        owner = owners[square]
        if owner is None:
            continue
        prop = BOARD[square].property
        level = buildings[square]
        capital[owner] += prop.price + level * prop.build_cost
        rent = prop.rents[level]
        if level == 0 and prop.group in completed:
            rent *= 2
        incomes[owner] += chance * rent
    return tuple(capital), tuple(incomes), tuple(sets), tuple(tuple(b) for b in builds)


class _Forecast(Game):
    """Private copy using engine landing/payment rules, without bots or RNG.

    Unowned purchases are deferred to the search's decision node. Only view,
    landing, payment, build-options and net-worth helpers are used, never play.
    """

    def __init__(self, view: GameView):
        self.players = [Player(p.cash, p.position, p.jailed, p.bankrupt) for p in view.players]
        self.owners = list(view.owners)
        self.buildings = list(view.buildings)
        self.active_player = view.active_player
        self.turns = view.turns
        self.turn_limit = view.turn_limit
        self.trace = None
        self.chance = list(CHANCE)
        self.chest = list(CHEST)

    def _land_on_property(self, player: int, square: int) -> None:
        if self.owners[square] is not None:
            super()._land_on_property(player, square)


def _buy(view: GameView, player: int, square: int) -> GameView:
    players = list(view.players)
    players[player] = replace(players[player], cash=players[player].cash - view.board[square].property.price)
    owners = list(view.owners)
    owners[square] = player
    return replace(view, players=tuple(players), owners=tuple(owners))


def _build(view: GameView, player: int, square: int) -> GameView:
    players = list(view.players)
    players[player] = replace(players[player], cash=players[player].cash - view.board[square].property.build_cost)
    buildings = list(view.buildings)
    buildings[square] += 1
    return replace(view, players=tuple(players), buildings=tuple(buildings))


class ExpectimaxDecisions:
    """Maximize estimated relative wealth over a bounded chance tree.

    Depth counts future individual player turns, not full rounds. Dice outcomes
    are exact; unknown cards are independent uniform draws from the fixed deck.
    Opponents use expected-value buy/build choices. Future trades are omitted.
    Leaves value current assets, stationary expected net rent, complete sets and
    liquidity. This is a heuristic planner, not a trained model or win oracle.
    """

    def __init__(self, depth: int = 2, max_nodes: int = 1000):
        if type(depth) is not int or not 1 <= depth <= 3:
            raise ValueError("Expectimax depth must be 1, 2, or 3 player turns.")
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("Expectimax max_nodes must be a positive integer.")
        self.depth = depth
        self.max_nodes = max_nodes
        self._transpositions: dict[tuple, tuple[float, int]] = {}

    @staticmethod
    def _can_buy(view: GameView, player: int, square: int) -> bool:
        prop = view.board[square].property
        return (not view.players[player].bankrupt and prop is not None
                and view.owners[square] is None
                and view.players[player].cash >= prop.price + 150)

    @staticmethod
    def _build_options(view: GameView, player: int) -> tuple[int, ...]:
        candidates = _assets(view.owners, view.buildings, len(view.players))[3][player]
        return tuple(square for square in candidates
                     if view.players[player].cash >=
                     view.board[square].property.build_cost + BUILD_RESERVE)

    def buy(self, view: GameView, player: int, square: int) -> bool:
        self._transpositions.clear()
        if not self._can_buy(view, player, square):
            return False
        # A purchase is followed by this turn's build opportunity in the engine.
        skip = self._finish_build(view, player, self.depth, [0])
        purchase = self._finish_build(_buy(view, player, square), player, self.depth, [0])
        return purchase > skip + 1e-9

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        self._transpositions.clear()
        best, value = None, self._next_turn(view, player, self.depth, [0])
        for square in self._build_options(view, player):
            if square not in legal:
                continue
            candidate = self._next_turn(_build(view, player, square), player, self.depth, [0])
            if candidate > value + 1e-9:
                best, value = square, candidate
        return best

    def _next_turn(self, view: GameView, player: int, depth: int, budget: list[int]) -> float:
        survivors = [i for i, p in enumerate(view.players) if not p.bankrupt]
        if len(survivors) <= 1:
            return self._evaluate(view, player)
        active = (view.active_player + 1) % len(view.players)
        while active not in survivors:
            active = (active + 1) % len(view.players)
        next_view = GameView(view.board, view.players, view.owners, view.buildings,
                             active, view.turns + 1, view.turn_limit)
        return self._search(next_view, player, depth, budget)

    def _search(self, view: GameView, player: int, depth: int, budget: list[int]) -> float:
        if (depth == 0 or view.turns >= view.turn_limit or view.players[player].bankrupt
                or sum(not p.bankrupt for p in view.players) <= 1
                or budget[0] >= self.max_nodes):
            return self._evaluate(view, player)
        # Only complete subtrees are reusable. Charge their original expansion
        # cost on a hit: otherwise memoization silently changes budget cutoffs.
        key = (view.players, view.owners, view.buildings, view.active_player,
               view.turns, view.turn_limit, player, depth)
        cached = self._transpositions.get(key)
        if cached is not None and budget[0] + cached[1] < self.max_nodes:
            budget[0] += cached[1]
            return cached[0]
        start = budget[0]
        budget[0] += 1
        
        value = sum(probability * self._after_landing(outcome, player, depth - 1, budget, skipped)
                    for probability, outcome, skipped in self._outcomes(view))
        
        if budget[0] < self.max_nodes and len(self._transpositions) < 4096:
            self._transpositions[key] = (value, budget[0] - start)
        return value

    @staticmethod
    def _outcomes(view: GameView):
        """Yield (probability, state, skip-actions) for one future movement."""
        active = view.active_player
        if view.players[active].jailed:
            game = _Forecast(view)
            game.players[active].jailed = False
            yield 1.0, game.view(), True
            return
        for total, ways in DICE_TOTALS.items():
            position = (view.players[active].position + total) % len(view.board)
            kind = view.board[position].kind
            cards = CHANCE if kind is Kind.CHANCE else CHEST if kind is Kind.CHEST else (None,)
            for card in cards:
                game = _Forecast(view)
                if view.players[active].position + total >= len(view.board):
                    game.players[active].cash += 200
                game.players[active].position = position
                if kind is Kind.CHANCE:
                    game.chance = [card]
                elif kind is Kind.CHEST:
                    game.chest = [card]
                game._land(active)
                state = game.players[active]
                yield ways / (36 * len(cards)), game.view(), state.jailed or state.bankrupt

    def _after_landing(self, view: GameView, player: int, depth: int,
                       budget: list[int], skipped: bool) -> float:
        active = view.active_player
        if skipped or sum(not p.bankrupt for p in view.players) <= 1:
            return self._next_turn(view, player, depth, budget)
        square = view.players[active].position
        if self._can_buy(view, active, square):
            if active == player:
                return max(self._finish_build(view, player, depth, budget),
                           self._finish_build(_buy(view, active, square), player, depth, budget))
            if purchase_gain(view, active, square) > 0:
                view = _buy(view, active, square)
        return self._finish_build(view, player, depth, budget)

    def _finish_build(self, view: GameView, player: int, depth: int, budget: list[int]) -> float:
        active = view.active_player
        options = self._build_options(view, active)
        if not options:
            return self._next_turn(view, player, depth, budget)
        if active == player:
            values = [self._next_turn(view, player, depth, budget)]
            values.extend(self._next_turn(_build(view, active, square), player, depth, budget)
                          for square in options)
            return max(values)
        best = max(options, key=lambda square: building_gain(view, square), default=None)
        if best is not None and building_gain(view, best) > 0:
            view = _build(view, active, best)
        return self._next_turn(view, player, depth, budget)

    @staticmethod
    def _evaluate(view: GameView, player: int) -> float:
        if view.players[player].bankrupt:
            return -100_000.0
        survivors = [i for i, p in enumerate(view.players) if not p.bankrupt]
        if len(survivors) == 1:
            return 100_000.0
        capital, incomes, sets, _ = _assets(view.owners, view.buildings, len(view.players))
        worth = [p.cash + capital[i] for i, p in enumerate(view.players)]
        if view.turns >= view.turn_limit:
            return worth[player] - max(worth[i] for i in survivors if i != player)
        remaining = max(0, view.turn_limit - view.turns)
        for i in survivors:
            others = sum(incomes[j] for j in survivors if j != i)
            net_rent = ((len(survivors) - 1) * incomes[i] - others) * remaining / len(survivors)
            worth[i] += net_rent + SET_BONUS * sets[i]
            worth[i] += CAPITAL_HURDLE * view.players[i].cash
        return worth[player] - max(worth[i] for i in survivors if i != player)


@dataclass
class _MCTSEdge:
    visits: int = 0
    total: float = 0.0
    # One action can produce many sampled chance outcomes. Its UCT statistics
    # aggregate all of them; chance outcomes themselves are never maximized.
    outcomes: dict[tuple[GameView, str, int], "_MCTSNode"] = field(default_factory=dict)


@dataclass
class _MCTSNode:
    view: GameView
    phase: str
    remaining: int
    edges: dict[int | None, _MCTSEdge]
    visits: int = 0


class MCTSDecisions:
    """UCT search over public-state copies, with expected-value rollouts.

    Each iteration selects actions, expands at most one sampled state, rolls
    out to a bounded horizon, then backs up a bounded relative-wealth reward.
    Depth counts future individual turns, excluding the current buy/build.
    Opponents use expected-value decisions; future trades are omitted. Cards
    are independent uniform draws because GameView hides the shuffled decks.
    A fresh local RNG per decision makes repeated calls reproducible without
    consuming the engine's dice or retaining a tree across changed boards.
    """

    def __init__(self, iterations: int = 128, depth: int = 8,
                 exploration: float = sqrt(2), seed: int = 0):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("MCTS iterations must be a positive integer.")
        if type(depth) is not int or not 1 <= depth <= 150:
            raise ValueError("MCTS depth must be between 1 and 150 player turns.")
        if (isinstance(exploration, bool) or not isinstance(exploration, (int, float))
                or not isfinite(exploration) or exploration < 0):
            raise ValueError("MCTS exploration must be finite and nonnegative.")
        if type(seed) is not int:
            raise ValueError("MCTS seed must be an integer.")
        self.iterations = iterations
        self.depth = depth
        self.exploration = exploration
        self.seed = seed

    def buy(self, view: GameView, player: int, square: int) -> bool:
        if self._terminal(view, player) or not ExpectimaxDecisions._can_buy(view, player, square):
            return False
        # The engine calls buy at the landed square, before building.
        return self._choose(view, player, "buy", (None, square)) is not None

    def build(self, view: GameView, player: int, legal: tuple[int, ...]) -> int | None:
        if self._terminal(view, player):
            return None
        options = tuple(square for square in ExpectimaxDecisions._build_options(view, player)
                        if square in legal)
        if not options:
            return None
        return self._choose(view, player, "build", (None,) + options)

    @staticmethod
    def _terminal(view: GameView, player: int) -> bool:
        return (view.turns >= view.turn_limit or view.players[player].bankrupt
                or sum(not p.bankrupt for p in view.players) <= 1)

    def _stopped(self, view: GameView, phase: str, remaining: int, player: int) -> bool:
        # Always finish the last sampled turn's purchase/build before evaluating.
        return self._terminal(view, player) or (phase == "move" and remaining == 0)

    @staticmethod
    def _actions(view: GameView, phase: str, player: int) -> tuple[int | None, ...]:
        active = view.active_player
        if phase == "move":
            return (None,)
        if phase == "buy":
            square = view.players[active].position
            if not ExpectimaxDecisions._can_buy(view, active, square):
                return (None,)
            if active == player:
                return (None, square)
            return (square,) if purchase_gain(view, active, square) > 0 else (None,)
        options = ExpectimaxDecisions._build_options(view, active)
        if active == player:
            return (None,) + options
        best = max(options, key=lambda square: building_gain(view, square), default=None)
        return (best,) if best is not None and building_gain(view, best) > 0 else (None,)

    @staticmethod
    def _advance(view: GameView) -> GameView:
        active = (view.active_player + 1) % len(view.players)
        while view.players[active].bankrupt:
            active = (active + 1) % len(view.players)
        return replace(view, active_player=active, turns=view.turns + 1)

    @staticmethod
    def _sample_move(view: GameView, rng: Random) -> tuple[GameView, bool]:
        """Sample movement, using the engine for tax, rent, cards and insolvency."""
        game = _Forecast(view)
        active = view.active_player
        state = game.players[active]
        if state.jailed:
            state.jailed = False
            return game.view(), True
        total = rng.randint(1, 6) + rng.randint(1, 6)
        position = state.position + total
        if position >= len(view.board):
            state.cash += 200
        state.position = position % len(view.board)
        kind = view.board[state.position].kind
        if kind is Kind.CHANCE:
            game.chance = [rng.choice(CHANCE)]
        elif kind is Kind.CHEST:
            game.chest = [rng.choice(CHEST)]
        game._land(active)
        return game.view(), state.jailed or state.bankrupt

    def _step(self, view: GameView, phase: str, remaining: int,
              action: int | None, rng: Random) -> tuple[GameView, str, int]:
        if phase == "buy":
            if action is not None:
                view = _buy(view, view.active_player, action)
            return view, "build", remaining
        if phase == "build":
            if action is not None:
                view = _build(view, view.active_player, action)
            return self._advance(view), "move", remaining
        view, skipped = self._sample_move(view, rng)
        if skipped or sum(not p.bankrupt for p in view.players) <= 1:
            return self._advance(view), "move", remaining - 1
        return view, "buy", remaining - 1

    @staticmethod
    def _reward(view: GameView, player: int) -> float:
        # Scale dollars into [-1, 1] so the UCT exploration term is meaningful.
        # This heuristic reward is not a calibrated probability of winning.
        return tanh(ExpectimaxDecisions._evaluate(view, player) / 1500)

    def _rollout(self, view: GameView, phase: str, remaining: int,
                 player: int, rng: Random) -> float:
        while not self._stopped(view, phase, remaining, player):
            actions = self._actions(view, phase, player)
            action = None
            if phase == "buy" and len(actions) > 1:
                square = view.players[view.active_player].position
                if purchase_gain(view, view.active_player, square) > 0:
                    action = square
            elif phase == "build" and len(actions) > 1:
                best = max(actions[1:], key=lambda square: building_gain(view, square))
                if building_gain(view, best) > 0:
                    action = best
            else:
                action = actions[0]
            view, phase, remaining = self._step(view, phase, remaining, action, rng)
        return self._reward(view, player)

    def _choose(self, view: GameView, player: int, phase: str,
                actions: tuple[int | None, ...]) -> int | None:
        root = _MCTSNode(view, phase, self.depth, {a: _MCTSEdge() for a in actions})
        rng = Random(self.seed)
        for _ in range(self.iterations):
            node = root
            nodes, edges = [root], []
            while not self._stopped(node.view, node.phase, node.remaining, player):
                # Explore every action before UCT; insertion order resolves ties.
                unvisited = [a for a, e in node.edges.items() if e.visits == 0]
                if unvisited:
                    action = unvisited[0]
                else:
                    action = max(node.edges, key=lambda a: (
                        node.edges[a].total / node.edges[a].visits
                        + self.exploration * sqrt(log(node.visits) / node.edges[a].visits)))
                edge = node.edges[action]
                edges.append(edge)
                state, next_phase, remaining = self._step(
                    node.view, node.phase, node.remaining, action, rng)
                key = (state, next_phase, remaining)
                child = edge.outcomes.get(key)
                expanded = child is None
                if expanded:
                    options = self._actions(state, next_phase, player)
                    child = _MCTSNode(state, next_phase, remaining,
                                      {a: _MCTSEdge() for a in options})
                    edge.outcomes[key] = child
                node = child
                nodes.append(node)
                if expanded:
                    break
            reward = self._rollout(node.view, node.phase, node.remaining, player, rng)
            for visited in nodes:
                visited.visits += 1
            for edge in edges:
                edge.visits += 1
                edge.total += reward
        # Recommend the most visited action, then its mean reward. Exact ties
        # favor skipping, which is first. Tiny budgets can leave actions untried.
        return max(root.edges, key=lambda a: (root.edges[a].visits,
                   root.edges[a].total / root.edges[a].visits if root.edges[a].visits else -float("inf")))


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
        gross_expected_benefit = (
            expected_gain + CAPITAL_HURDLE * cost
        )

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
