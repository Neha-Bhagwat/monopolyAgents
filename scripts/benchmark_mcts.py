"""Time one development game per four-bot lineup; this is not a tournament.

Run from the project root: python -B -m scripts.benchmark_mcts
"""

import json
import platform
from dataclasses import asdict
from itertools import combinations
from statistics import mean, median
from time import perf_counter

from monopoly.advanced_decisions import MCTSDecisions
from monopoly.bots import (adaptive_bot, baseline_bot, cooperator_bot,
                           hard_bargainer_bot, reciprocator_bot, risk_averse_bot)
from monopoly.engine import Game
from monopoly.experiments import source_hash


def main():
    timings = []

    class TimedMCTS(MCTSDecisions):
        def _choose(self, *args):
            start = perf_counter()
            result = super()._choose(*args)
            timings.append(perf_counter() - start)
            return result

    factories = (baseline_bot, cooperator_bot, reciprocator_bot,
                 hard_bargainer_bot, adaptive_bot, risk_averse_bot)
    runs = []
    for lineup in combinations(factories, 4):
        bots = [factory() for factory in lineup]
        for bot in bots:
            if isinstance(bot.decisions, MCTSDecisions):
                bot.decisions = TimedMCTS()
        game = Game(bots, seed=42)
        start = perf_counter()
        result = game.play()
        runs.append({'lineup': [f.__name__ for f in lineup],
                     'seconds': perf_counter() - start, 'result': asdict(result)})
    report = {
        'note': 'Development smoke/timing check: one seed, fixed lineup order, no strength claim.',
        'python': platform.python_version(), 'source_sha256': source_hash(),
        'mcts': {'iterations': 128, 'depth': 8, 'seed': 0},
        'games': len(runs), 'mean_game_seconds': mean(r['seconds'] for r in runs),
        'max_game_seconds': max(r['seconds'] for r in runs),
        'mcts_searches': len(timings), 'mean_search_ms': mean(timings) * 1000,
        'median_search_ms': median(timings) * 1000,
        'max_search_ms': max(timings) * 1000, 'runs': runs,
    }
    # Print JSON so callers choose where to save it without overwriting old runs.
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
