"""Run, summarize, and replay the small controlled trading experiment."""

import csv
import hashlib
import json
import platform
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .bots import (ExpectedValueDecisions, NoTrading, PolicyBot, adaptive_bot, baseline_bot,
                   cooperator_bot, hard_bargainer_bot, reciprocator_bot, risk_averse_bot)
from .engine import Game


STRATEGIES = ("no_trade", "cooperator", "reciprocator", "hard_bargainer", "adaptive", "risk_averse")
PILOT_STRATEGIES = STRATEGIES[:3]
SCHEMA_VERSION = 1


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_game(strategy: str, seed: int, seat: int, trace: Callable[[str], None] | None = None) -> Game:
    if strategy not in STRATEGIES or not 0 <= seat < 4:
        raise ValueError("Choose a pilot strategy and a seat from 0 to 3.")
    factories = {
        "no_trade": lambda: PolicyBot(ExpectedValueDecisions(), NoTrading()),
        "cooperator": cooperator_bot,
        "reciprocator": reciprocator_bot,
        "hard_bargainer": lambda: hard_bargainer_bot(use_expectimax=False),
        "adaptive": adaptive_bot,
        "risk_averse": risk_averse_bot,
    }
    bots = [baseline_bot() for _ in range(4)]
    bots[seat] = factories[strategy]()
    return Game(bots, seed=seed, trace=trace)


def run_case(strategy: str, seed: int, seat: int, trace: Callable[[str], None] | None = None) -> tuple[dict, list[dict]]:
    game_id = f"{strategy}-seed{seed}-seat{seat + 1}"
    game = make_game(strategy, seed, seat, trace)
    result = game.play()
    row = {
        "game_id": game_id,
        "matchup_id": f"{strategy}_vs_3_baselines",
        "strategy": strategy,
        "seed": seed,
        "test_seat": seat + 1,
        "turn_limit": game.turn_limit,
        "winner": result.winner + 1,
        "test_won": int(result.winner == seat),
        "turns": result.turns,
        "end_reason": result.end_reason,
        "trades_proposed": result.trades_proposed,
        "trades_accepted": result.trades_accepted,
        "test_trades_accepted": sum(
            seat in (record.offer.proposer, record.offer.recipient) for record in game.trade_history
        ),
    }
    for player in range(4):
        row[f"seat_{player + 1}"] = strategy if player == seat else "baseline"
        row[f"net_worth_{player + 1}"] = result.net_worth[player]
        row[f"bankrupt_{player + 1}"] = int(result.bankrupt[player])
    trades = []
    for record in game.trade_history:
        trade = asdict(record)
        trade["game_id"] = game_id
        # CSV seats use 1–4. Use the same seat convention in the trade log.
        trade["initiated_by"] += 1
        trade["offer"]["proposer"] += 1
        trade["offer"]["recipient"] += 1
        trades.append(trade)
    return row, trades


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for strategy in STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        if not group:
            continue
        wins = sum(int(row["test_won"]) for row in group)
        summary.append({
            "strategy": strategy,
            "games": len(group),
            "wins": wins,
            "win_rate": wins / len(group),
            "mean_net_worth": sum(
                int(row[f"net_worth_{row['test_seat']}"]) for row in group
            ) / len(group),
            "test_trades_accepted": sum(int(row["test_trades_accepted"]) for row in group),
        })
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_pilot(
    output: Path, progress: Callable[[str], None] = print,
    *, seed_start: int = 0, seed_count: int = 10,
    strategies: tuple[str, ...] = PILOT_STRATEGIES,
) -> list[dict]:
    if type(seed_start) is not int or seed_start < 0 or type(seed_count) is not int or seed_count < 1:
        raise ValueError("Use a nonnegative starting seed and a positive seed count.")
    if not strategies or any(s not in STRATEGIES for s in strategies) or len(set(strategies)) != len(strategies):
        raise ValueError("Choose distinct supported strategies.")
    if output.exists():
        raise FileExistsError(f"{output} already exists. Use --output with a new folder.")
    version = source_hash()
    rows, trades = [], []
    for strategy in strategies:
        for seed in range(seed_start, seed_start + seed_count):
            for seat in range(4):
                row, records = run_case(strategy, seed, seat)
                rows.append(row)
                trades.extend(records)
        progress(f"Finished {seed_count * 4} games for {strategy}.")
    if source_hash() != version:
        raise RuntimeError("The source changed during the run. Rerun with a stable version.")
    summary = summarize(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the complete directory only after every game and file succeeds.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".pilot-") as temporary:
        bundle = Path(temporary) / "results"
        bundle.mkdir()
        _write_csv(bundle / "games.csv", rows)
        _write_csv(bundle / "summary.csv", summary)
        with (bundle / "trades.jsonl").open("w", encoding="utf-8") as stream:
            for trade in trades:
                stream.write(json.dumps(trade, sort_keys=True) + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION, "source_sha256": version,
            "python_version": platform.python_version(),
            "seeds": list(range(seed_start, seed_start + seed_count)), "test_seats": [1, 2, 3, 4],
            "strategies": list(strategies), "games": len(rows), "turn_limit": 150,
            "note": "Pilot only. Seeds repeat across seats and strategies.",
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(bundle / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(Path(__file__).parent.glob("*.py")):
                archive.write(path, f"monopoly/{path.name}")
        if source_hash() != version:
            raise RuntimeError("Source changed while saving the run. Rerun with a stable version.")
        bundle.rename(output)
    return summary


def print_summary(summary: list[dict]) -> None:
    print("Strategy        Wins / Games   Win rate   Test-bot trades")
    for row in summary:
        print(f"{row['strategy']:15} {row['wins']:2} / {row['games']:2}        "
              f"{row['win_rate']:6.1%}       {row['test_trades_accepted']}")


def replay(folder: Path, game_id: str, trace: Callable[[str], None] = print) -> None:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["source_sha256"] != source_hash():
        raise ValueError("This result uses a different code version. Replay it with that version.")
    with (folder / "games.csv").open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row["game_id"] == game_id]
    if len(matches) != 1:
        raise ValueError("Game ID must match exactly one saved game.")
    saved = matches[0]
    row, trades = run_case(saved["strategy"], int(saved["seed"]), int(saved["test_seat"]) - 1, trace)
    if saved != {key: str(value) for key, value in row.items()}:
        raise ValueError("Replayed game differs from the saved result.")
    saved_trades = [
        json.loads(line) for line in (folder / "trades.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    saved_trades = [trade for trade in saved_trades if trade["game_id"] == game_id]
    if saved_trades != json.loads(json.dumps(trades)):
        raise ValueError("Replayed trades differ from the saved records.")
    trace(f"Replay matched the result and all {len(trades)} accepted trades for {game_id}.")
