"""E1 round-robin tournament for the simplified Monopoly simulator.

Place this file at monopoly/tournament.py and run:
    python -m monopoly.tournament --output results/e1_tournament_5 --workers 4

Existing engine API assumed (check the short _run_one adapter if yours differs):
    game = Game(bots, seed=seed)
    outcome = game.play()  # outcome.winner is a zero-based seat index

The run journal is authoritative; CSV/JSONL reports are derived atomically from it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, is_dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import zipfile

# All of these names are already used by the existing project's bot factories.
from .board import BOARD
from .bots import (
    PolicyBot,
    baseline_bot,
    cooperator_bot,
    reciprocator_bot,
    risk_averse_bot,
)
from .advanced_bargaining import BayesianTrading, ZeuthenTrading
from .advanced_decisions import ExpectimaxDecisions
from .engine import Game

# SAFETY SWITCH: Change to True only after MCTSDecisions.buy/build are implemented
# and checked. --include-mcts alone cannot silently enable a placeholder agent.
MCTS_IMPLEMENTED = False

FIVE_AGENTS = (
    "baseline",
    "cooperator",
    "reciprocator",
    "hard_bargainer",
    "risk_averse",
)

GAME_FIELDS = (
    "game_id", "matchup_id", "seed", "rotation", "seat_1", "seat_2",
    "seat_3", "seat_4", "winner", "winner_seat", "turns",
    "end_reason", "net_worth_1", "net_worth_2", "net_worth_3",
    "net_worth_4", "bankrupt_1", "bankrupt_2", "bankrupt_3",
    "bankrupt_4", "trades_proposed", "trades_accepted", "seconds",
)


@dataclass(frozen=True)
class Case:
    game_id: str
    matchup_id: str
    seed: int
    rotation: int
    seats: tuple[str, str, str, str]
    depth: int
    max_nodes: int


def make_schedule(
    *, include_mcts: bool, seed_start: int, seeds_per_matchup: int,
    depth: int, max_nodes: int,
) -> list[Case]:
    """Five groups x 50 seeds x 4 rotations = 1,000 games by default."""
    if seeds_per_matchup < 1 or seed_start < 0:
        raise ValueError("seeds_per_matchup >= 1 and seed_start >= 0 required")
    if depth not in (1, 2, 3) or max_nodes < 1:
        raise ValueError("Expectimax depth must be 1..3; max_nodes must be positive")
    if include_mcts and not MCTS_IMPLEMENTED:
        raise RuntimeError(
            "MCTS is not ready. Implement MCTSDecisions, set "
            "MCTS_IMPLEMENTED=True, and rerun --include-mcts."
        )
    agents = FIVE_AGENTS + (("adaptive",) if include_mcts else ())
    schedule = []
    for matchup_no, lineup in enumerate(combinations(agents, 4), start=1):
        for seed in range(seed_start, seed_start + seeds_per_matchup):
            for rotation in range(4):
                seats = lineup[rotation:] + lineup[:rotation]
                schedule.append(Case(
                    game_id=f"e1-m{matchup_no:02d}-s{seed}-r{rotation}",
                    matchup_id=f"m{matchup_no:02d}",
                    seed=seed,
                    rotation=rotation,
                    seats=seats,
                    depth=depth,
                    max_nodes=max_nodes,
                ))
    return schedule


def _make_bot(name: str, depth: int, max_nodes: int):
    if name == "baseline":
        return baseline_bot()
    if name == "cooperator":
        return cooperator_bot()
    if name == "reciprocator":
        return reciprocator_bot()
    if name == "hard_bargainer":
        # Identical pairing to hard_bargainer_bot(use_expectimax=True), but
        # allows the tournament to explicitly specify and record the search budget.
        return PolicyBot(ExpectimaxDecisions(depth=depth, max_nodes=max_nodes),
                         ZeuthenTrading())
    if name == "risk_averse":
        return risk_averse_bot(use_prospect_theory=True)
    if name == "adaptive":
        if not MCTS_IMPLEMENTED:
            raise RuntimeError("Refusing to run Adaptive with an incomplete MCTS")
        from .advanced_decisions import MCTSDecisions
        return PolicyBot(MCTSDecisions(), BayesianTrading())
    raise ValueError(f"Unrecognized agent: {name}")


def _read_field(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _jsonable(value):
    """Convert accepted-trade records to JSON without dropping their fields."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):  # e.g. enum
        return _jsonable(value.value)
    return str(value)


def _net_worth(game: Game, seat: int) -> int:
    """Same accounting as the project's 150-turn ranking rule."""
    player = game.players[seat]
    if player.bankrupt:
        return 0
    total = player.cash
    for square, owner in enumerate(game.owners):
        if owner == seat:
            prop = BOARD[square].property
            if prop is not None:
                total += prop.price + game.buildings[square] * prop.build_cost
    return int(total)


def _run_one(case: Case) -> dict:
    """Worker: fresh agents, board and RNG for every game. No file writes."""
    t0 = perf_counter()
    bots = [_make_bot(name, case.depth, case.max_nodes) for name in case.seats]

    # ENGINE ADAPTER: if your Game constructor or play return differs,
    # these are the only two statements you need to align with __main__.py.
    game = Game(bots, seed=case.seed)
    outcome = game.play()

    winner = _read_field(outcome, "winner", "winner_index", "winner_seat")
    if winner is None:
        winner = _read_field(game, "winner", "winner_index")
    if winner is None and isinstance(outcome, int) and not isinstance(outcome, bool):
        winner = outcome
    if not isinstance(winner, int) or isinstance(winner, bool) or not 0 <= winner < 4:
        raise RuntimeError(
            f"Game {case.game_id}: expected a zero-based winner seat 0..3; "
            f"got {winner!r}. Check _run_one's engine adapter."
        )

    turns = _read_field(outcome, "turns", default=getattr(game, "turns", None))
    if turns is None:
        raise RuntimeError(f"Game {case.game_id}: cannot locate turn count")
    reason = _read_field(outcome, "end_reason")
    if reason is None:
        reason = "last player standing" if sum(not p.bankrupt for p in game.players) == 1 else "turn limit"
    if hasattr(reason, "value"):
        reason = reason.value

    accepted_history = list(getattr(game, "trade_history", []))
    proposals = _read_field(outcome, "trades_proposed", "trade_attempts", "trade_proposals", "negotiation_attempts")
    if proposals is None:
        proposals = _read_field(game, "trades_proposed", "trade_attempts", "trade_proposals", "negotiation_attempts")
    # Do not fabricate 0 if your engine does not expose attempts.

    row = {
        "game_id": case.game_id,
        "matchup_id": case.matchup_id,
        "seed": case.seed,
        "rotation": case.rotation,
        **{f"seat_{i+1}": name for i, name in enumerate(case.seats)},
        "winner": case.seats[winner],
        "winner_seat": winner + 1,
        "turns": int(turns),
        "end_reason": str(reason),
        **{f"net_worth_{i+1}": _net_worth(game, i) for i in range(4)},
        **{f"bankrupt_{i+1}": int(p.bankrupt) for i, p in enumerate(game.players)},
        "trades_proposed": proposals,
        "trades_accepted": len(accepted_history),
        "seconds": round(perf_counter() - t0, 5),
    }
    trades = [
        {"game_id": case.game_id, "matchup_id": case.matchup_id,
         "accepted_trade": _jsonable(record)}
        for record in accepted_history
    ]
    return {"row": row, "trades": trades}


def _source_fingerprint(repo_root: Path) -> str:
    h = sha256()
    sources = sorted((repo_root / "monopoly").rglob("*.py"))
    if not sources:
        raise RuntimeError(f"No Python package found in {repo_root / 'monopoly'}")
    for file in sources:
        h.update(file.relative_to(repo_root).as_posix().encode())
        h.update(b"\0")
        h.update(file.read_bytes())
    return h.hexdigest()


def _write_source_zip(repo_root: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in ("monopoly", "tests"):
            folder = repo_root / directory
            if folder.exists():
                for file in sorted(folder.rglob("*.py")):
                    archive.write(file, file.relative_to(repo_root))
        for filename in ("README.md", "run.cmd", "requirements.txt"):
            file = repo_root / filename
            if file.is_file():
                archive.write(file, filename)


def _write_json_atomic(target: Path, obj) -> None:
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, target)


def _write_csv_atomic(target: Path, columns: tuple[str, ...], rows: list[dict]):
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, target)


def _ci(wins: int, games: int) -> tuple[float, float]:
    """Proposal's normal-approximation 95% interval, clipped to [0,1]."""
    if games == 0:
        return (0.0, 0.0)
    p = wins / games
    halfwidth = 1.96 * math.sqrt(p * (1 - p) / games)
    return (max(0.0, p - halfwidth), min(1.0, p + halfwidth))


def _report_rows(game_rows: list[dict], names: tuple[str, ...]):
    summary, matchup = [], []
    matchups = sorted({r["matchup_id"] for r in game_rows})
    for name in names:
        appearances = [(r, i) for r in game_rows for i in range(4)
                       if r[f"seat_{i+1}"] == name]
        if not appearances:
            continue
        wins = sum(r["winner"] == name for r, _ in appearances)
        low, high = _ci(wins, len(appearances))
        summary.append({
            "agent": name, "games": len(appearances), "wins": wins,
            "win_rate": round(wins / len(appearances), 6),
            "ci_95_low": round(low, 6), "ci_95_high": round(high, 6),
            "avg_net_worth": round(sum(r[f"net_worth_{i+1}"]
                                       for r, i in appearances) / len(appearances), 2),
            "avg_turns": round(sum(r["turns"] for r, _ in appearances) / len(appearances), 2),
            "bankruptcy_rate": round(sum(r[f"bankrupt_{i+1}"]
                                         for r, i in appearances) / len(appearances), 6),
        })
    for matchup_id in matchups:
        for name in names:
            appearances = [(r, i) for r in game_rows for i in range(4)
                           if r["matchup_id"] == matchup_id and r[f"seat_{i+1}"] == name]
            if not appearances:
                continue
            wins = sum(r["winner"] == name for r, _ in appearances)
            low, high = _ci(wins, len(appearances))
            matchup.append({
                "matchup_id": matchup_id, "agents": ",".join(sorted(
                    {r[f"seat_{i+1}"] for i in range(4) for r, _ in appearances[:1]})),
                "agent": name, "games": len(appearances), "wins": wins,
                "win_rate": round(wins / len(appearances), 6),
                "ci_95_low": round(low, 6), "ci_95_high": round(high, 6),
                "avg_net_worth": round(sum(r[f"net_worth_{i+1}"]
                                           for r, i in appearances) / len(appearances), 2),
            })
    return summary, matchup


def _export(directory: Path, completed: dict[str, dict], schedule: list[Case], names: tuple[str, ...]):
    # Deterministic order regardless of which parallel worker finishes first.
    records = [completed[c.game_id] for c in schedule if c.game_id in completed]
    rows = [record["row"] for record in records]
    _write_csv_atomic(directory / "games.csv", GAME_FIELDS, rows)
    summaries, matchups = _report_rows(rows, names)
    _write_csv_atomic(directory / "summary.csv", (
        "agent", "games", "wins", "win_rate", "ci_95_low", "ci_95_high",
        "avg_net_worth", "avg_turns"), summaries)
    _write_csv_atomic(directory / "matchups.csv", (
        "matchup_id", "agents", "agent", "games", "wins", "win_rate",
        "ci_95_low", "ci_95_high", "avg_net_worth"), matchups)
    tmp = directory / "trades.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as out:
        for record in records:
            for trade in record["trades"]:
                out.write(json.dumps(trade, sort_keys=True) + "\n")
    os.replace(tmp, directory / "trades.jsonl")
    _write_json_atomic(directory / "progress.json", {
        "completed": len(completed), "scheduled": len(schedule),
        "remaining": len(schedule) - len(completed),
    })


def _load_journal(path: Path, allowed_ids: set[str]) -> dict[str, dict]:
    if not path.is_file():
        return {}
    # A process killed during a write can leave one partial final JSON line.
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1] if b"\n" in data else b""
        path.write_bytes(data)
    completed = {}
    for line in data.splitlines():
        record = json.loads(line)
        gid = record["row"]["game_id"]
        if gid not in allowed_ids or gid in completed:
            raise RuntimeError(f"Unexpected or repeated game ID in journal: {gid}")
        completed[gid] = record
    return completed


def run_tournament(
    *, output: Path, include_mcts: bool = False, seed_start: int = 1000,
    seeds_per_matchup: int = 50, depth: int = 1, max_nodes: int = 200,
    workers: int = 1, export_every: int = 25, resume: bool = False,
) -> None:
    if workers < 1 or export_every < 1:
        raise ValueError("workers and export_every must be >= 1")
    schedule = make_schedule(
        include_mcts=include_mcts, seed_start=seed_start,
        seeds_per_matchup=seeds_per_matchup, depth=depth, max_nodes=max_nodes)
    names = FIVE_AGENTS + (("adaptive",) if include_mcts else ())
    root = Path(__file__).resolve().parent.parent
    config = {
        "experiment": "E1 tournament", "include_mcts": include_mcts,
        "agents": list(names), "matchups": len(set(c.matchup_id for c in schedule)),
        "seeds_per_matchup": seeds_per_matchup, "seed_start": seed_start,
        "seat_rotations": 4, "games_planned": len(schedule),
        "expectimax_depth": depth, "expectimax_max_nodes": max_nodes,
        "source_sha256": _source_fingerprint(root),
        "python": sys.version, "platform": platform.platform(),
        "game_turn_limit": 150,
    }
    output = output.resolve()
    if resume:
        if not (output / "manifest.json").is_file():
            raise RuntimeError("Cannot resume: manifest.json not found")
        old_config = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if old_config != config:
            raise RuntimeError("Cannot resume with changed code or tournament settings")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output already exists: {output}; use --resume")
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output / "manifest.json", config)
        _write_source_zip(root, output / "source.zip")

    journal_path = output / "games.journal.jsonl"
    completed = _load_journal(journal_path, {c.game_id for c in schedule})
    todo = [c for c in schedule if c.game_id not in completed]
    print(f"E1: {len(schedule)} total games; {len(completed)} done; "
          f"{len(todo)} remaining; {workers} worker(s)", flush=True)

    def save(record):
        gid = record["row"]["game_id"]
        if gid in completed:
            raise RuntimeError(f"Duplicate completed game: {gid}")
        # Append each completed game exactly once; this is the resume checkpoint.
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        completed[gid] = record
        if len(completed) % export_every == 0:
            _export(output, completed, schedule, names)
            print(f"Saved {len(completed)}/{len(schedule)} games", flush=True)

    try:
        if workers == 1:
            for case in todo:
                save(_run_one(case))
        else:
            # Keep only a few games in flight. Ctrl+C must not leave hundreds
            # of queued games running before the checkpoint can be used.
            pool = ProcessPoolExecutor(max_workers=workers)
            pending = set()
            remaining = iter(todo)
            try:
                for _ in range(workers * 2):
                    case = next(remaining, None)
                    if case is not None:
                        pending.add(pool.submit(_run_one, case))
                while pending:
                    future = next(as_completed(pending))
                    pending.remove(future)
                    save(future.result())
                    case = next(remaining, None)
                    if case is not None:
                        pending.add(pool.submit(_run_one, case))
            except BaseException:
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
    finally:
        _export(output, completed, schedule, names)

    print(f"Finished. Results in: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E1 four-player tournament")
    parser.add_argument("--include-mcts", action="store_true",
                        help="Enable Adaptive only when MCTS_IMPLEMENTED=True")
    parser.add_argument("--output", type=Path, default=Path("results/e1_tournament_5"))
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seeds-per-matchup", type=int, default=50)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--export-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_tournament(
        output=args.output, include_mcts=args.include_mcts,
        seed_start=args.seed_start, seeds_per_matchup=args.seeds_per_matchup,
        depth=args.depth, max_nodes=args.max_nodes, workers=args.workers,
        export_every=args.export_every, resume=args.resume,
    )


if __name__ == "__main__":
    main()
