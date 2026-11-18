"""Analyze an E1 Monopoly tournament produced by monopoly/tournament.py.

Place at analysis/e1_analysis.py (no third-party packages required for CSVs).

Examples (from the repository root):
    python analysis/e1_analysis.py --input results/e1_tournament_5/games.csv --strict
    python analysis/e1_analysis.py --input results/e1_smoke/games.csv --plots

Outputs default to <input folder>/analysis/:
    wins_detail.csv       one row per game: winner, seat, matchup, all four bots
    overall_results.csv   each bot's appearances, wins, rate, wealth, bankruptcies
    matchup_results.csv   each bot's performance in each four-bot lineup
    seat_results.csv      each bot's performance in seats 1..4
    matchup_summary.csv   matchups, numbers of games, and wins for each bot
    validation.json       counts, completeness checks, and limitations
    trade_results.csv     accepted deals per bot, if trade logs contain offers
    win_rates.png         optional with --plots and matplotlib installed
    seat_win_rates.png    optional with --plots and matplotlib installed

Notes: Win-rate intervals use the project's simple normal approximation;
rotations with the same seed are correlated and should be block-bootstrapped
for formal statistical inference. Accepted-trade totals do NOT measure
per-bot offer acceptance rates: rejected offers are not logged by the engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

SEATS = (1, 2, 3, 4)


@dataclass(frozen=True)
class GameRow:
    game_id: str
    matchup_id: str
    seed: int
    rotation: int | None
    seats: tuple[str, str, str, str]
    winner: str
    winner_seat: int
    turns: int
    end_reason: str
    net_worth: tuple[float, float, float, float]
    bankrupt: tuple[int, int, int, int]
    trades_proposed: int
    trades_accepted: int
    seconds: float | None


def read_games(path: Path) -> list[GameRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing tournament file: {path}")
    records: list[GameRow] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"game_id", "matchup_id", "seed", "winner", "turns", "end_reason"}
        required |= {f"seat_{seat}" for seat in SEATS}
        required |= {f"net_worth_{seat}" for seat in SEATS}
        required |= {f"bankrupt_{seat}" for seat in SEATS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"games.csv is missing columns: {', '.join(sorted(missing))}")
        for line_no, row in enumerate(reader, start=2):
            try:
                gid = row["game_id"].strip()
                if not gid or gid in seen:
                    raise ValueError(f"empty or duplicate game_id: {gid!r}")
                seen.add(gid)
                seats = tuple(row[f"seat_{s}"].strip() for s in SEATS)
                if len(set(seats)) != 4 or any(not name for name in seats):
                    raise ValueError(f"expected four distinct, nonempty agents, got {seats}")
                # New E1 runner: winner=<agent>, winner_seat=<1..4>.
                # Also support experiment-style CSVs: winner=<seat number>.
                winner_value = row["winner"].strip()
                if row.get("winner_seat"):
                    winner_seat = int(row["winner_seat"])
                    if not 1 <= winner_seat <= 4:
                        raise ValueError("winner_seat must be between 1 and 4")
                    winner = seats[winner_seat - 1]
                    if winner_value != winner and winner_value != str(winner_seat):
                        raise ValueError(f"winner {winner_value!r} disagrees with winner_seat")
                else:
                    winner_seat = int(winner_value)
                    if not 1 <= winner_seat <= 4:
                        raise ValueError("winner must be a 1-based seat without winner_seat")
                    winner = seats[winner_seat - 1]
                net_worth = tuple(float(row[f"net_worth_{s}"]) for s in SEATS)
                bankrupt = tuple(int(row[f"bankrupt_{s}"]) for s in SEATS)
                if any(b not in (0, 1) for b in bankrupt):
                    raise ValueError("bankrupt columns must be 0/1")
                if bankrupt[winner_seat - 1]:
                    raise ValueError("winner cannot be bankrupt")
                turns = int(row["turns"])
                if turns < 0:
                    raise ValueError("turns must be nonnegative")
                proposed = int(row.get("trades_proposed") or 0)
                accepted = int(row.get("trades_accepted") or 0)
                if proposed < 0 or accepted < 0 or accepted > proposed:
                    raise ValueError("trade counts are inconsistent")
                records.append(GameRow(
                    game_id=gid,
                    matchup_id=row["matchup_id"].strip(),
                    seed=int(row["seed"]),
                    rotation=int(row["rotation"]) if row.get("rotation") else None,
                    seats=seats, winner=winner, winner_seat=winner_seat,
                    turns=turns, end_reason=row["end_reason"].strip(),
                    net_worth=net_worth, bankrupt=bankrupt,
                    trades_proposed=proposed, trades_accepted=accepted,
                    seconds=float(row["seconds"]) if row.get("seconds") else None,
                ))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Bad games.csv row {line_no}: {exc}") from exc
    if not records:
        raise ValueError("games.csv contains no game results")
    return records


def _ci_95(wins: int, games: int) -> tuple[float, float]:
    """Normal approximation used in the proposal; descriptive, not block-adjusted."""
    p = wins / games
    margin = 1.96 * math.sqrt(p * (1 - p) / games)
    return max(0.0, p - margin), min(1.0, p + margin)


def _aggregate(appearances: list[tuple[GameRow, int]]) -> dict[str, Any]:
    n = len(appearances)
    wins = sum(g.winner_seat == seat for g, seat in appearances)
    low, high = _ci_95(wins, n)
    return {
        "games_played": n,
        "wins": wins,
        "win_rate": round(wins / n, 6),
        "win_rate_pct": round(100 * wins / n, 2),
        "ci_95_low": round(low, 6),
        "ci_95_high": round(high, 6),
        "avg_net_worth": round(sum(g.net_worth[seat - 1] for g, seat in appearances) / n, 2),
        "bankruptcies": sum(g.bankrupt[seat - 1] for g, seat in appearances),
        "bankruptcy_rate": round(sum(g.bankrupt[seat - 1] for g, seat in appearances) / n, 6),
        "avg_game_turns": round(sum(g.turns for g, _ in appearances) / n, 2),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate(games: list[GameRow], manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Validate tournament integrity; incomplete batches are warnings, not errors."""
    warnings: list[str] = []
    if manifest is None:
        warnings.append("manifest.json is missing; planned game counts and agent coverage cannot be verified")
    lineups: dict[str, tuple[str, ...]] = {}
    per_matchup = Counter()
    per_matchup_seed: dict[tuple[str, int], set[int]] = defaultdict(set)
    seats_by_agent = Counter()
    agents = sorted({bot for game in games for bot in game.seats})

    for g in games:
        lineup = tuple(sorted(g.seats))
        if g.matchup_id in lineups and lineups[g.matchup_id] != lineup:
            raise ValueError(f"matchup_id {g.matchup_id} has inconsistent agent lineups")
        lineups[g.matchup_id] = lineup
        per_matchup[g.matchup_id] += 1
        for seat, agent in enumerate(g.seats, start=1):
            seats_by_agent[(agent, seat)] += 1
        if g.rotation is not None:
            key = (g.matchup_id, g.seed)
            if g.rotation in per_matchup_seed[key]:
                raise ValueError(f"duplicate seat rotation for matchup/seed {key}: {g.rotation}")
            per_matchup_seed[key].add(g.rotation)

    expected = manifest.get("games_planned") if manifest else None
    if expected is not None and len(games) != int(expected):
        warnings.append(f"Only {len(games)} of {expected} planned games are present")
    if manifest:
        manifest_agents = set(manifest.get("agents", []))
        if manifest_agents and set(agents) != manifest_agents:
            warnings.append(f"Agents in games {agents} differ from manifest {sorted(manifest_agents)}")
        if manifest_agents:
            expected_lineups = {tuple(sorted(x)) for x in combinations(manifest_agents, 4)}
            observed = set(lineups.values())
            if observed != expected_lineups:
                warnings.append(f"Matchups are incomplete: found {len(observed)}/{len(expected_lineups)} expected lineups")
        expected_per_matchup = int(manifest.get("seeds_per_matchup", 0)) * int(manifest.get("seat_rotations", 4))
        if expected_per_matchup:
            for matchup_id, count in sorted(per_matchup.items()):
                if count != expected_per_matchup:
                    warnings.append(f"{matchup_id}: {count} games rather than {expected_per_matchup}")
        if manifest_agents and manifest.get("seeds_per_matchup"):
            n_matchups_agent = math.comb(len(manifest_agents) - 1, 3)
            seat_expected = n_matchups_agent * int(manifest["seeds_per_matchup"])
            for agent in sorted(manifest_agents):
                for seat in SEATS:
                    observed = seats_by_agent[(agent, seat)]
                    if observed != seat_expected:
                        warnings.append(f"{agent}: seat {seat} played {observed} times; expected {seat_expected}")
    if per_matchup_seed:
        for key, rotations in sorted(per_matchup_seed.items()):
            if rotations != {0, 1, 2, 3}:
                warnings.append(f"{key[0]}, seed {key[1]}: seat rotations are {sorted(rotations)}")

    wins = Counter(game.winner for game in games)
    return {
        "games_found": len(games),
        "games_expected": expected,
        "complete": not warnings,
        "agents": agents,
        "matchups_found": len(lineups),
        "matchup_lineups": {key: list(value) for key, value in sorted(lineups.items())},
        "games_per_matchup": dict(sorted(per_matchup.items())),
        "wins_per_agent": {agent: wins[agent] for agent in agents},
        "total_wins": sum(wins.values()),
        "gamewide_trade_proposals": sum(g.trades_proposed for g in games),
        "gamewide_accepted_trades": sum(g.trades_accepted for g in games),
        "sum_game_elapsed_seconds": round(sum(g.seconds or 0.0 for g in games), 4),
        "warning_messages": warnings,
        "statistical_note": "95% intervals use the proposal's binomial normal approximation; seed/seat dependence is not accounted for.",
    }


def _read_trade_results(trade_path: Path, games: list[GameRow]) -> tuple[list[dict[str, Any]], list[str]]:
    """Summarize only confirmed trades. Doesn't infer rejected offers by bot."""
    if not trade_path.is_file():
        return [], ["No trades.jsonl found; agent trade statistics skipped"]
    by_id = {g.game_id: g for g in games}
    counts = Counter()
    count_per_game = Counter()
    skipped = 0
    with trade_path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            game = by_id.get(entry.get("game_id"))
            if game is None:
                # A journal/CSV may describe a partial batch; ignore unknown games.
                continue
            # Tournament records wrap the original zero-based engine seats.
            # Earlier flat pilot exports use one-based seats.
            if "accepted_trade" in entry:
                record = entry["accepted_trade"]
                offset = 0
            else:
                record = entry
                offset = 1
            offer = record.get("offer") if isinstance(record, dict) else None
            if not isinstance(offer, dict):
                skipped += 1
                continue
            proposer = int(offer["proposer"]) - offset
            recipient = int(offer["recipient"]) - offset
            if not 0 <= proposer < 4 or not 0 <= recipient < 4 or proposer == recipient:
                raise ValueError(f"Invalid trade participant on line {line_no}")
            counts[(game.seats[proposer], "final_proposer")] += 1
            counts[(game.seats[recipient], "final_recipient")] += 1
            counts[(game.seats[proposer], "involved")] += 1
            counts[(game.seats[recipient], "involved")] += 1
            if record.get("initiated_by") is not None:
                initial_seat = int(record["initiated_by"]) - offset
                if 0 <= initial_seat < 4:
                    counts[(game.seats[initial_seat], "initiated")] += 1
            count_per_game[game.game_id] += 1
    warnings = []
    if skipped:
        warnings.append(f"Skipped {skipped} trade records lacking offer details; per-bot trade statistics omitted")
    mismatches = [g.game_id for g in games if count_per_game[g.game_id] != g.trades_accepted]
    if mismatches and not skipped:
        warnings.append(f"Accepted trade log disagrees with games.csv for {len(mismatches)} games")
    names = sorted({bot for g in games for bot in g.seats})
    if skipped:
        return [], warnings
    rows = [
        {"agent": name,
         "accepted_trades_involved": counts[(name, "involved")],
         "accepted_trades_initiated": counts[(name, "initiated")],
         "accepted_trades_as_final_proposer": counts[(name, "final_proposer")],
         "accepted_trades_as_final_recipient": counts[(name, "final_recipient")]}
        for name in names
    ]
    return rows, warnings


def analyze(input_csv: Path, output_dir: Path, *, strict: bool = False, plots: bool = False) -> dict[str, Any]:
    games = read_games(input_csv)
    manifest_path = input_csv.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    checks = validate(games, manifest)
    if strict and checks["warning_messages"]:
        raise ValueError("Tournament validation failed:\n- " + "\n- ".join(checks["warning_messages"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    appearances: dict[tuple[str, ...], list[tuple[GameRow, int]]] = defaultdict(list)
    by_matchup: dict[str, list[GameRow]] = defaultdict(list)
    win_details: list[dict[str, Any]] = []
    for g in games:
        by_matchup[g.matchup_id].append(g)
        win_details.append({
            "game_id": g.game_id, "matchup_id": g.matchup_id, "seed": g.seed,
            "rotation": "" if g.rotation is None else g.rotation,
            "matchup_agents": ", ".join(sorted(g.seats)),
            **{f"seat_{s}": g.seats[s-1] for s in SEATS},
            "winner_bot": g.winner, "winner_seat": g.winner_seat,
            "turns": g.turns, "end_reason": g.end_reason,
            "trades_proposed": g.trades_proposed, "trades_accepted": g.trades_accepted,
        })
        for seat, name in enumerate(g.seats, start=1):
            appearances[("overall", name)].append((g, seat))
            appearances[("matchup", g.matchup_id, name)].append((g, seat))
            appearances[("seat", name, str(seat))].append((g, seat))

    names = checks["agents"]
    overall = [{"agent": agent, **_aggregate(appearances[("overall", agent)])} for agent in names]
    matchup = []
    for mid in sorted(by_matchup):
        for agent in sorted(checks["matchup_lineups"][mid]):
            matchup.append({
                "matchup_id": mid, "matchup_agents": ", ".join(checks["matchup_lineups"][mid]),
                "agent": agent, **_aggregate(appearances[("matchup", mid, agent)]),
            })
    seat = [
        {"agent": agent, "seat_number": number, **_aggregate(appearances[("seat", agent, str(number))])}
        for agent in names for number in SEATS
        if appearances[("seat", agent, str(number))]
    ]
    summary = []
    for mid in sorted(by_matchup):
        lineup = checks["matchup_lineups"][mid]
        row = {"matchup_id": mid, "matchup_agents": ", ".join(lineup), "games": len(by_matchup[mid])}
        for agent in names:
            row[f"wins_{agent}"] = sum(g.winner == agent for g in by_matchup[mid]) if agent in lineup else ""
        summary.append(row)

    stats = ["games_played", "wins", "win_rate", "win_rate_pct", "ci_95_low", "ci_95_high",
             "avg_net_worth", "bankruptcies", "bankruptcy_rate", "avg_game_turns"]
    _write_csv(output_dir / "wins_detail.csv", win_details,
               ["game_id", "matchup_id", "seed", "rotation", "matchup_agents", *[f"seat_{s}" for s in SEATS],
                "winner_bot", "winner_seat", "turns", "end_reason", "trades_proposed", "trades_accepted"])
    _write_csv(output_dir / "overall_results.csv", overall, ["agent", *stats])
    _write_csv(output_dir / "matchup_results.csv", matchup, ["matchup_id", "matchup_agents", "agent", *stats])
    _write_csv(output_dir / "seat_results.csv", seat, ["agent", "seat_number", *stats])
    _write_csv(output_dir / "matchup_summary.csv", summary,
               ["matchup_id", "matchup_agents", "games", *[f"wins_{agent}" for agent in names]])

    trade_results, trade_warnings = _read_trade_results(input_csv.parent / "trades.jsonl", games)
    if trade_results:
        _write_csv(output_dir / "trade_results.csv", trade_results,
                   ["agent", "accepted_trades_involved", "accepted_trades_initiated",
                    "accepted_trades_as_final_proposer", "accepted_trades_as_final_recipient"])
    checks["warning_messages"].extend(trade_warnings)
    checks["trade_log_warnings"] = trade_warnings
    (output_dir / "validation.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    if plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("Plots require matplotlib: python -m pip install matplotlib") from exc
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([r["agent"] for r in overall], [r["win_rate_pct"] for r in overall])
        ax.axhline(25.0, linestyle="--", label="Equal-chance reference (25%)")
        ax.set(ylabel="Win rate (%)", title="E1: wins by tournament agent")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "win_rates.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for bot in names:
            rows = [r for r in seat if r["agent"] == bot]
            ax.plot([r["seat_number"] for r in rows], [r["win_rate_pct"] for r in rows], marker="o", label=bot)
        ax.axhline(25.0, linestyle="--", label="Equal-chance reference (25%)")
        ax.set(xticks=list(SEATS), xlabel="Starting seat", ylabel="Win rate (%)",
               title="E1: win rates by starting seat")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "seat_win_rates.png", dpi=160)
        plt.close(fig)

    print(f"Analyzed {len(games)} games across {len(by_matchup)} matchups.")
    print(f"Results saved in: {output_dir.resolve()}")
    print(f"{'Agent':<18} {'Played':>6} {'Wins':>6} {'Win rate':>10}")
    for r in overall:
        print(f"{r['agent']:<18} {r['games_played']:>6} {r['wins']:>6} {r['win_rate_pct']:>9.2f}%")
    if checks["warning_messages"]:
        print(f"Warnings ({len(checks['warning_messages'])}): see validation.json")
        for warning in checks["warning_messages"][:4]:
            print(f"  - {warning}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the E1 Monopoly tournament CSV")
    parser.add_argument("--input", type=Path, default=Path("results/e1_tournament_5/games.csv"),
                        help="Path to games.csv; manifest.json/trades.jsonl read from its folder")
    parser.add_argument("--output", type=Path, default=None, help="Destination (default: beside games.csv, in analysis/)")
    parser.add_argument("--strict", action="store_true", help="Require all planned games/matchups/rotations to be present")
    parser.add_argument("--plots", action="store_true", help="Also generate PNG charts (requires matplotlib)")
    args = parser.parse_args()
    input_csv = args.input / "games.csv" if args.input.is_dir() else args.input
    output = args.output or input_csv.parent / "analysis"
    analyze(input_csv, output, strict=args.strict, plots=args.plots)


if __name__ == "__main__":
    main()
