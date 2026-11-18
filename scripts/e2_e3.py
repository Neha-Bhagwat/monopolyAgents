"""E2 robustness and E3 surrogate equilibrium from frozen E1; no new games.

python -B -m scripts.e2_e3 --experiment all
"""
import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from importlib.metadata import distributions as installed_distributions
from pathlib import Path
import platform
from statistics import mean, pstdev
import tempfile
import zipfile

from monopoly.research_io import archive_source, digest, write_csv, write_json
from scripts.freeze_e1 import verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_e1(folder):
    require(__debug__, 'Run without -O: the frozen E1 audit uses assertions')
    report = verify(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    with (folder / 'games.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    agents = manifest['agents']
    lineups, counts, wins = {}, Counter(), Counter()
    for row in rows:
        m = row['matchup_id']
        participants = frozenset(row[f'seat_{s}'] for s in range(1, 5))
        require(m not in lineups or lineups[m] == participants, 'Lineup changed within matchup')
        lineups[m] = participants
        counts[m] += 1
        wins[m, row['winner']] += 1
    require(len(lineups) == 15 and all(n == 200 for n in counts.values()), 'Unbalanced lineups')
    rates = {(m, a): wins[m, a] / counts[m] for m in lineups for a in lineups[m]}
    require(all(sum(wins[m, a] for a in lineups[m]) == 200 for m in lineups), 'Invalid win totals')
    return agents, lineups, rates, wins, report


def robustness(agents, lineups, rates):
    detailed, summaries, table = [], [], []
    for agent in agents:
        matchups = sorted(m for m in lineups if agent in lineups[m])
        require(len(matchups) == 10, 'Each agent must appear in ten lineups')
        values = [rates[m, agent] for m in matchups]
        require(all(0 <= p <= 1 for p in values), 'Invalid win rate')
        for m in matchups:
            detailed.append({'matchup_id': m, 'agent': agent,
                             'lineup': ', '.join(sorted(lineups[m])), 'games': 200,
                             'win_rate': rates[m, agent]})
        summaries.append({'agent': agent, 'lineups': 10, 'games': 2000,
                          'mean_win_rate': mean(values), 'worst_case_win_rate': min(values),
                          'population_sd': pstdev(values), 'best_case_win_rate': max(values),
                          'worst_lineups': ','.join(m for m in matchups if rates[m, agent] == min(values))})
        table.append({'agent': agent, **{m: rates.get((m, agent), '') for m in sorted(lineups)}})
    summaries.sort(key=lambda r: (-r['worst_case_win_rate'], -r['mean_win_rate'], r['agent']))
    return summaries, detailed, table


def payoff(agents, lineups, rates):
    import numpy as np
    matrix = np.zeros((len(agents), len(agents)))
    contributions = []
    for i, j in combinations(range(len(agents)), 2):
        first, second = agents[i], agents[j]
        shared = sorted(m for m in lineups if {first, second} <= lineups[m])
        require(len(shared) == 6, 'Each pair requires six shared lineups')
        deltas = []
        for m in shared:
            delta = rates[m, first] - rates[m, second]
            deltas.append(delta)
            contributions.append({'row_agent': first, 'column_agent': second, 'matchup_id': m,
                                  'row_win_rate': rates[m, first], 'column_win_rate': rates[m, second],
                                  'difference': delta, 'weight': 1 / 6})
        matrix[i, j] = mean(deltas)
        matrix[j, i] = -matrix[i, j]
    require(np.allclose(matrix, -matrix.T, atol=1e-12, rtol=0), 'Matrix not antisymmetric')
    require(np.all(np.diag(matrix) == 0), 'Nonzero diagonal')
    return matrix, contributions


def equilibrium(matrix, tolerance=1e-9):
    """Solve primal/dual zero-sum LPs and independently check best responses."""
    import numpy as np
    from scipy.optimize import linprog
    a = np.asarray(matrix, dtype=float)
    require(a.ndim == 2 and a.shape[0] == a.shape[1] and len(a) > 0, 'Square nonempty matrix required')
    require(np.isfinite(a).all(), 'Nonfinite payoff')
    require(0 < tolerance < 1, 'Invalid tolerance')
    n = len(a)
    row = linprog([0.] * n + [-1.], A_ub=np.column_stack((-a.T, np.ones(n))), b_ub=np.zeros(n),
                  A_eq=[[1.] * n + [0.]], b_eq=[1.], bounds=[(0, None)] * n + [(None, None)], method='highs')
    col = linprog([0.] * n + [1.], A_ub=np.column_stack((a, -np.ones(n))), b_ub=np.zeros(n),
                  A_eq=[[1.] * n + [0.]], b_eq=[1.], bounds=[(0, None)] * n + [(None, None)], method='highs')
    require(row.success and col.success, f'Solver failed: {row.message}; {col.message}')
    p, q = row.x[:n], col.x[:n]
    require(min(p.min(), q.min()) >= -tolerance, 'Negative probability')
    require(abs(p.sum() - 1) <= tolerance and abs(q.sum() - 1) <= tolerance, 'Probabilities do not sum to one')
    value = float(p @ a @ q)
    lower, upper = float(np.min(p @ a)), float(np.max(a @ q))
    gap = upper - lower
    require(gap <= tolerance and lower - tolerance <= value <= upper + tolerance, 'Equilibrium certificate failed')
    return {'row_distribution': p.tolist(), 'column_distribution': q.tolist(),
            'expected_row_payoff': value, 'row_guarantee': lower, 'column_upper_bound': upper,
            'best_response_gap': gap, 'tolerance': tolerance,
            'row_lp_value': float(row.x[-1]), 'column_lp_value': float(col.x[-1]),
            'solver': 'scipy.optimize.linprog / HiGHS', 'row_solver_status': row.message,
            'column_solver_status': col.message,
            'interpretation': 'One equilibrium of the constructed pairwise zero-sum game, not Monopoly itself. Uniqueness is not claimed.',
            'alpha_rank_implemented': False}


def heatmap(values, rows, columns, path, title, diverging=False):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.array(values, dtype=float) * 100
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * .75), 5.5))
    cmap = plt.get_cmap('RdBu_r' if diverging else 'YlGnBu').with_extremes(bad='#e8e8e8')
    limit = max(float(np.nanmax(abs(data))), .01)
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect='auto',
                      vmin=-limit if diverging else 0, vmax=limit)
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha='right')
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, pad=16)
    for i in range(len(rows)):
        for j in range(len(columns)):
            value = data[i, j]
            if np.isfinite(value):
                rgba = image.cmap(image.norm(value))
                light = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if light < .45 else 'black')
    fig.colorbar(image, ax=ax, label='Row advantage (percentage points)' if diverging else 'Win rate (%)')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate(experiment, config_path, output=None):
    import matplotlib
    import numpy as np
    import scipy
    require(experiment in ('e2', 'e3'), 'Unknown experiment')
    config = json.loads(Path(config_path).read_text())
    require(config['schema_version'] == 1 and config['sd_denominator'] == 10
            and config['alpha_rank'] is False, 'Configuration contradicts team design')
    source = Path(config['input'])
    target = Path(output or config[f'{experiment}_output'])
    require(not target.exists(), f'Output exists: {target}; use a new folder')
    agents, lineups, rates, wins, audit = load_e1(source)
    lock_hash = digest(source / 'freeze.json')
    manifest = {'experiment': experiment.upper(), 'config': config, 'e1_input': str(source.resolve()),
                'frozen_lock_sha256': lock_hash, 'e1_games_sha256': digest(source / 'games.csv'),
                'e1_source_sha256': json.loads((source / 'manifest.json').read_text())['source_sha256'],
                'analysis_script_sha256': digest(Path(__file__)), 'python': platform.python_version(),
                'dependencies': {'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
                'new_games_played': 0}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f'.{experiment}-') as tmp:
        bundle = Path(tmp) / 'bundle'
        analysis = bundle / 'analysis'
        analysis.mkdir(parents=True)
        write_json(bundle / 'manifest.json', manifest)
        write_json(bundle / 'config.json', config)
        archive_source(bundle / 'source.zip', config_path)
        dependencies = sorted(f'{d.metadata["Name"]}=={d.version}' for d in installed_distributions()
                              if d.metadata['Name'].lower() not in ('pip', 'setuptools'))
        (bundle / 'requirements.lock.txt').write_text('\n'.join(dependencies) + '\n', encoding='utf-8')
        with zipfile.ZipFile(bundle / 'source.zip', 'a') as archive:
            archive.write(config_path, 'configs/e2_e3.json')
            archive.write(bundle / 'requirements.lock.txt', 'requirements.lock.txt')
        validation = {'valid': True, 'e1_audit': audit, 'input_freeze_sha256': lock_hash, 'new_games_played': 0}
        if experiment == 'e2':
            summaries, detailed, table = robustness(agents, lineups, rates)
            for row in summaries:
                require(abs(row['mean_win_rate'] - sum(wins[m, row['agent']] for m in lineups) / 2000) < 1e-12,
                        'Mean does not match overall rate')
            write_csv(bundle / 'summary.csv', summaries)
            write_csv(analysis / 'robustness_summary.csv', summaries)
            write_csv(analysis / 'lineup_win_rates.csv', detailed)
            write_csv(analysis / 'lineup_table.csv', table)
            heatmap([[rates.get((m, a), float('nan')) for m in sorted(lineups)] for a in agents],
                    agents, sorted(lineups), analysis / 'lineup_heatmap.png',
                    'E2 robustness — 200 games per lineup; grey = did not participate')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            positions = np.arange(len(summaries))
            ax.bar(positions - .18, [100 * r['mean_win_rate'] for r in summaries], .36, label='Mean')
            ax.bar(positions + .18, [100 * r['worst_case_win_rate'] for r in summaries], .36, label='Worst lineup')
            ax.set_xticks(positions, [r['agent'] for r in summaries], rotation=20, ha='right')
            ax.set_ylabel('Win rate (%)')
            ax.set_title('E2 — ordered by worst-case performance; 10 lineups per agent')
            ax.legend()
            fig.tight_layout()
            fig.savefig(analysis / 'robustness_comparison.png', dpi=180)
            plt.close(fig)
            validation.update({'agents': 6, 'participating_cells': len(detailed), 'blank_cells': 30,
                               'sd_definition': 'Population SD across ten observed lineup rates; denominator 10',
                               'interpretation': 'Descriptive robustness across observed lineups; the minimum is noisy, not a guarantee against all opponents.'})
        else:
            matrix, contributions = payoff(agents, lineups, rates)
            solution = equilibrium(matrix, config['equilibrium_tolerance'])
            require(abs(solution['expected_row_payoff']) <= config['equilibrium_tolerance'], 'Skew game value must be zero')
            solution['agents'] = agents
            matrix_rows = [{'agent': a, **{b: float(matrix[i, j]) for j, b in enumerate(agents)}} for i, a in enumerate(agents)]
            distributions = [{'agent': a, 'row_probability': solution['row_distribution'][i],
                              'column_probability': solution['column_distribution'][i]} for i, a in enumerate(agents)]
            write_csv(bundle / 'summary.csv', distributions)
            write_csv(analysis / 'payoff_matrix.csv', matrix_rows)
            write_csv(analysis / 'pairwise_contributions.csv', contributions)
            write_csv(analysis / 'equilibrium_distribution.csv', distributions)
            write_json(analysis / 'equilibrium.json', solution)
            heatmap(matrix, agents, agents, analysis / 'payoff_heatmap.png',
                    'E3 surrogate payoffs — average over 6 shared lineups', True)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            positions = np.arange(len(agents))
            ax.bar(positions - .18, solution['row_distribution'], .36, label='Row player')
            ax.bar(positions + .18, solution['column_distribution'], .36, label='Column player')
            ax.set_xticks(positions, agents, rotation=20, ha='right')
            ax.set_ylabel('Equilibrium probability')
            ax.set_ylim(0, 1.05)
            ax.set_title('E3 — one equilibrium of the surrogate game, not Monopoly itself')
            ax.legend()
            fig.tight_layout()
            fig.savefig(analysis / 'equilibrium_distribution.png', dpi=180)
            plt.close(fig)
            validation.update({'matrix_shape': [6, 6], 'pairwise_contribution_rows': len(contributions),
                               'max_antisymmetry_error': float(np.max(abs(matrix + matrix.T))),
                               'diagonal_zero': bool(np.all(np.diag(matrix) == 0)),
                               'equilibrium_best_response_gap': solution['best_response_gap'],
                               'alpha_rank_implemented': False})
        verify(source)
        require(digest(source / 'freeze.json') == lock_hash, 'Freeze lock changed during analysis')
        write_json(analysis / 'validation.json', validation)
        write_json(bundle / 'progress.json', {'status': 'complete', 'new_games_played': 0})
        bundle.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('e2', 'e3', 'all'), default='all')
    parser.add_argument('--config', type=Path, default=Path('configs/e2_e3.json'))
    parser.add_argument('--output', type=Path, help='New folder for one experiment')
    args = parser.parse_args()
    require(args.output is None or args.experiment != 'all', '--output requires e2 or e3')
    for exp in ('e2', 'e3') if args.experiment == 'all' else (args.experiment,):
        print(generate(exp, args.config, args.output))


if __name__ == '__main__':
    main()
