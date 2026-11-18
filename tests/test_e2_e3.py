from importlib.util import find_spec
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.e2_e3 import equilibrium, generate, load_e1, payoff, robustness


class E2E3Tests(unittest.TestCase):
    def synthetic(self):
        agents = list('abcdef')
        lineups = {f'm{i:02}': frozenset(group) for i, group in enumerate(combinations(agents, 4), 1)}
        rates = {(m, a): .25 for m, group in lineups.items() for a in group}
        return agents, lineups, rates

    def test_population_sd_and_missing_cells(self):
        agents, lineups, rates = self.synthetic()
        participating = sorted(m for m in lineups if 'a' in lineups[m])
        for i, m in enumerate(participating):
            rates[m, 'a'] = i / 10
        summary, detail, table = robustness(agents, lineups, rates)
        first = next(r for r in summary if r['agent'] == 'a')
        self.assertAlmostEqual(first['mean_win_rate'], .45)
        self.assertEqual(first['worst_case_win_rate'], 0)
        expected = (sum((i / 10 - .45) ** 2 for i in range(10)) / 10) ** .5
        self.assertAlmostEqual(first['population_sd'], expected)
        self.assertEqual(len(detail), 60)
        self.assertEqual(sum(v == '' for r in table for k, v in r.items() if k != 'agent'), 30)

    @unittest.skipUnless(find_spec('numpy'), 'Install requirements-analysis.txt for matrix tests')
    def test_shared_lineup_formula_and_sign(self):
        import numpy as np
        agents, lineups, rates = self.synthetic()
        shared = sorted(m for m in lineups if {'a', 'b'} <= lineups[m])
        for i, m in enumerate(shared):
            rates[m, 'a'] = .2 + i / 100
            rates[m, 'b'] = .1
        matrix, detail = payoff(agents, lineups, rates)
        self.assertAlmostEqual(matrix[0, 1], .125)
        self.assertAlmostEqual(matrix[1, 0], -.125)
        np.testing.assert_array_equal(matrix, -matrix.T)
        self.assertEqual(len(detail), 90)
        self.assertTrue(all(sum(r['row_agent'] == a and r['column_agent'] == b for r in detail) == 6
                            for a, b in combinations(agents, 2)))

    @unittest.skipUnless(find_spec('scipy'), 'Install requirements-analysis.txt for solver tests')
    def test_equilibrium_rock_paper_scissors_and_pure_dominance(self):
        import numpy as np
        rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        result = equilibrium(rps)
        np.testing.assert_allclose(result['row_distribution'], [1 / 3] * 3)
        np.testing.assert_allclose(result['column_distribution'], [1 / 3] * 3)
        self.assertLessEqual(result['best_response_gap'], 1e-9)
        pure = equilibrium([[0, 1], [-1, 0]])
        self.assertEqual(pure['row_distribution'], [1, 0])
        self.assertEqual(pure['column_distribution'], [1, 0])

    @unittest.skipUnless(find_spec('scipy'), 'Install requirements-analysis.txt for solver tests')
    def test_degenerate_and_invalid_payoffs(self):
        self.assertLessEqual(equilibrium([[0, 0], [0, 0]])['best_response_gap'], 1e-9)
        for matrix in ([], [[0, 1]], [[float('nan')]], [[float('inf')]]):
            with self.assertRaises(ValueError):
                equilibrium(matrix)

    def test_missing_lineups_are_rejected(self):
        agents, lineups, rates = self.synthetic()
        del lineups['m01']
        with self.assertRaises(ValueError):
            robustness(agents, lineups, rates)
        if find_spec('numpy'):
            with self.assertRaises(ValueError):
                payoff(agents, lineups, rates)

    @unittest.skipUnless(Path('results/e1_frozen').exists(), 'Optional local evidence')
    @unittest.skipUnless(all(find_spec(p) for p in ('matplotlib', 'numpy', 'scipy')),
                         'Install requirements-analysis.txt for analysis artifacts')
    def test_full_outputs_no_simulations_and_no_overwrite(self):
        agents, lineups, rates, wins, report = load_e1(Path('results/e1_frozen'))
        matrix, _ = payoff(agents, lineups, rates)
        for i, j in combinations(range(6), 2):
            shared = [m for m in lineups if {agents[i], agents[j]} <= lineups[m]]
            expected = sum(wins[m, agents[i]] - wins[m, agents[j]] for m in shared) / 1200
            self.assertAlmostEqual(matrix[i, j], expected)
        with tempfile.TemporaryDirectory() as tmp, patch('monopoly.engine.Game.play', side_effect=AssertionError('No simulations')):
            for exp in ('e2', 'e3'):
                output = Path(tmp) / exp
                generate(exp, Path('configs/e2_e3.json'), output)
                self.assertTrue(json.loads((output / 'analysis/validation.json').read_text())['valid'])
                self.assertEqual(len(list((output / 'analysis').glob('*.png'))), 2)
                self.assertTrue((output / 'requirements.lock.txt').exists())
                with self.assertRaises(ValueError):
                    generate(exp, Path('configs/e2_e3.json'), output)


if __name__ == '__main__':
    unittest.main()
