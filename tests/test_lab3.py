import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab3

class Lab3Tests(unittest.TestCase):
    def test_repair_obeys_all_hard_constraints(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            x = lab3.decode(rng.random(lab3.N) < 0.7)
            self.assertTrue(np.all(lab3.MANDATORY <= x))
            self.assertLessEqual(lab3.raw_metrics(x)["hours"], lab3.BUDGET_HOURS)
            self.assertTrue(all(not x[i] or x[j] for i, j in lab3.REQUIRES))
            selected = {f.id for f, keep in zip(lab3.FEATURES, x) if keep}
            for f in lab3.FEATURES:
                if f.id in selected:
                    self.assertTrue(selected.isdisjoint(f.conflicts))

    def test_repair_one_sided_conflicts(self):
        for a, b in (("F12", "F13"), ("F31", "F32")):
            with self.subTest(pair=(a, b)):
                x = np.zeros(lab3.N, dtype=bool)
                i, j = lab3.INDEX[a], lab3.INDEX[b]
                x[[i, j]] = True
                fixed = lab3.decode(x)
                self.assertEqual(int(fixed[i]) + int(fixed[j]), 1)

    def test_dominance(self):
        self.assertTrue(lab3.dominates([0, 1, 1], [1, 1, 1]))
        self.assertFalse(lab3.dominates([1, 1, 1], [0, 1, 1]))
        self.assertFalse(lab3.dominates([0, 2, 1], [1, 1, 1]))

    def test_nondominated_sort(self):
        F = np.array([[0, 1], [1, 0], [1, 1], [2, 2]], dtype=float)
        fronts = lab3.nondominated_sort(F)
        self.assertEqual(set(fronts[0]), {0, 1})
        self.assertEqual(fronts[1], [2])
        self.assertEqual(fronts[2], [3])

    def test_hypervolume(self):
        got = lab3.hypervolume_3d(np.array([[0.5, 0.5, 0.5]]), ref=1.0)
        self.assertAlmostEqual(got, 0.125)

    def test_run_reproducible_and_budgeted(self):
        a = lab3.run_nsga(77, pop_size=20, generations=4)
        b = lab3.run_nsga(77, pop_size=20, generations=4)
        np.testing.assert_array_equal(a["X"], b["X"])
        self.assertEqual(a["evaluations"], 20 * 5)
        s = lab3.scalar_ga(10, lab3.WEIGHTS[0], pop_size=4, generations=3)
        self.assertEqual(s["evaluations"], 4 * 4)

if __name__ == "__main__":
    unittest.main()
