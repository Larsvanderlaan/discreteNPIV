import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from discreteNPIV import (
    estimate_average_functional,
    estimate_long_term_effect_from_surrogates,
    fit_dual_nuisance,
    fit_structural_nuisance,
    generate_synthetic_data,
)


def _example_surrogate_data() -> dict[str, object]:
    rng = np.random.default_rng(12)
    historical_experiment_ids = np.repeat(["exp_a", "exp_b", "exp_c"], 80)
    historical_arm_labels = np.tile(np.repeat(["control", "treatment"], 40), 3)

    arm_shift = {
        "exp_a:control": np.array([0.0, 0.1, -0.1]),
        "exp_a:treatment": np.array([0.4, 0.2, 0.0]),
        "exp_b:control": np.array([-0.1, 0.0, 0.1]),
        "exp_b:treatment": np.array([0.2, 0.3, 0.2]),
        "exp_c:control": np.array([0.1, -0.2, 0.0]),
        "exp_c:treatment": np.array([0.3, 0.1, 0.3]),
    }
    theta = np.array([0.8, -0.5, 0.4])

    surrogates = []
    outcomes = []
    for experiment_id, arm_label in zip(historical_experiment_ids, historical_arm_labels, strict=True):
        key = f"{experiment_id}:{arm_label}"
        confounder = rng.normal(scale=0.35)
        x = arm_shift[key] + confounder + rng.normal(scale=0.2, size=3)
        y = x @ theta + 0.5 * confounder + rng.normal(scale=0.2)
        surrogates.append(x)
        outcomes.append(y)

    return {
        "X_hist": np.asarray(surrogates),
        "Y_hist": np.asarray(outcomes),
        "historical_arm_labels": historical_arm_labels,
        "historical_experiment_ids": historical_experiment_ids,
        "X_new_control": rng.normal(loc=[0.05, 0.0, 0.0], scale=0.2, size=(200, 3)),
        "X_new_treated": rng.normal(loc=[0.25, 0.15, 0.1], scale=0.2, size=(200, 3)),
    }


class ExampleSmokeTests(unittest.TestCase):
    def test_core_quickstart_example_runs(self) -> None:
        data = generate_synthetic_data(
            n_per_instrument=20,
            n_instruments=8,
            n_features=5,
            n_target_samples=300,
            random_state=7,
        )
        structural = fit_structural_nuisance(
            data["X"],
            data["Z"],
            data["Y"],
            n_splits=2,
            random_state=7,
        )
        dual = fit_dual_nuisance(
            data["X"],
            data["Z"],
            data["X_new"],
            n_splits=2,
            random_state=7,
        )
        result = estimate_average_functional(
            data["X"],
            data["Z"],
            data["Y"],
            data["X_new"],
            n_splits=2,
            random_state=7,
        )
        self.assertIn(structural.selected_method, {"npjive", "2sls"})
        self.assertIn(dual.selected_method, {"npjive", "2sls"})
        self.assertTrue(np.isfinite(result.selected.estimate))

    def test_surrogate_quickstart_example_runs(self) -> None:
        data = _example_surrogate_data()
        effect = estimate_long_term_effect_from_surrogates(
            X_hist=data["X_hist"],
            Y_hist=data["Y_hist"],
            historical_arms=data["historical_arm_labels"],
            historical_experiment_ids=data["historical_experiment_ids"],
            X_new_treated=data["X_new_treated"],
            X_new_control=data["X_new_control"],
            n_splits=2,
            random_state=12,
        )
        self.assertTrue(np.isfinite(effect.selected.estimate))
        self.assertTrue(np.isfinite(effect.selected.se))
        self.assertGreater(effect.encoding.n_levels, 1)

    def test_surrogate_case_study_script_runs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        pythonpath = str(root / "src")
        if env.get("PYTHONPATH"):
            pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "reproduce_surrogate_case_study.py"),
                "--n-experiments",
                "3",
                "--n-per-arm",
                "12",
                "--n-features",
                "3",
                "--n-target-samples",
                "150",
                "--seed",
                "9",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Historical design", completed.stdout)
        self.assertIn("Long-term treatment effect", completed.stdout)


if __name__ == "__main__":
    unittest.main()
