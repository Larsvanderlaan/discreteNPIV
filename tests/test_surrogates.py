import unittest

import numpy as np

from discreteNPIV import (
    encode_experiment_arms,
    estimate_average_functional,
    estimate_long_term_effect_from_surrogates,
    estimate_long_term_mean_from_surrogates,
)


def _make_surrogate_data(
    *,
    n_experiments: int = 3,
    n_per_arm: int = 12,
    n_features: int = 3,
    n_target_samples: int = 200,
    random_state: int = 17,
) -> dict[str, object]:
    rng = np.random.default_rng(random_state)

    experiment_ids = [f"exp_{idx + 1}" for idx in range(n_experiments)]
    theta = rng.normal(size=n_features)
    theta /= np.linalg.norm(theta)

    arm_shift: dict[str, np.ndarray] = {}
    for experiment_id in experiment_ids:
        baseline = rng.normal(scale=0.30, size=n_features)
        treatment_lift = rng.normal(loc=0.12, scale=0.10, size=n_features)
        arm_shift[f"{experiment_id}:control"] = baseline
        arm_shift[f"{experiment_id}:treatment"] = baseline + treatment_lift

    historical_experiment_ids: list[str] = []
    historical_arm_labels: list[str] = []
    X_hist_rows: list[np.ndarray] = []
    Y_hist_rows: list[float] = []
    for experiment_id in experiment_ids:
        for arm_label in ("control", "treatment"):
            key = f"{experiment_id}:{arm_label}"
            for _ in range(n_per_arm):
                confounder = rng.normal(scale=0.35)
                surrogate = arm_shift[key] + confounder + rng.normal(scale=0.20, size=n_features)
                outcome = float(surrogate @ theta + 0.40 * confounder + rng.normal(scale=0.20))
                historical_experiment_ids.append(experiment_id)
                historical_arm_labels.append(arm_label)
                X_hist_rows.append(surrogate)
                Y_hist_rows.append(outcome)

    control_centers = np.vstack([arm_shift[f"{experiment_id}:control"] for experiment_id in experiment_ids])
    treatment_centers = np.vstack([arm_shift[f"{experiment_id}:treatment"] for experiment_id in experiment_ids])
    control_mean = np.mean(control_centers, axis=0)
    treated_mean = control_mean + 0.70 * np.mean(treatment_centers - control_centers, axis=0)

    return {
        "historical_experiment_ids": np.asarray(historical_experiment_ids, dtype=object),
        "historical_arm_labels": np.asarray(historical_arm_labels, dtype=object),
        "X_hist": np.asarray(X_hist_rows, dtype=float),
        "Y_hist": np.asarray(Y_hist_rows, dtype=float),
        "X_new_control": rng.normal(loc=control_mean, scale=0.20, size=(n_target_samples, n_features)),
        "X_new_treated": rng.normal(loc=treated_mean, scale=0.20, size=(n_target_samples, n_features)),
    }


def _run_single_linear_effect(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_experiments = 6
    n_per_arm = 200
    n_features = 3
    n_target_samples = 5_000
    theta = np.array([0.8, -0.4, 0.3])

    base_shifts = rng.normal(scale=0.40, size=(n_experiments, n_features))
    treatment_shifts = rng.normal(loc=0.25, scale=0.10, size=(n_experiments, n_features))

    historical_experiment_ids: list[str] = []
    historical_arm_labels: list[str] = []
    X_hist_rows: list[np.ndarray] = []
    Y_hist_rows: list[float] = []
    for experiment_idx in range(n_experiments):
        for arm_label, delta in (("control", np.zeros(n_features)), ("treatment", treatment_shifts[experiment_idx])):
            for _ in range(n_per_arm):
                confounder = rng.normal(scale=0.60)
                surrogate = base_shifts[experiment_idx] + delta + confounder + rng.normal(scale=0.15, size=n_features)
                outcome = float(surrogate @ theta + 0.50 * confounder + rng.normal(scale=0.15))
                historical_experiment_ids.append(f"exp_{experiment_idx}")
                historical_arm_labels.append(arm_label)
                X_hist_rows.append(surrogate)
                Y_hist_rows.append(outcome)

    X_hist = np.asarray(X_hist_rows)
    Y_hist = np.asarray(Y_hist_rows)
    X_new_control = rng.normal(loc=np.mean(base_shifts, axis=0), scale=0.20, size=(n_target_samples, n_features))
    X_new_treated = rng.normal(
        loc=np.mean(base_shifts + 0.60 * treatment_shifts, axis=0),
        scale=0.20,
        size=(n_target_samples, n_features),
    )
    truth = float(np.mean(X_new_treated @ theta) - np.mean(X_new_control @ theta))

    result = estimate_long_term_effect_from_surrogates(
        X_hist=X_hist,
        Y_hist=Y_hist,
        historical_arms=np.asarray(historical_arm_labels, dtype=object),
        historical_experiment_ids=np.asarray(historical_experiment_ids, dtype=object),
        X_new_treated=X_new_treated,
        X_new_control=X_new_control,
        n_splits=2,
        lambda_grid=np.array([1e-6, 1e-4, 1e-2]),
        gamma_grid=np.array([0.0]),
        selection="2sls",
        random_state=seed,
    )
    return truth, float(result.selected.estimate)


def _run_overlap_linear_effect(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_features = 3
    n_per_cell = 250
    n_target_samples = 5_000
    theta = np.array([0.8, -0.4, 0.3])

    shift_a = {
        "control": np.array([0.0, 0.0, 0.0]),
        "treatment": np.array([0.35, 0.10, 0.05]),
    }
    shift_b = {
        "control": np.array([0.0, 0.0, 0.0]),
        "treatment": np.array([0.15, 0.25, -0.05]),
    }

    X_hist_rows: list[np.ndarray] = []
    Y_hist_rows: list[float] = []
    historical_overlaps: list[list[tuple[str, str]]] = []
    for arm_a in ("control", "treatment"):
        for arm_b in ("control", "treatment"):
            overlap_level = [("exp_a", arm_a), ("exp_b", arm_b)]
            mean_shift = shift_a[arm_a] + shift_b[arm_b]
            for _ in range(n_per_cell):
                confounder = rng.normal(scale=0.60)
                surrogate = mean_shift + confounder + rng.normal(scale=0.15, size=n_features)
                outcome = float(surrogate @ theta + 0.50 * confounder + rng.normal(scale=0.15))
                X_hist_rows.append(surrogate)
                Y_hist_rows.append(outcome)
                historical_overlaps.append(overlap_level)

    X_hist = np.asarray(X_hist_rows)
    Y_hist = np.asarray(Y_hist_rows)
    X_new_control = rng.normal(
        loc=shift_a["control"] + 0.70 * shift_b["control"],
        scale=0.20,
        size=(n_target_samples, n_features),
    )
    X_new_treated = rng.normal(
        loc=shift_a["treatment"] + 0.70 * shift_b["treatment"],
        scale=0.20,
        size=(n_target_samples, n_features),
    )
    truth = float(np.mean(X_new_treated @ theta) - np.mean(X_new_control @ theta))

    result = estimate_long_term_effect_from_surrogates(
        X_hist=X_hist,
        Y_hist=Y_hist,
        historical_arms=historical_overlaps,
        X_new_treated=X_new_treated,
        X_new_control=X_new_control,
        encoding_mode="overlap",
        allow_empty=False,
        n_splits=2,
        lambda_grid=np.array([1e-6, 1e-4, 1e-2]),
        gamma_grid=np.array([0.0]),
        selection="2sls",
        random_state=seed,
    )
    return truth, float(result.selected.estimate)


class SurrogateEncodingTests(unittest.TestCase):
    def test_single_mode_with_experiment_ids_creates_global_keys(self) -> None:
        encoding = encode_experiment_arms(
            assignments=["control", "treatment", "control", "treatment"],
            experiment_ids=["exp_a", "exp_a", "exp_b", "exp_b"],
            mode="single",
            low_support_threshold=2,
        )
        self.assertEqual(encoding.n_levels, 4)
        self.assertEqual(encoding.min_count, 1)
        self.assertIn("exp_a:control", encoding.levels)
        self.assertIn("exp_b:treatment", encoding.levels)

    def test_overlap_mode_is_order_invariant_and_allows_empty_set(self) -> None:
        encoding = encode_experiment_arms(
            [
                ["exp_a:treatment", "exp_b:baseline"],
                ["exp_b:baseline", "exp_a:treatment"],
                [],
            ],
            mode="overlap",
            allow_empty=True,
            low_support_threshold=2,
        )
        self.assertEqual(int(encoding.codes[0]), int(encoding.codes[1]))
        self.assertIn((), encoding.levels)
        self.assertIn((), encoding.low_support_levels)

    def test_overlap_mode_rejects_duplicate_experiment_in_same_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "two arms from the same experiment"):
            encode_experiment_arms(
                [[("exp_a", "control"), ("exp_a", "treatment")]],
                mode="overlap",
            )


class SurrogateEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _make_surrogate_data()
        self.lambda_grid = np.array([1e-1, 1e-3])
        self.gamma_grid = np.array([1e-2, 0.0])

    def test_long_term_mean_wrapper_matches_core_api(self) -> None:
        encoding = encode_experiment_arms(
            assignments=self.data["historical_arm_labels"],
            experiment_ids=self.data["historical_experiment_ids"],
            mode="single",
            low_support_threshold=2,
        )
        wrapper = estimate_long_term_mean_from_surrogates(
            X_hist=self.data["X_hist"],
            Y_hist=self.data["Y_hist"],
            historical_arms=self.data["historical_arm_labels"],
            historical_experiment_ids=self.data["historical_experiment_ids"],
            X_new=self.data["X_new_treated"],
            n_splits=2,
            lambda_grid=self.lambda_grid,
            gamma_grid=self.gamma_grid,
            random_state=5,
        )
        core = estimate_average_functional(
            self.data["X_hist"],
            encoding.codes,
            self.data["Y_hist"],
            self.data["X_new_treated"],
            n_splits=2,
            lambda_grid=self.lambda_grid,
            gamma_grid=self.gamma_grid,
            random_state=5,
        )
        self.assertAlmostEqual(wrapper.selected.estimate, core.selected.estimate)
        self.assertAlmostEqual(wrapper.selected.se, core.selected.se)
        self.assertEqual(wrapper.encoding.levels, encoding.levels)

    def test_long_term_effect_is_difference_of_arm_level_means(self) -> None:
        effect = estimate_long_term_effect_from_surrogates(
            X_hist=self.data["X_hist"],
            Y_hist=self.data["Y_hist"],
            historical_arms=self.data["historical_arm_labels"],
            historical_experiment_ids=self.data["historical_experiment_ids"],
            X_new_treated=self.data["X_new_treated"],
            X_new_control=self.data["X_new_control"],
            n_splits=2,
            lambda_grid=self.lambda_grid,
            gamma_grid=self.gamma_grid,
            random_state=5,
        )
        self.assertAlmostEqual(
            effect.selected.estimate,
            effect.treated_mean.selected.estimate - effect.control_mean.selected.estimate,
        )
        np.testing.assert_allclose(
            effect.treated_mean.structural_fit.coef_selected,
            effect.control_mean.structural_fit.coef_selected,
        )
        self.assertEqual(effect.structural_fit.selected_method, effect.treated_mean.structural_fit.selected_method)
        self.assertEqual(effect.treated_dual_fit.coef_selected.shape, effect.control_dual_fit.coef_selected.shape)

    def test_sparse_overlap_support_raises_helpful_error(self) -> None:
        X_hist = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.3, 0.4],
                [0.4, 0.3],
            ]
        )
        Y_hist = np.array([0.0, 0.1, 0.2, 0.3])
        historical_arms = [
            ["exp_a:treatment", "exp_b:baseline"],
            ["exp_a:treatment"],
            ["exp_b:baseline"],
            [],
        ]
        X_new = np.array([[0.2, 0.2], [0.25, 0.15]])
        with self.assertRaisesRegex(ValueError, "stable non-overlapping slice"):
            estimate_long_term_mean_from_surrogates(
                X_hist=X_hist,
                Y_hist=Y_hist,
                historical_arms=historical_arms,
                X_new=X_new,
                encoding_mode="overlap",
                allow_empty=True,
                n_splits=2,
                lambda_grid=np.array([1e-2]),
                gamma_grid=np.array([0.0]),
                random_state=0,
            )

    def test_recovers_linear_effect_under_valid_single_arm_design(self) -> None:
        errors = []
        for seed in range(3):
            truth, estimate = _run_single_linear_effect(seed)
            errors.append(estimate - truth)
        error_arr = np.asarray(errors)
        self.assertLess(abs(float(np.mean(error_arr))), 0.02)
        self.assertLess(float(np.sqrt(np.mean(error_arr**2))), 0.02)

    def test_recovers_linear_effect_under_valid_overlap_design(self) -> None:
        errors = []
        for seed in range(3):
            truth, estimate = _run_overlap_linear_effect(seed)
            errors.append(estimate - truth)
        error_arr = np.asarray(errors)
        self.assertLess(abs(float(np.mean(error_arr))), 0.02)
        self.assertLess(float(np.sqrt(np.mean(error_arr**2))), 0.02)


if __name__ == "__main__":
    unittest.main()
