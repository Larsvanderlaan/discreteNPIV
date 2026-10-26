import csv
from pathlib import Path
import unittest

import numpy as np

from experiments.paper.design import generate_longterm_surrogate_data
from experiments.paper.diagnostics import run_replication
from experiments.paper.model import LongTermSurrogateConfig
from experiments.run import CONFIG_PATHS, REPORTED_METHODS, load_configs


ROOT = Path(__file__).resolve().parents[1]


class PaperExperimentTests(unittest.TestCase):
    def _small_config(self) -> LongTermSurrogateConfig:
        return LongTermSurrogateConfig(
            cell_id="test_persistent",
            n_replications=1,
            n_instruments=10,
            n_per_instrument=8,
            n_new_per_arm=100,
            n_fourier_pairs=4,
            regime="persistent",
            seed_base=101,
            lambda_grid=(1.0, 0.1, 0.01),
            gamma_grid=(1.0, 0.1, 0.01),
            inverse_stability_multiplier_replications=19,
            projection_multiplier_replications=19,
        )

    def test_frozen_configurations_match_their_hashes(self) -> None:
        self.assertEqual(len(load_configs(CONFIG_PATHS["main"])), 12)
        self.assertEqual(len(load_configs(CONFIG_PATHS["large_n"])), 1)

    def test_data_generation_is_deterministic(self) -> None:
        config = self._small_config()
        first = generate_longterm_surrogate_data(config, replication=0)
        second = generate_longterm_surrogate_data(config, replication=0)
        changed = generate_longterm_surrogate_data(config, replication=1)

        np.testing.assert_allclose(first["X"], second["X"])
        np.testing.assert_allclose(first["Y"], second["Y"])
        self.assertFalse(np.allclose(first["Y"], changed["Y"]))

    def test_replication_reports_only_paper_estimators(self) -> None:
        rows = run_replication(
            self._small_config(),
            stage="confirmatory",
            replication=0,
        )
        self.assertTrue(all(row["row_type"] == "estimate" for row in rows))
        self.assertEqual({row["method"] for row in rows}, REPORTED_METHODS["main"])
        self.assertTrue(all(np.isfinite(float(row["estimate"])) for row in rows))
        self.assertTrue(all(np.isfinite(float(row["se"])) for row in rows))
        for row in rows:
            if row["method"].startswith("loo_"):
                self.assertEqual(row["reported_inference_method"], "loo_disjoint_pairs")
                self.assertAlmostEqual(
                    row["se"] ** 2,
                    row["if_historical_variance"] + row["target_sample_variance"],
                )

    def test_summaries_are_complete_and_minimal(self) -> None:
        expected = {"main": (36, 500), "large_n": (2, 200)}
        for study, (expected_rows, expected_replications) in expected.items():
            path = ROOT / "experiments" / "results" / study / "summary.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                columns = set(reader.fieldnames or ())

            self.assertEqual(len(rows), expected_rows)
            self.assertEqual({row["method"] for row in rows}, REPORTED_METHODS[study])
            self.assertTrue(all(
                int(row["n_replications"]) + int(row["failed_replications"])
                == expected_replications for row in rows
            ))
            self.assertNotIn("run_command", columns)
            self.assertNotIn("run_started_at", columns)


if __name__ == "__main__":
    unittest.main()
