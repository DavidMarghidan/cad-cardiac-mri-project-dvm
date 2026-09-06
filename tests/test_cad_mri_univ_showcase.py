"""Small deterministic tests for the custom mathematics in the showcase pipeline.

These tests do not require the CAD dataset, MONAI, or a GPU. They verify the
mathematical contracts that should remain true even before the expensive Kaggle
run is launched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("cad-cardiac-mri-project-dvm.py")
SPEC = importlib.util.spec_from_file_location("cad_mri_showcase", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class ShowcaseMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PIPELINE.Config(
            dataset_root=Path("/tmp/not-needed"),
            output_root=Path("/tmp/not-needed"),
            gradient_steps=1500,
            annealing_steps=400,
            bootstrap_replicates=20,
        )

    def test_affine_shuffle_preserves_every_visible_value(self) -> None:
        image = np.zeros((12, 12), dtype=np.float32)
        support = np.zeros_like(image)
        support[2:10, 3:9] = 1.0
        image[support > 0.5] = np.linspace(0.0, 1.0, int(support.sum()))
        digest = hashlib.sha256(b"unit-test-image").hexdigest()

        first = PIPELINE.affine_shuffle_visible_values(image, support, digest)
        second = PIPELINE.affine_shuffle_visible_values(image, support, digest)

        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(
            np.array_equal(
                np.sort(first[support > 0.5]),
                np.sort(image[support > 0.5]),
            )
        )
        self.assertFalse(np.array_equal(first, image))
        self.assertTrue(np.all(first[support <= 0.5] == 0.0))

    def test_hierarchical_pool_weights_series_equally(self) -> None:
        # Patient A has ten zero-valued slices in one series and one value-10
        # slice in another. Equal series weighting should return 5, not 10/11.
        records = [
            PIPELINE.SliceRecord(Path(f"/tmp/a{i}.jpg"), 0, "Directory_1", "Directory_1/S1")
            for i in range(10)
        ]
        records.append(
            PIPELINE.SliceRecord(Path("/tmp/b.jpg"), 0, "Directory_1", "Directory_1/S2")
        )
        records.extend(
            [
                PIPELINE.SliceRecord(Path(f"/tmp/c{i}.jpg"), 1, "Directory_2", "Directory_2/S1")
                for i in range(2)
            ]
        )
        features = np.concatenate(
            [
                np.zeros((10, 1), dtype=np.float32),
                np.asarray([[10.0]], dtype=np.float32),
                np.asarray([[2.0], [2.0]], dtype=np.float32),
            ],
            axis=0,
        )
        pooled, labels, patients = PIPELINE.hierarchical_patient_pool(features, records)
        self.assertEqual(patients.tolist(), ["Directory_1", "Directory_2"])
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertAlmostEqual(float(pooled[0, 0]), 5.0, places=6)
        self.assertAlmostEqual(float(pooled[1, 0]), 2.0, places=6)

    def test_nuisance_projection_removes_linear_predictability(self) -> None:
        rng = np.random.default_rng(5)
        n = 120
        nuisance = rng.normal(size=(n, 2))
        y = (rng.random(n) > 0.5).astype(np.int64)
        disease = (2 * y - 1)[:, None] + rng.normal(0, 0.5, size=(n, 1))
        heart = np.concatenate(
            [
                disease + 2.0 * nuisance[:, :1],
                nuisance,
                rng.normal(size=(n, 4)),
            ],
            axis=1,
        )

        model = PIPELINE.NuisanceProjectedModel(
            alpha=0.1,
            c_value=0.1,
            cfg=self.cfg,
            seed=7,
        ).fit(heart, nuisance, y)
        original_scaled = model.heart_scaler.transform(heart)
        cleaned = model.transform(heart, nuisance)

        before = PIPELINE.nuisance_correlation(original_scaled[:, 0], nuisance)
        after = PIPELINE.nuisance_correlation(cleaned[:, 0], nuisance)
        self.assertLess(after, before * 0.25)

    def test_covariance_penalty_reduces_score_nuisance_correlation(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        nuisance = rng.normal(size=(n, 1))
        y = (rng.random(n) > 0.5).astype(np.int64)
        disease_signal = (2 * y - 1) + rng.normal(scale=0.7, size=n)
        x = np.column_stack(
            [
                disease_signal + 2.0 * nuisance[:, 0],
                nuisance[:, 0],
                rng.normal(size=n),
            ]
        )

        unpenalized = PIPELINE.ConfounderPenalizedLogisticRegression(
            l2=0.01,
            gamma=0.0,
            steps=2000,
            learning_rate=0.03,
            seed=1,
        ).fit(x, nuisance, y)
        penalized = PIPELINE.ConfounderPenalizedLogisticRegression(
            l2=0.01,
            gamma=10.0,
            steps=2000,
            learning_rate=0.03,
            seed=1,
        ).fit(x, nuisance, y)

        corr_unpenalized = PIPELINE.safe_abs_correlation(
            PIPELINE.safe_logit(unpenalized.predict_proba(x)), nuisance[:, 0]
        )
        corr_penalized = PIPELINE.safe_abs_correlation(
            PIPELINE.safe_logit(penalized.predict_proba(x)), nuisance[:, 0]
        )
        auc_penalized = PIPELINE.roc_auc_score(y, penalized.predict_proba(x))

        self.assertLess(corr_penalized, corr_unpenalized)
        self.assertGreater(auc_penalized, 0.80)

    def test_simulated_annealing_is_deterministic_and_not_worse_in_objective(self) -> None:
        rng = np.random.default_rng(11)
        n = 80
        y = (rng.random(n) > 0.5).astype(np.int64)
        nuisance = rng.normal(size=n)
        heart = PIPELINE.safe_sigmoid(1.2 * (2 * y - 1) + 0.8 * nuisance)
        mask = PIPELINE.safe_sigmoid(1.5 * nuisance)
        outside = PIPELINE.safe_sigmoid(1.0 * nuisance + rng.normal(0, 0.2, n))

        model_1 = PIPELINE.SimulatedAnnealingDebiaser(self.cfg, seed=13).fit(
            y, heart, mask, outside
        )
        model_2 = PIPELINE.SimulatedAnnealingDebiaser(self.cfg, seed=13).fit(
            y, heart, mask, outside
        )

        heart_z, mask_z, outside_z = [
            (PIPELINE.safe_logit(values) - mean) / std
            for values, mean, std in zip(
                (heart, mask, outside), model_1.means, model_1.stds
            )
        ]
        initial = model_1._objective(y, heart_z, mask_z, outside_z, 0.0, 0.0)

        self.assertAlmostEqual(model_1.alpha_, model_2.alpha_, places=12)
        self.assertAlmostEqual(model_1.beta_, model_2.beta_, places=12)
        self.assertGreaterEqual(model_1.objective_, initial - 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
