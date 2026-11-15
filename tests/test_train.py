import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pandas as pd

from train import FEATURES, TARGET, ensure_dataset, load_dataset, train_model


class WineQualityTrainingTests(unittest.TestCase):
    def write_dataset(self, path: Path, row_count: int = 30) -> None:
        rows = []
        for index in range(row_count):
            row = {
                name: index * 0.1 + feature_index
                for feature_index, name in enumerate(FEATURES)
            }
            row[TARGET] = 4 + index % 4
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, sep=";", index=False)

    def test_training_saves_metrics_model_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "wine.csv"
            output = root / "artifacts"
            self.write_dataset(dataset)

            results = train_model(dataset, output)

            self.assertEqual(results["rows"], 30)
            self.assertEqual(set(results["models"]), {"mean_baseline", "random_forest"})
            for name in ("model.joblib", "metrics.json", "test_predictions.csv"):
                self.assertTrue((output / name).is_file())
            self.assertEqual(len(pd.read_csv(output / "test_predictions.csv")), 6)

    def test_missing_required_column_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory) / "incomplete.csv"
            pd.DataFrame({"quality": [5, 6]}).to_csv(dataset, sep=";", index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_dataset(dataset)

    def test_download_failure_is_actionable_and_leaves_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "data" / "wine.csv"
            with patch("train.urllib.request.urlopen", side_effect=URLError("offline")):
                with self.assertRaisesRegex(RuntimeError, "Check your internet connection"):
                    ensure_dataset(cache_path)

            self.assertFalse(cache_path.exists())
            self.assertEqual(list(cache_path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
