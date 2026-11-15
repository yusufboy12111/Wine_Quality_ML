"""Download the UCI red wine dataset and train a simple quality regressor."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/186/wine+quality"
ARCHIVE_MEMBER = "winequality-red.csv"
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET = "quality"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "raw" / ARCHIVE_MEMBER
DEFAULT_OUTPUT = PROJECT_DIR / "outputs"


def ensure_dataset(cache_path: Path, force_download: bool = False) -> Path:
    """Download and atomically cache the red wine CSV when needed."""
    if cache_path.is_file() and not force_download:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(
            DATASET_URL, headers={"User-Agent": "wine-quality-ml/1.0"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            archive_bytes = response.read()

        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            try:
                source = archive.open(ARCHIVE_MEMBER)
            except KeyError as error:
                raise RuntimeError(
                    f"Dataset archive did not contain {ARCHIVE_MEMBER!r}."
                ) from error
            with source, tempfile.NamedTemporaryFile(
                mode="wb", dir=cache_path.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(source, temporary_file)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("The downloaded dataset file was empty.")
        temporary_path.replace(cache_path)
        temporary_path = None
        return cache_path
    except (OSError, urllib.error.URLError, TimeoutError, zipfile.BadZipFile) as error:
        raise RuntimeError(
            "Could not download the UCI dataset. Check your internet connection "
            f"and retry. Source: {DATASET_URL} ({error})"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the semicolon-separated CSV and validate its model columns."""
    try:
        frame = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError as error:
        raise ValueError(f"Dataset file not found: {csv_path}") from error
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read dataset file {csv_path}: {error}") from error

    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    try:
        numeric = frame[required].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Dataset features and quality target must be numeric.") from error

    missing_targets = int(numeric[TARGET].isna().sum())
    if missing_targets:
        raise ValueError(
            f"Dataset contains {missing_targets} rows without a quality target."
        )
    if numeric[TARGET].nunique() < 2:
        raise ValueError("Dataset needs at least two distinct quality values.")

    return numeric[FEATURES], numeric[TARGET]


def make_model(seed: int) -> Pipeline:
    """Build the preprocessing and random-forest regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def regression_metrics(actual: pd.Series, predicted: Any) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train_model(
    csv_path: Path, output_dir: Path, test_size: float = 0.2, seed: int = 42
) -> dict[str, Any]:
    """Train baseline and random forest models and save evaluation artifacts."""
    features, target = load_dataset(csv_path)
    if len(features) < 5:
        raise ValueError("Dataset needs at least 5 rows for a train/test split.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed
    )
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)

    model = make_model(seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results = {
        "dataset": DATASET_PAGE,
        "target": TARGET,
        "rows": int(len(features)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": seed,
        "models": {
            "mean_baseline": regression_metrics(y_test, baseline_predictions),
            "random_forest": regression_metrics(y_test, predictions),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {
            "actual_quality": y_test.to_numpy(),
            "baseline_prediction": baseline_predictions,
            "random_forest_prediction": predictions,
        },
        index=y_test.index,
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a red wine quality regression model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to a local semicolon-separated dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the model, metrics, and test predictions.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction held out for testing."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download a fresh copy instead of using the cached dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ensure_dataset(args.data, force_download=args.force_download)
        results = train_model(args.data, args.output_dir, args.test_size, args.seed)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Dataset rows: {results['rows']}")
    print(f"Train/test rows: {results['train_rows']}/{results['test_rows']}")
    for model_name, metrics in results["models"].items():
        print(
            f"{model_name}: MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}"
        )
    print(f"Saved model and evaluation files to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
