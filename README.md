# Wine Quality: A First Regression Project

Train a model to estimate the quality score of red wine from 11 measured
properties. The project compares a simple mean-prediction baseline with a
random forest and evaluates both on the same held-out test set.

## Run it

From this directory, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

The first run downloads and caches the red-wine CSV in `data/raw/`. Later runs
reuse that copy. If the download fails, check your connection and retry; the
script prints an actionable error and does not leave a partial cache file.
Use `python train.py --force-download` to fetch a fresh copy. You can also use
your own compatible semicolon-separated CSV with `--data path/to/file.csv`.

The script prints mean absolute error (MAE), root mean squared error (RMSE),
and R-squared (R2) for each model. It saves the fitted model, metrics, and
held-out predictions under `outputs/`. Set `--seed`, `--test-size`,
`--data`, or `--output-dir` to change the run. Run the lightweight tests with:

```bash
python -m unittest discover -s tests
```

## Dataset

Source: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality).
The red wine data is downloaded from the [UCI dataset archive](https://archive.ics.uci.edu/static/public/186/wine+quality.zip).
It is a public dataset of Portuguese Vinho Verde samples. The quality score is
the numeric target; this is a small educational example, not a production
quality-control system.
