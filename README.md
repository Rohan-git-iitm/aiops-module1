# AIOps Module 1 Assignment

**Rohan Sai P. — DA24B049**

| Item | Location |
| --- | --- |
| Report | `assgn1_AIOps_DA24B049.pdf` |
| Video | `assgn1_aiops_video.mp4` |
| Screenshots | `screenshots/` |
| Q4 repository | https://github.com/Rohan-git-iitm/module_1_q4 |

Q1 is written up in the report. Q2 and Q3 are in this repository. Q4 is in a
separate repository, linked above.

---

## Environment

Python 3.11 is required. `requirements.txt` pins package versions but not the
interpreter, so if your system Python differs, use `uv` to fetch 3.11:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you already have Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Q2 — MLflow experiment comparison

**1. Start the tracking server** in a separate terminal and leave it running:

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The UI is at http://localhost:5000.

**2. Run the sweep** (three learning rates × two batch sizes = six runs):

```bash
cd q2
for lr in 0.0001 0.001 0.01; do
  for bs in 32 128; do
    python src/train.py --lr $lr --batch_size $bs --epochs 20
  done
done
```

The first run downloads MNIST from OpenML (~55 MB) and caches it to
`~/scikit_learn_data/`. The full sweep takes roughly ten minutes.

**3. Inspect the results:**

```bash
python src/summarize.py   # writes summary.csv — one row per run, sorted by accuracy
python src/curves.py      # writes curves.csv — per-epoch metrics for all six runs
```

**4. View the comparison table** in the MLflow UI: open the `mnist-mlp`
experiment, select all six runs, and click **Compare**.

### Files

- `q2/src/train.py` — training script. Loads MNIST, subsamples 10,000 training
  and 2,000 validation images, trains an `MLPClassifier`, and logs parameters and
  per-epoch metrics to MLflow.
- `q2/src/summarize.py` — queries the tracking server for all runs and writes a
  sorted summary.
- `q2/src/curves.py` — pulls per-epoch metric histories for the overfitting
  analysis.

---

## Q3 — DVC data versioning and rollback

### Setup

```bash
dvc init
dvc remote add --default macremote ssh://<host>/Users/<user>/dvcstore
dvc remote modify macremote user <user>
git add .dvc/config && git commit -m "Configure SSH remote storage"
```

### Dataset v1

```bash
cd q3
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip
unzip -q data.zip && rm -f data.zip
python make_csv.py                      # 1800 rows + header

dvc add filenames.csv
git add filenames.csv.dvc .gitignore
git commit -m "Add dataset v1: 1800 rows"
git tag -a v1 -m "dataset v1, 1800 rows + header"
dvc push
```

### Dataset v2

```bash
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/new-labels.zip
unzip -q -o new-labels.zip && rm -f new-labels.zip
python make_csv.py                      # 2800 rows + header

dvc add filenames.csv
git commit -am "Update dataset to v2: 2801 lines (2800 rows + header) after adding new-labels"
git tag -a v2 -m "dataset v2, 2800 rows + header"
dvc push
```

### Rollback

```bash
wc -l filenames.csv         # 2801

git checkout v1
wc -l filenames.csv         # still 2801
dvc status

dvc checkout
wc -l filenames.csv         # 1801
dvc status

git checkout main
dvc checkout
```

Terminal output is in `screenshots/q3_terminal_log.png`.

### Files

- `q3/make_csv.py` — writes a sorted CSV of filenames from `data/`.
- `q3/filenames.csv.dvc` — the DVC pointer file tracked by git.
- Tags `v1` and `v2` mark the two dataset versions.

---

## Q4 — End-to-end reproducibility drill

Separate repository: **https://github.com/Rohan-git-iitm/module_1_q4**

### Files

- `src/prepare_data.py` — fetches MNIST and writes a fixed 10,000-image
  subsample to `data/mnist.csv`.
- `src/train.py` — trains the model and logs parameters, per-epoch metrics, the
  seed, a `git_commit` tag, and `metrics.json` as an artifact. With `--register`
  it also registers the model and sets the `staging` alias.
- `setup_q4.sh` — runs the full sequence: prepare data, `dvc add` and `dvc push`,
  commit the code and `.dvc` pointer in the same commit, train and register, then
  print the commit hash.
- Tag `partner-a-v1` marks the handoff commit.

The reproduction matched exactly: both runs produced `final_val_accuracy =
0.9435`, a difference of 0.0000.

---

## AI disclosure

**Tools used:** Claude (Anthropic).

**How it was used:**

- Drafting and structuring this README file.
- LaTeX formatting for the report PDF, including document structure, listings
  configuration, and page layout to fit the one-page limit.
- Troubleshooting a `skops.io.exceptions.UntrustedTypesFoundException` raised by
  `mlflow.sklearn.log_model` in Q2. MLflow 3.x serialises scikit-learn models
  with skops by default, which refuses to serialise
  `sklearn.neural_network._stochastic_optimizers.AdamOptimizer` unless it is
  explicitly trusted. The fix was to pass `skops_trusted_types` to `log_model`.

**Impact:** The tool assisted with document formatting and with diagnosing one
library-specific serialisation error.
