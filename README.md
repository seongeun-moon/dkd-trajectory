# Kidney-trajectory experiments

Analysis code for a Transformer-based model that predicts kidney-function
trajectories in diabetic kidney disease from electronic health records.

This repository holds **code only** — no patient data, no model checkpoints.
Every script takes the location of the restricted data at run time and stops
with an explicit message if it has not been configured.

## Layout

```
.
├── common/         Shared across stages: endpoint definitions, metric helpers,
│                   class balancing, and the tensor loader/padder
├── preprocessing/  01-05: raw EHR export -> model-ready tensors
├── het_trans/      The Performer model and its training driver
├── baselines/      LR / SVM / LightGBM / XGBoost, through PyCaret
└── analysis/       Metrics, figures, feature importance, robustness checks
```

Each stage is run from its own directory and adds the repository root to
`sys.path` to import `common/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. For CUDA, install the matching PyTorch build from pytorch.org
first.

`baselines/` needs a **separate environment**: PyCaret pins older numpy/pandas
than the rest of the repository, and installing it alongside downgrades numpy
and breaks the analysis scripts.

```bash
python -m venv .venv-baselines && source .venv-baselines/bin/activate
pip install -r requirements-baselines.txt
```

Point the code at your copy of the restricted data:

```bash
export CKD_RAW_ROOT=/secure/.../export        # hospital CSV/XLSX export tree
export CKD_DATA_ROOT=/secure/.../12_normal    # preprocessed reduced24_*.pkl
export CKD_RESULT_ROOT=/secure/.../result     # training run directories
export CKD_CKPT_ROOT=/secure/.../ckd-stage    # archived model checkpoints
export CKD_RAW_RESULTS=/secure/.../raw_results # archived prediction CSVs
export CKD_WORK_DIR=/secure/.../work          # where results are written
```

Most analysis scripts also accept `--data-root`, `--ckpt-root`,
`--raw-results` and `--out-dir` per invocation. Everything written goes to the
work directory, which is git-ignored — derived per-slice files are still
individual-level data and must not be committed.

## The cohort

`common/` and `preprocessing/config.py` define the study population once:

| | |
|---|---|
| Observation window | 1 year, ending at the index date |
| Prediction horizons | Δ ∈ {12, 36, 60, 120} months |
| Exclusions | eGFR < 60 at any point in the window; fewer than three eGFR measurements |
| Dataset name | `12_normal`, derived in `preprocessing/config.py` from the parameters above |

Two binary endpoints, defined once in `common/endpoints.py`:

- `ckd-stage` — the CKD stage at the horizon is above the baseline band
- `egfr-decline` — horizon eGFR has fallen to ≤ 70 % of baseline

They are **not interchangeable**: every driver requires `--target_type` when
`--binary` is set.

## Running

### 1. Preprocessing

Five steps, in order, turning the raw export under `$CKD_RAW_ROOT` into the
tensors every later stage reads.

```bash
cd preprocessing
python 01_data_processing.py       # per-table export -> one frame per variable
python 02_data_aggregation.py      # cohort filters -> DM_all.csv
python 03_divide_data.py           # observation-window slices + endpoints
python 04_tokenization_unified.py  # tokenise, pad to maxlen=24, time embeddings
python 05_integrate_for_baselines.py  # homogeneous array for the tabular models
```

Cohort parameters live in `preprocessing/config.py`; the dataset directory name
is derived from them, so it cannot disagree with the data inside it.

The source export carries Korean column headers; the mapping tables in `01`/`02`
reproduce that vocabulary. They describe the schema, not any patient.

`03` reads a clinician-curated table mapping drug ingredient to kidney
relevance (`code_label_integ.csv`: `ingredient`, `importance` ∈ {0, 1, 2}). It
is specific to the source formulary and is not distributed here.

### 2. Training

```bash
cd het_trans
python main_graph_snubh.py \
    --data_path "$CKD_DATA_ROOT" --trans_pos_encode --balanced \
    --binary --target_type ckd-stage --seed 2022
```

`--target_type egfr-decline` trains the other endpoint. Seeds 2022/2023/2024
were averaged.

Defaults match the reported hyperparameters: Performer attention, 1 layer,
4 heads, embedding 64, feed-forward 2048, dropout 0.5, VIME masking rate 0.3,
AdamP with lr 1e-4 and weight decay 0.01, batch size 200, up to 10,000 epochs,
early stopping with patience 20, and auxiliary eGFR-regression and
masked-reconstruction losses weighted 1e-4 and 0.1.

`--config <run>/config.yaml` reads `exp_setting.target_type` from an archived
run; an explicit `--target_type` overrides it.

`train()` takes a `save_log` argument (not a CLI flag). Passing `save_log=False`
runs the loop headless — no TensorBoard event files, no `log.txt`, but the same
checkpoint — which is how `analysis/transformer_temporal_split.py` reuses this
training loop instead of reimplementing it.

### 3. Baselines

```bash
cd baselines
./baseline.sh                        # four models x three seeds x three balancing runs
TARGET_TYPE=ckd-stage ./baseline.sh  # the other endpoint
```

Or one run at a time:

```bash
python main_pycaret.py \
    --data_path "$CKD_DATA_ROOT" --result_path "$CKD_RESULT_ROOT" \
    --agg average --concat_time --concat_cdt --binary --target_type egfr-decline \
    --balanced --seed 2022 --balance_run 26 --model lightgbm
```

PyCaret runs the hyperparameter search and writes `result.json` into the run
directory.

### 4. Analysis

```bash
cd analysis

# from the archived prediction CSVs — nothing needs retraining
python compute_metrics.py            # metric tables, calibration and decision curves

# from the training run directories
python check_result.py               # per-horizon AUROC / recall / F1 over seeds
python extract_raw_predictions.py    # per-horizon prediction CSVs, ROC / PR curves
python exam_result.py                # sweep every run into one table

# tabular analyses share one feature matrix, built once
python build_feature_matrix.py --data-root "$CKD_DATA_ROOT"
python shap_importance.py            # TreeSHAP attribution
python perturbation_stress_test.py   # distribution-shift robustness
python synthetic_stress_test.py      # synthetic-cohort validation (negative result)

# need the trained transformer
python transformer_permutation_importance.py   # 45 variables x 3 seeds
python transformer_temporal_split.py           # 2003-2010 -> 2011-2014 retraining

python ckd_progression_sankey.py     # kidney-function trajectory diagram
```

`build_feature_matrix.py` must run before `shap_importance.py`,
`perturbation_stress_test.py` and `synthetic_stress_test.py`.

`synthetic_stress_test.py` reports a **negative result**: a Gaussian-Copula
synthesiser reproduces the marginals and the linear correlation structure but
not the feature-outcome relationship for an outcome this rare, so baseline
AUROC on the synthetic cohort collapses to ~0.55 against ~0.82 on the real test
set. It is kept because it documents why the targeted perturbation test is used
instead of a fully generative one.

## Third-party code

`het_trans/model/performer_attention.py` and `het_trans/model/reversible.py`
are redistributed unchanged from
[`lucidrains/performer-pytorch`](https://github.com/lucidrains/performer-pytorch)
under its MIT licence — see `het_trans/NOTICE.md`. Do not modify them.
