## RL Trading (DQN) — How to Run

This repo trains a **DQN-based RL trading agent** on a deterministic trading environment. The RL observation is built from:

- Market features from `hft_gitlab/data/selected_features.pkl`
- Optional model prediction signals (XGBoost / LSTM / TCN / Transformer)
- Current position and cumulative PnL

The training/evaluation scripts are under `scripts/`.

## Directory Layout (Where to Put Data)

### 1) Raw order book snapshots (input to `prepare_data.py`)

Put your compressed book snapshot files here:

- `iex_data/book_snapshots/`
  - e.g. `20251020_book_updates.csv.gz`, `20251021_book_updates.csv.gz`, ...

`scripts/prepare_data.py` currently expects those filenames/dates (edit the `days = [...]` list in that script if you use different dates).

### 2) Prepared RL dataset (output of `prepare_data.py`)

`scripts/prepare_data.py` writes:

- `hft_gitlab/data/train_data.csv`
- `hft_gitlab/data/val_data.csv`
- `hft_gitlab/data/test_data.csv`
- `hft_gitlab/data/selected_features.pkl`
- `hft_gitlab/data/scaler.pkl`

These files are the **source of truth** for the RL pipeline.

### 3) Supervised model checkpoints (inputs to `gerenate_prediction.py`)

The prediction generator uses the best model checkpoints already in:

- `outputs/xgboost/xgb_best_model.json`
- `outputs/lstm/best_model.pt`
- `outputs/tcn/best_model.pt`
- `outputs/transformer/best_transformer_model.pth`

### 4) RL prediction files and RL signals

After running prediction generation + signal building, you will have:

- **Split predictions (generated from prepared CSVs)**:
  - `RL_signal/model/xgb_{train,val,test}_predictions.csv`
  - `RL_signal/model/lstm_{train,val,test}_predictions.csv`
  - `RL_signal/model/tcn_{train,val,test}_predictions.csv`
  - `RL_signal/model/transformer_{train,val,test}_predictions.csv`

- **Final RL signals (single-column vectors used by RL)**:
  - `RL_signal/rl_signal_xgb.csv`
  - `RL_signal/rl_signal_lstm.csv`
  - `RL_signal/rl_signal_tcn.csv`
  - `RL_signal/rl_signal_transformer.csv`

## Install

```bash
pip3 install -r requirements.txt
```

## End-to-End Pipeline

### Step 1 — Prepare train/val/test data

```bash
python3 scripts/prepare_data.py
```

### Step 2 — Generate model predictions from prepared CSVs

This reads `hft_gitlab/data/{train,val,test}_data.csv` and writes split prediction files under `RL_signal/model/`.

```bash
python3 scripts/gerenate_prediction.py
```

Notes:
- Progress is shown via `tqdm`.
- XGBoost may fall back to copying `outputs/xgboost/xgb_{train,val,test}_predictions.csv` if required features are not present in the prepared CSVs.

### Step 3 — Build RL signal vectors

This reads the split prediction files under `RL_signal/model/` and writes the final single-column signal CSVs under `RL_signal/`.

```bash
python3 scripts/make_rl_signal.py
```

### Step 4 — Train the RL agent (DQN)

```bash
python3 scripts/run_rl_training.py \
  --data_dir ~/Desktop/code/group_01_project/hft_gitlab/data \
  --timesteps 500000 \
  --learning_rate 1e-4 \
  --cost_rate 0.0001 \
  --batch_size 256 \
  --output_dir RL_outputs/rl \
  --prediction_file xgb=RL_signal/rl_signal_xgb.csv \
  --prediction_file lstm=RL_signal/rl_signal_lstm.csv \
  --prediction_file tcn=RL_signal/rl_signal_tcn.csv \
  --prediction_file transf=RL_signal/rl_signal_transformer.csv
```

Optional:
- Add `--drop_constant_predictions` to automatically drop near-constant `pred_*` columns in the training split.

### Step 5 — Evaluate on test split

```bash
python3 scripts/eval_rl_agent.py \
  --data_dir ~/Desktop/code/group_01_project/hft_gitlab/data \
  --model_path RL_outputs/rl/dqn_trading_model.zip \
  --run_config RL_outputs/rl/run_config.json \
  --plot
```

You can also evaluate an existing run:

```bash
python3 scripts/eval_rl_agent.py \
  --data_dir ~/Desktop/code/group_01_project/hft_gitlab/data \
  --model_path RL_outputs/best_RL/dqn_trading_model.zip \
  --run_config RL_outputs/best_RL/run_config.json \
  --plot
```

## Hyperparameter Sweep

Runs a grid of (timesteps, learning_rate, cost_rate) and stores results under `models/rl_sweep/`.

```bash
python3 scripts/rl_sweep.py
```

## Smoke Test

To verify the RL plumbing without real data:

```bash
python3 scripts/debug_rl_smoketest.py --random_steps 15 --dqn_timesteps 1000
```