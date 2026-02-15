#!/usr/bin/env python3
"""
Simple hyperparameter sweep for the RL DQN trading agent.

This script:
  1. Loops over several (timesteps, learning_rate, cost_rate) combinations.
  2. For each combo:
       - runs `run_rl_training.py` with a dedicated output_dir
       - runs `eval_rl_agent.py` on the trained model
       - reads `rl_eval_summary.json` and records RL PnL / Sharpe / drawdown
  3. Writes a CSV summary so you can quickly see which combo works best.

Usage (from project root):

    python3 scripts/rl_sweep.py

You can also edit the GRIDS below to try more / fewer combinations.
"""

from __future__ import annotations

import csv
import json
import subprocess
from itertools import product
from pathlib import Path
from typing import List, Dict, Any


ROOT = Path(__file__).resolve().parents[1]

# === Sweep grids (feel free to edit) =========================================
TIMESTEPS_GRID = [400_000, 500_000, 600_000]
LEARNING_RATE_GRID = [7e-5, 1e-4, 3e-4]
COST_RATE_GRID = [0.00007, 0.0001, 0.0003]

PREDICTION_FILES = {
    "xgb": ROOT / "RL_signal" / "rl_signal_xgb.csv",
    "lstm": ROOT / "RL_signal" / "rl_signal_lstm.csv",
    "tcn": ROOT / "RL_signal" / "rl_signal_tcn.csv",
    "transf": ROOT / "RL_signal" / "rl_signal_transformer.csv",
}

DATA_DIR = Path("~/Desktop/code/group_01_project/hft_gitlab/data").expanduser()


def _run_cmd(cmd: List[str]) -> None:
    """Run a subprocess command, streaming output."""
    print(f"\n[rl_sweep] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _read_eval_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    sweep_root = ROOT / "models" / "rl_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for timesteps, lr, cost in product(TIMESTEPS_GRID, LEARNING_RATE_GRID, COST_RATE_GRID):
        run_name = f"ts{timesteps}_lr{lr:g}_cost{cost:g}"
        output_dir = sweep_root / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Train
        cmd_train = [
            "python3",
            str(ROOT / "scripts" / "run_rl_training.py"),
            "--data_dir",
            str(DATA_DIR),
            "--timesteps",
            str(timesteps),
            "--learning_rate",
            str(lr),
            "--cost_rate",
            str(cost),
            "--output_dir",
            str(output_dir),
        ]
        for name, path in PREDICTION_FILES.items():
            cmd_train += ["--prediction_file", f"{name}={path}"]
        _run_cmd(cmd_train)

        model_path = output_dir / "dqn_trading_model.zip"
        run_config = output_dir / "run_config.json"

        # 2) Evaluate
        cmd_eval = [
            "python3",
            str(ROOT / "scripts" / "eval_rl_agent.py"),
            "--data_dir",
            str(DATA_DIR),
            "--model_path",
            str(model_path),
            "--run_config",
            str(run_config),
            "--plot",
        ]
        _run_cmd(cmd_eval)

        eval_dir = output_dir / "eval"
        summary_path = eval_dir / "rl_eval_summary.json"
        if not summary_path.exists():
            print(f"[rl_sweep] WARNING: Summary file not found at {summary_path}")
            continue
        summary = _read_eval_summary(summary_path)
        rl_stats = summary["strategies"]["rl"]

        results.append(
            {
                "run_name": run_name,
                "timesteps": timesteps,
                "learning_rate": lr,
                "cost_rate": cost,
                "rl_total_pnl": summary.get("rl_total_pnl", rl_stats.get("cumulative_return", 0.0)),
                "rl_sharpe": rl_stats.get("sharpe_ratio", 0.0),
                "rl_max_drawdown": rl_stats.get("max_drawdown", 0.0),
            }
        )

    # 3) Write CSV summary
    summary_csv = sweep_root / "rl_sweep_results.csv"
    if results:
        fieldnames = [
            "run_name",
            "timesteps",
            "learning_rate",
            "cost_rate",
            "rl_total_pnl",
            "rl_sharpe",
            "rl_max_drawdown",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\n[rl_sweep] Wrote sweep summary to {summary_csv}")
    else:
        print("\n[rl_sweep] No successful runs to summarize.")


if __name__ == "__main__":
    main()


