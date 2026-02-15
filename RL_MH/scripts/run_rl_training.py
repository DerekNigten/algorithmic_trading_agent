#!/usr/bin/env python3
"""
Train a DQN-based RL trading agent using signals from existing supervised models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN

from src.rl_data import RLDataConfig, build_rl_dataframe, split_train_test
from src.rl_trading_env import make_trading_env


def _parse_prediction_files(values: list[str] | None) -> Dict[str, Path]:
    """Parse CLI key=value pairs into {model_name: Path}."""
    if not values:
        return {}
    result: Dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid prediction file spec '{item}', expected name=path")
        name, path_str = item.split("=", 1)
        result[name.strip()] = Path(path_str).expanduser().resolve()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL trading agent (DQN).")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/Desktop/code/group_01_project/hft_gitlab/data",
        help="Directory containing train/val/test CSVs from prepare_data.py",
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="Optional path to selected_features.pkl (defaults to data_dir/selected_features.pkl)",
    )
    parser.add_argument(
        "--price_col",
        type=str,
        default="mid_price",
        help="Name of the price column to use for reward computation",
    )
    parser.add_argument(
        "--prediction_file",
        action="append",
        help="Optional model prediction file spec in the form name=path",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Number of environment steps for DQN training",
    )
    parser.add_argument(
        "--cost_rate",
        type=float,
        default=0.0005,
        help="Transaction cost rate applied per trade",
    )
    parser.add_argument(
        "--max_position",
        type=int,
        default=1,
        help="Maximum absolute position size allowed in the environment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/rl",
        help="Directory to store trained model and run metadata",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for DQN optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for DQN updates",
    )
    parser.add_argument(
        "--drop_constant_predictions",
        action="store_true",
        help=(
            "If set, drop prediction columns that are (near) constant in the training split. "
            "Useful when some signal files only contain test predictions (train/val filled with 0), "
            "which can create train/test distribution shift."
        ),
    )
    parser.add_argument(
        "--prediction_min_std",
        type=float,
        default=1e-8,
        help="Minimum std threshold for keeping a prediction column when --drop_constant_predictions is set.",
    )
    return parser.parse_args()


def _filter_prediction_cols_by_variance(
    df: "pd.DataFrame",
    prediction_cols: list[str],
    min_std: float,
) -> tuple[list[str], list[str]]:
    """Drop prediction columns with near-zero variance to avoid train/test distribution shift."""
    if not prediction_cols:
        return [], []
    keep: list[str] = []
    dropped: list[str] = []
    for col in prediction_cols:
        series = df[col]
        if series.isna().all():
            dropped.append(col)
            continue
        # Use population std to be stable for large n; treat NaNs as missing.
        std = float(series.astype(float).std(ddof=0, skipna=True))
        if std < min_std:
            dropped.append(col)
        else:
            keep.append(col)
    return keep, dropped


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = _parse_prediction_files(args.prediction_file)

    config = RLDataConfig(
        data_dir=data_dir,
        feature_file=Path(args.feature_file).expanduser().resolve()
        if args.feature_file
        else None,
        price_col=args.price_col,
        prediction_files=prediction_files,
    )

    print("[run_rl_training] Building RL dataframe...")
    df_all = build_rl_dataframe(config)
    df_train, _ = split_train_test(df_all)

    feature_cols = config.resolve_feature_list()
    feature_cols = [col for col in feature_cols if col != config.price_col]
    prediction_cols = [col for col in df_train.columns if col.startswith("pred_")]
    dropped_prediction_cols: list[str] = []
    if args.drop_constant_predictions and prediction_cols:
        prediction_cols, dropped_prediction_cols = _filter_prediction_cols_by_variance(
            df_train, prediction_cols, min_std=float(args.prediction_min_std)
        )
        if dropped_prediction_cols:
            print(
                "[run_rl_training] Dropped constant prediction cols:",
                ", ".join(dropped_prediction_cols),
            )

    env = make_trading_env(
        df=df_train,
        price_col=config.price_col,
        feature_cols=feature_cols,
        prediction_cols=prediction_cols,
        cost_rate=args.cost_rate,
        max_position=args.max_position,
    )

    print(
        "[run_rl_training] Observation size:",
        env.observation_space.shape,
        "| Actions:",
        env.action_space.n,
    )
    print("[run_rl_training] Starting DQN training...")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=200_000,
        learning_starts=5_000,
        target_update_interval=10_000,
        train_freq=4,
        gradient_steps=1,
        gamma=0.99,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_path = output_dir / "dqn_trading_model"
    model.save(model_path)
    print(f"[run_rl_training] Model saved to {model_path}")

    run_metadata = {
        "data_dir": str(data_dir),
        "price_col": config.price_col,
        "feature_cols": feature_cols,
        "prediction_cols": prediction_cols,
        "dropped_prediction_cols": dropped_prediction_cols,
        "timesteps": args.timesteps,
        "cost_rate": args.cost_rate,
        "max_position": args.max_position,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "drop_constant_predictions": bool(args.drop_constant_predictions),
        "prediction_min_std": float(args.prediction_min_std),
        "prediction_files": {k: str(v) for k, v in prediction_files.items()},
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"[run_rl_training] Run metadata saved to {output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()

