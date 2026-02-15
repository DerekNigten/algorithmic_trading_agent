#!/usr/bin/env python3
"""
Evaluate a trained RL trading agent on the test split and compute PnL metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN

from src.rl_data import RLDataConfig, build_rl_dataframe, split_train_test
from src.rl_trading_env import make_trading_env


def _parse_prediction_files(values: list[str] | None) -> Dict[str, Path]:
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
    parser = argparse.ArgumentParser(description="Evaluate RL trading agent.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/Desktop/hft_gitlab/data",
        help="Directory containing train/val/test CSVs from prepare_data.py",
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="Optional path to selected_features.pkl",
    )
    parser.add_argument(
        "--price_col",
        type=str,
        default="mid_price",
        help="Price column used for reward computation",
    )
    parser.add_argument(
        "--prediction_file",
        action="append",
        help="Optional model prediction file spec in the form name=path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained DQN model (dqn_trading_model.zip)",
    )
    parser.add_argument(
        "--run_config",
        type=str,
        default=None,
        help="Optional JSON config produced during training to reuse settings",
    )
    parser.add_argument(
        "--cost_rate",
        type=float,
        default=0.0005,
        help="Transaction cost rate (ignored if run_config provided)",
    )
    parser.add_argument(
        "--max_position",
        type=int,
        default=1,
        help="Maximum absolute position (ignored if run_config provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/rl/eval",
        help="Directory to store evaluation artifacts",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, save a PNG of the cumulative PnL curve",
    )
    return parser.parse_args()


def _load_run_metadata(path: Path | None) -> Dict | None:
    if not path:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    cumulative = pnl
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak)
    return float(drawdown.min())


def _compute_pnl_from_positions(
    prices: np.ndarray,
    positions: np.ndarray,
    cost_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replay the reward logic used in TradingEnv for an arbitrary position vector.

    Args:
        prices: array of prices
        positions: desired position after observing each time step (same length as prices)
        cost_rate: transaction cost rate

    Returns:
        cumulative_pnl (len = len(prices) - 1)
        rewards (per-step returns, len = len(prices) - 1)
    """
    if len(prices) != len(positions):
        raise ValueError("prices and positions must have the same length")
    if len(prices) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    pnl = []
    rewards = []
    cumulative = 0.0
    prev_pos = 0.0
    prev_price = prices[0]

    for t in range(1, len(prices)):
        pos_t = positions[t]
        price_t = prices[t]
        reward = prev_pos * (price_t - prev_price)
        reward -= cost_rate * abs(pos_t - prev_pos) * price_t
        cumulative += reward
        rewards.append(reward)
        pnl.append(cumulative)
        prev_pos = pos_t
        prev_price = price_t

    return np.array(pnl, dtype=float), np.array(rewards, dtype=float)


def _compute_turnover_and_cost(
    prices: np.ndarray,
    positions: np.ndarray,
    cost_rate: float,
) -> Tuple[float, int, float]:
    """
    Compute turnover (sum abs delta position), number of trades, and total transaction cost.

    positions is aligned with prices (len must match). Transaction cost at step t uses price_t.
    """
    if len(prices) != len(positions):
        raise ValueError("prices and positions must have the same length")
    if len(prices) < 2:
        return 0.0, 0, 0.0
    deltas = np.diff(positions)
    turnover = float(np.sum(np.abs(deltas)))
    n_trades = int(np.sum(np.abs(deltas) > 0))
    # cost uses price_t for t=1..n-1
    total_cost = float(np.sum(cost_rate * np.abs(deltas) * prices[1:]))
    return turnover, n_trades, total_cost


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    # If run_config is provided and output_dir was left as default,
    # place evaluation artifacts next to the training run directory:
    #   <train_output_dir>/run_config.json
    #   <train_output_dir>/eval/...
    if args.run_config and args.output_dir == "models/rl/eval":
        base_dir = Path(args.run_config).expanduser().resolve().parent
        output_dir = (base_dir / "eval").expanduser().resolve()
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = _load_run_metadata(Path(args.run_config) if args.run_config else None)
    # Build prediction file mapping.
    # Precedence: CLI --prediction_file overrides run_config (so users can fix paths).
    prediction_files: Dict[str, Path] = {}
    if run_metadata and run_metadata.get("prediction_files"):
        prediction_files.update(
            {
                name: Path(path).expanduser().resolve()
                for name, path in run_metadata["prediction_files"].items()
                if path
            }
        )
    prediction_files.update(_parse_prediction_files(args.prediction_file))

    config = RLDataConfig(
        data_dir=data_dir,
        feature_file=Path(args.feature_file).expanduser().resolve()
        if args.feature_file
        else None,
        price_col=run_metadata.get("price_col", args.price_col)
        if run_metadata
        else args.price_col,
        prediction_files=prediction_files,
    )

    print("[eval_rl_agent] Building RL dataframe...")
    df_all = build_rl_dataframe(config)
    _, df_test = split_train_test(df_all)

    if run_metadata:
        feature_cols = run_metadata["feature_cols"]
        prediction_cols = run_metadata.get("prediction_cols", [])
        cost_rate = run_metadata.get("cost_rate", args.cost_rate)
        max_position = run_metadata.get("max_position", args.max_position)
    else:
        feature_cols = config.resolve_feature_list()
        feature_cols = [col for col in feature_cols if col != config.price_col]
        prediction_cols = [col for col in df_test.columns if col.startswith("pred_")]
        cost_rate = args.cost_rate
        max_position = args.max_position

    env = make_trading_env(
        df=df_test,
        price_col=config.price_col,
        feature_cols=feature_cols,
        prediction_cols=prediction_cols,
        cost_rate=cost_rate,
        max_position=max_position,
    )

    model_path = Path(args.model_path).expanduser().resolve()
    print(f"[eval_rl_agent] Loading model from {model_path}")
    model = DQN.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    rewards: List[float] = []
    records: List[Dict] = []
    action_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        rewards.append(reward)
        action_int = int(action)
        if action_int in action_counts:
            action_counts[action_int] += 1
        records.append(
            {
                "t": info["t"],
                "price": info["price"],
                "position": info["position"],
                "action": info.get("action", action_int),
                "pnl": info["pnl"],
                "reward": reward,
            }
        )

    pnl_series = np.array([r["pnl"] for r in records], dtype=float)
    reward_array = np.array(rewards, dtype=float)
    cumulative_return = float(pnl_series[-1]) if pnl_series.size else 0.0
    sharpe_ratio = (
        float((reward_array.mean() / (reward_array.std() + 1e-8)) * np.sqrt(252))
        if reward_array.size
        else 0.0
    )
    max_drawdown = _compute_max_drawdown(pnl_series)

    action_stats = {
        "flat": int(action_counts.get(0, 0)),
        "long": int(action_counts.get(1, 0)),
        "short": int(action_counts.get(2, 0)),
        "total": int(sum(action_counts.values())),
    }

    strategies: Dict[str, Dict[str, float]] = {}
    strategies["rl"] = {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "steps": len(records),
    }

    prices = df_test[config.price_col].to_numpy(dtype=float)
    # Reconstruct a positions vector aligned with prices for turnover/cost accounting.
    rl_positions = np.zeros(len(prices), dtype=float)
    for row in records:
        t = int(row["t"])
        if 0 <= t < len(rl_positions):
            rl_positions[t] = float(row["position"])
    turnover, n_trades, total_trade_cost = _compute_turnover_and_cost(
        prices, rl_positions, cost_rate
    )
    strategies["rl"].update(
        {
            "turnover": float(turnover),
            "n_trades": int(n_trades),
            "total_trade_cost": float(total_trade_cost),
            "gross_pnl": float(cumulative_return + total_trade_cost),
        }
    )

    flat_positions = np.zeros(len(prices), dtype=float)
    flat_pnl, flat_rewards = _compute_pnl_from_positions(prices, flat_positions, cost_rate)
    strategies["always_flat"] = {
        "cumulative_return": float(flat_pnl[-1]) if flat_pnl.size else 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "steps": flat_rewards.size,
    }

    baseline_curves: Dict[str, np.ndarray] = {"rl": pnl_series}

    prediction_cols = [col for col in df_test.columns if col.startswith("pred_")]
    for follow_col in prediction_cols:
        baseline_name = f"follow_{follow_col[5:]}"
        signals = np.sign(df_test[follow_col].to_numpy(dtype=float))
        positions = signals * max_position
        baseline_pnl, baseline_rewards = _compute_pnl_from_positions(
            prices, positions, cost_rate
        )
        baseline_curves[baseline_name] = baseline_pnl
        strategies[baseline_name] = {
            "cumulative_return": float(baseline_pnl[-1]) if baseline_pnl.size else 0.0,
            "sharpe_ratio": float(
                (baseline_rewards.mean() / (baseline_rewards.std() + 1e-8)) * np.sqrt(252)
            )
            if baseline_rewards.size
            else 0.0,
            "max_drawdown": _compute_max_drawdown(baseline_pnl),
            "steps": baseline_rewards.size,
        }

    trace_df = pd.DataFrame(records)
    trace_path = output_dir / "rl_eval_trace.csv"
    trace_df.to_csv(trace_path, index=False)
    print(f"[eval_rl_agent] Saved step trace to {trace_path}")

    with open(output_dir / "rl_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "strategies": strategies,
                "cost_rate": cost_rate,
                "max_position": max_position,
                "rl_total_pnl": cumulative_return,
                "rl_action_counts": action_stats,
            },
            f,
            indent=2,
        )
    print(f"[eval_rl_agent] RL total PnL: {cumulative_return:.6f}")
    print(
        "[eval_rl_agent] Action counts (flat/long/short): "
        f"{action_stats['flat']}/{action_stats['long']}/{action_stats['short']}"
    )
    print(f"[eval_rl_agent] Summary: {strategies}")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(trace_df["pnl"], label="RL Agent")
        if flat_pnl.size:
            plt.plot(np.arange(1, flat_pnl.size + 1), flat_pnl, label="Always Flat")
        for name, curve in baseline_curves.items():
            if name == "rl":
                continue
            plt.plot(np.arange(1, curve.size + 1), curve, label=name)
        plt.title("Strategy PnL Comparison")
        plt.xlabel("Step")
        plt.ylabel("PnL")
        plt.axhline(0, color="black", linestyle="--", alpha=0.3)
        plt.legend()
        plot_path = output_dir / "rl_eval_pnl.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[eval_rl_agent] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()

