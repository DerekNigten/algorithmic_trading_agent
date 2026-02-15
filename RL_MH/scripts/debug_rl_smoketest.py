#!/usr/bin/env python3
"""
Lightweight smoke test for the RL trading environment and training loop.

Creates a synthetic price series with dummy features/predictions, then:
    1. Runs a short random policy rollout while printing step diagnostics.
    2. Optionally trains a small DQN agent for a few steps to ensure plumbing works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import DQN

from src.rl_trading_env import make_trading_env


def build_synthetic_dataframe(n_steps: int = 200) -> pd.DataFrame:
    """Create a simple synthetic price + feature dataset."""
    rng = np.random.default_rng(42)
    price = np.linspace(100, 101, n_steps) + rng.normal(scale=0.05, size=n_steps)
    df = pd.DataFrame(
        {
            "mid_price": price,
            "ret_1": np.append([0], np.diff(price)),
            "ret_5": pd.Series(price).diff(5).fillna(0.0),
            "pred_dummy": np.zeros(n_steps),
        }
    )
    return df


def run_random_rollout(env, steps: int = 10) -> None:
    """Sample random actions and print a short trace."""
    obs, _ = env.reset()
    print("\n[debug_rl_smoketest] Random rollout:")
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"t={info['t']:03d}, price={info['price']:.4f}, "
            f"pos={info['position']}, reward={reward:.6f}, pnl={info['pnl']:.6f}"
        )
        if done:
            break


def maybe_train_dqn(env, timesteps: int):
    """Run a tiny DQN training loop if timesteps > 0."""
    if timesteps <= 0:
        return
    print(f"\n[debug_rl_smoketest] Training DQN for {timesteps} timesteps...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        batch_size=32,
        buffer_size=1000,
        learning_starts=100,
        target_update_interval=200,
        train_freq=4,
        gradient_steps=1,
        gamma=0.95,
        verbose=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    print("[debug_rl_smoketest] DQN training completed.")
    return model


def collect_trained_trace(model, env, steps: int) -> pd.DataFrame:
    """Run the trained agent and return the recorded trace (pnl/time)."""
    obs, _ = env.reset()
    records: list[dict] = []
    done = False
    while not done and len(records) < steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        records.append(info)
    return pd.DataFrame(records)


def save_trace_plot(trace: pd.DataFrame, path: str) -> None:
    """Persist cumulative PnL chart."""
    if trace.empty:
        return
    plot_path = Path(path).expanduser()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(trace["t"], trace["pnl"], label="RL Agent PnL", color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("PnL")
    plt.title("RL Smoke Test: Post-training PnL")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"[debug_rl_smoketest] Saved PnL plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic smoke test for RL pipeline.")
    parser.add_argument("--steps", type=int, default=200, help="Number of synthetic timesteps")
    parser.add_argument(
        "--dqn_timesteps",
        type=int,
        default=0,
        help="If >0, run a short DQN training on the synthetic environment",
    )
    parser.add_argument("--random_steps", type=int, default=10, help="Random rollout steps to print")
    parser.add_argument("--cost_rate", type=float, default=0.0001, help="Transaction cost rate")
    parser.add_argument(
        "--plot_output",
        type=str,
        default="logs/RL/rl_smoketest_pnl.png",
        help="Path to save the post-training PnL plot (empty to skip)",
    )
    parser.add_argument(
        "--plot_steps",
        type=int,
        default=200,
        help="Maximum steps to collect for the post-training PnL trace",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_synthetic_dataframe(args.steps)
    env = make_trading_env(
        df=df,
        price_col="mid_price",
        feature_cols=["ret_1", "ret_5"],
        prediction_cols=["pred_dummy"],
        cost_rate=args.cost_rate,
        max_position=1,
    )

    run_random_rollout(env, steps=args.random_steps)
    trained_model = maybe_train_dqn(env, timesteps=args.dqn_timesteps)
    if trained_model and args.plot_output:
        trace = collect_trained_trace(trained_model, env, steps=args.plot_steps)
        save_trace_plot(trace, args.plot_output)


if __name__ == "__main__":
    main()

