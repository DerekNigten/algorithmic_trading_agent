"""
Reinforcement learning trading environment.

State vector: [market_features_t, model_predictions_t, position_{t-1}, unrealized_pnl_{t-1}]
    - market_features_t: ordered values from feature_cols at time t
    - model_predictions_t: optional prediction columns from signal generators (XGBoost, LSTM, TCN, Transformer)
    - position_{t-1}: previous step position scaled in [-max_position, max_position]
    - unrealized_pnl_{t-1}: cumulative PnL up to t-1

Actions:
    0 -> flat (position = 0)
    1 -> long (position = +max_position)
    2 -> short (position = -max_position)

Reward:
    reward_t = pos_{t-1} * (price_t - price_{t-1})
               - cost_rate * abs(pos_t - pos_{t-1}) * price_t

DataFrame requirements:
    - price_col: column containing execution price (e.g., 'mid_price' or 'close')
    - feature_cols: numerical features aligned with existing models
    - prediction_cols: optional model prediction columns used as additional signals
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """Deterministic trading environment used for RL policy learning."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        price_col: str,
        feature_cols: Sequence[str],
        prediction_cols: Sequence[str] | None = None,
        cost_rate: float = 0.0005,
        max_position: int = 1,
    ) -> None:
        super().__init__()
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        if cost_rate < 0:
            raise ValueError("cost_rate must be non-negative")

        self.df = df.reset_index(drop=True).copy()
        missing_cols = [price_col, *feature_cols]
        if prediction_cols:
            missing_cols.extend(prediction_cols)
        for col in missing_cols:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' missing from dataframe")

        self.price_col = price_col
        self.feature_cols = list(feature_cols)
        self.prediction_cols = list(prediction_cols) if prediction_cols else []
        self.cost_rate = float(cost_rate)
        self.max_position = int(max_position)

        self.prices = self.df[price_col].to_numpy(dtype=np.float64)
        feature_matrix = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        if self.prediction_cols:
            pred_matrix = self.df[self.prediction_cols].to_numpy(dtype=np.float32)
            self.feature_matrix = np.hstack([feature_matrix, pred_matrix])
        else:
            self.feature_matrix = feature_matrix

        self.n_steps = len(self.df)
        if self.n_steps < 2:
            raise ValueError("DataFrame must contain at least two rows for trading.")

        obs_size = self.feature_matrix.shape[1] + 2  # position + unrealized_pnl
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        self._t = 1  # start from second row to have price_t and price_{t-1}
        self._position = 0
        self._prev_position = 0
        self._pnl = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment state and return initial observation."""
        super().reset(seed=seed)
        self._t = 1
        self._position = 0
        self._prev_position = 0
        self._pnl = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        """Execute trading action and return transition tuple."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        current_price = self.prices[self._t]
        prev_price = self.prices[self._t - 1]

        self._prev_position = self._position
        self._position = self._action_to_position(action)

        reward = self._prev_position * (current_price - prev_price)
        trade_cost = self.cost_rate * abs(self._position - self._prev_position) * current_price
        reward -= trade_cost
        self._pnl += reward

        done = self._t >= self.n_steps - 1
        info = {
            "pnl": self._pnl,
            "price": current_price,
            "position": self._position,
            "action": int(action),
            "t": self._t,
        }

        self._t += 1
        obs = self._get_observation()
        return obs, float(reward), done, False, info

    def _action_to_position(self, action: int) -> int:
        if action == 0:
            return 0
        if action == 1:
            return self.max_position
        if action == 2:
            return -self.max_position
        raise ValueError(f"Unsupported action {action}")

    def _get_observation(self) -> np.ndarray:
        features = self.feature_matrix[self._t - 1]
        obs = np.concatenate(
            [features, np.array([self._prev_position, self._pnl], dtype=np.float32)]
        )
        return obs.astype(np.float32)


def make_trading_env(
    df: pd.DataFrame,
    price_col: str,
    feature_cols: Iterable[str],
    prediction_cols: Iterable[str] | None = None,
    cost_rate: float = 0.0005,
    max_position: int = 1,
) -> TradingEnv:
    """Helper to instantiate TradingEnv with basic validation."""
    feature_cols = list(feature_cols)
    prediction_cols = list(prediction_cols) if prediction_cols else None

    required_columns: List[str] = [price_col, *feature_cols]
    if prediction_cols:
        required_columns.extend(prediction_cols)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")

    return TradingEnv(
        df=df,
        price_col=price_col,
        feature_cols=feature_cols,
        prediction_cols=prediction_cols,
        cost_rate=cost_rate,
        max_position=max_position,
    )

