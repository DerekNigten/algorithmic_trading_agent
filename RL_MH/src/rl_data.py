"""
Utilities for building RL-ready datasets from existing supervised learning artifacts.

This module expects the artifacts produced by `scripts/prepare_data.py`, namely:
    - train_data.csv, val_data.csv, test_data.csv stored under a common directory
    - selected_features.pkl describing the normalized feature set

Optional model prediction files (XGBoost, LSTM, TCN, Transformer) can also be supplied.
Each prediction file should align chronologically with the concatenated train/val/test
data so the RL environment can treat them as additional signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pickle


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RLDataConfig:
    """Configuration bundle for assembling RL datasets."""

    data_dir: Path
    train_file: Path | None = None
    val_file: Path | None = None
    test_file: Path | None = None
    feature_file: Path | None = None
    feature_list: Sequence[str] | None = None
    price_col: str = "mid_price"
    prediction_files: Mapping[str, Path] | None = None

    def resolve_train_file(self) -> Path:
        path = self.train_file or (self.data_dir / "train_data.csv")
        if not path.exists():
            raise FileNotFoundError(f"Train file not found at {path}")
        return path

    def resolve_val_file(self) -> Path | None:
        path = self.val_file or (self.data_dir / "val_data.csv")
        return path if path.exists() else None

    def resolve_test_file(self) -> Path:
        path = self.test_file or (self.data_dir / "test_data.csv")
        if not path.exists():
            raise FileNotFoundError(f"Test file not found at {path}")
        return path

    def resolve_feature_list(self) -> List[str]:
        if self.feature_list:
            return list(self.feature_list)
        if not self.feature_file:
            candidate = self.data_dir / "selected_features.pkl"
            if candidate.exists():
                self.feature_file = candidate
        if not self.feature_file or not self.feature_file.exists():
            raise FileNotFoundError(
                "Feature list not provided; expected feature_file or feature_list."
            )
        with open(self.feature_file, "rb") as f:
            features: List[str] = pickle.load(f)
        return features


def load_base_data(config: RLDataConfig) -> Dict[str, pd.DataFrame]:
    """Load train/val/test CSVs produced by prepare_data.py."""
    splits: Dict[str, pd.DataFrame] = {}
    splits["train"] = pd.read_csv(config.resolve_train_file())

    val_file = config.resolve_val_file()
    if val_file:
        splits["val"] = pd.read_csv(val_file)

    splits["test"] = pd.read_csv(config.resolve_test_file())

    return splits


def _load_prediction_file(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "prediction" in df.columns:
        series = df["prediction"]
    else:
        series = df.iloc[:, 0]
    return series.reset_index(drop=True)


def _resolve_prediction_path_with_fallback(path: Path) -> Path | None:
    """
    Try to resolve a missing prediction file by searching common project locations.

    This helps when run_config.json was generated on a different machine / folder layout,
    or when signals were moved between `RL_signal/` and `outputs/`.
    """
    try:
        if path.exists():
            return path
    except OSError:
        # In case of weird paths, fall through to candidates.
        pass

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / path.name,  # project root
        root / "RL_signal" / path.name,
        root / "outputs" / "model" / path.name,
        root / "outputs" / "RL_signal" / path.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_model_predictions(
    config: RLDataConfig,
    expected_length: int | None = None,
) -> Dict[str, pd.Series | None]:
    """
    Load prediction vectors for each signal generator.

    Files should contain a single column named 'prediction' (or the first column is used).
    Missing files are tolerated and will be represented as None.
    """
    predictions: Dict[str, pd.Series | None] = {}
    if not config.prediction_files:
        return predictions
    for name, path in config.prediction_files.items():
        series: pd.Series | None = None
        try:
            raw_path = Path(path)
            resolved = _resolve_prediction_path_with_fallback(raw_path)
            if resolved is None:
                raise FileNotFoundError(raw_path)
            if resolved != raw_path:
                logger.warning(
                    "Prediction file for '%s' not found at %s. Using fallback at %s.",
                    name,
                    raw_path,
                    resolved,
                )
            series = _load_prediction_file(resolved)
        except FileNotFoundError:
            if expected_length is None:
                logger.warning(
                    "Prediction file for '%s' missing at %s. Using zeros as placeholder.",
                    name,
                    path,
                )
                series = None
            else:
                logger.warning(
                    "Prediction file for '%s' missing at %s. Using zeros as placeholder.",
                    name,
                    path,
                )
                series = pd.Series(np.zeros(expected_length), name=name)
        if series is None:
            predictions[name] = None
            continue
        if expected_length is not None and len(series) != expected_length:
            raise ValueError(
                f"Prediction file for '{name}' has length {len(series)}, "
                f"but base RL DataFrame has length {expected_length}. "
                "Ensure predictions were generated over the combined train+val+test split "
                "and aligned chronologically."
            )
        predictions[name] = series.reset_index(drop=True)
    return predictions


def build_rl_dataframe(config: RLDataConfig) -> pd.DataFrame:
    """
    Merge base features, price column, and model predictions into one chronological DataFrame.
    The resulting DataFrame contains a 'split' column tagging train/val/test rows.
    """
    splits = load_base_data(config)
    feature_cols = config.resolve_feature_list()

    ordered_cols = list(feature_cols)
    if config.price_col not in ordered_cols:
        ordered_cols.append(config.price_col)
    required_set = set(ordered_cols)
    frames: List[pd.DataFrame] = []

    for split_name, df in splits.items():
        missing = [col for col in required_set if col not in df.columns]
        if missing:
            if config.price_col in missing:
                raise KeyError(
                    f"price_col '{config.price_col}' not found in {split_name} split. "
                    "Ensure prepare_data.py retained this column (e.g., include it in "
                    "selected_features) or add it manually before running RL."
                )
            raise KeyError(f"Missing columns in {split_name} split: {missing}")
        df_split = df.loc[:, ordered_cols].copy()
        df_split["split"] = split_name
        frames.append(df_split)

    combined = pd.concat(frames, axis=0, ignore_index=True)

    prediction_series = load_model_predictions(config, expected_length=len(combined))
    for model_name, series in prediction_series.items():
        if series is None:
            continue
        combined[f"pred_{model_name}"] = series.to_numpy()
    # NOTE: Future extension could support per-split prediction files by loading them
    # before concatenation and aligning each split separately.

    return combined


def split_train_test(
    df: pd.DataFrame,
    train_splits: Iterable[str] = ("train", "val"),
    test_splits: Iterable[str] = ("test",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return train/test DataFrames following prepare_data.py ordering.

    prepare_data.py builds Day1-3 (train), Day4 (val), Day5 (test) and we preserve
    that chronological order via the 'split' column.

    Args:
        df: Combined dataframe produced by build_rl_dataframe
        train_splits: Which split tags should be treated as training data
        test_splits: Which split tags should be treated as testing data
    """
    if "split" not in df.columns:
        raise KeyError(
            "DataFrame must include 'split' column for partitioning. "
            "Ensure build_rl_dataframe() was used and that prepare_data.py provided "
            "train/val/test splits."
        )

    train_mask = df["split"].isin(list(train_splits))
    test_mask = df["split"].isin(list(test_splits))

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)

    return df_train, df_test


__all__ = [
    "RLDataConfig",
    "load_base_data",
    "load_model_predictions",
    "build_rl_dataframe",
    "split_train_test",
]

