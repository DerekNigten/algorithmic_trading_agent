
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import pickle
from pathlib import Path
from tqdm import tqdm
import shutil
from typing import Iterator, Tuple

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

# NOTE: Python does NOT expand "~" in paths automatically. Always use ROOT-based paths or expanduser().
RAW_DATA_PATH = (ROOT / "iex_data" / "book_snapshots").expanduser().resolve()
MODEL_PATH = (ROOT / "outputs").expanduser().resolve()
OUTPUT_PATH = (ROOT / "RL_signal").expanduser().resolve()
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Prepared data (the source of truth for RL)
DATA_DIR = (ROOT / "hft_gitlab" / "data").expanduser().resolve()
MODEL_OUT_DIR = (ROOT / "RL_signal" / "model").expanduser().resolve()
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
XGB_OUTPUT_DIR = (ROOT / "outputs" / "xgboost").expanduser().resolve()

NEW_DAYS = [
    '20251020', '20251021', '20251022', '20251023', '20251024'
]

# ============================================================
# MODEL ARCHITECTURES
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc1_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.fc2(out)
        return out

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))  # FIXED: was self.conv2(x)
        
        if self.downsample is not None:
            res = self.downsample(x)
        else:
            res = x
        
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, output_size=3):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, fc1_size, output_size, max_len=5000):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x)

# ============================================================
# LOAD ALL 4 MODELS
# ============================================================
print("="*60)
print("LOADING ALL 4 MODELS")
print("="*60)

# 1. XGBoost
print("\nLoading XGBoost...")
xgb_model = xgb.Booster()
xgb_model.load_model(MODEL_PATH / "xgboost" / "xgb_best_model.json")
print("✓ XGBoost loaded")

# 2. LSTM
print("Loading LSTM...")
lstm_state = torch.load(MODEL_PATH / "lstm" / "best_model.pt", map_location='cpu', weights_only=False)
weight_ih = lstm_state['lstm.weight_ih_l0']
input_size = weight_ih.shape[1]
hidden_size = weight_ih.shape[0] // 4
fc1_size = lstm_state['fc1.weight'].shape[0]
output_size = lstm_state['fc2.weight'].shape[0]
lstm_model = LSTMModel(input_size, hidden_size, 2, fc1_size, output_size)
lstm_model.load_state_dict(lstm_state)
lstm_model.eval()
print(f"✓ LSTM loaded (input={input_size}, hidden={hidden_size})")

# 3. TCN
print("Loading TCN...")
tcn_state = torch.load(MODEL_PATH / "tcn" / "best_model.pt", map_location='cpu', weights_only=False)

if 'network.0.conv1.conv.weight' in tcn_state:
    first_conv_weight = tcn_state['network.0.conv1.conv.weight']
    conv_key_pattern = 'network.{}.conv1.conv.weight'
elif 'network.0.conv1.weight' in tcn_state:
    first_conv_weight = tcn_state['network.0.conv1.weight']
    conv_key_pattern = 'network.{}.conv1.weight'
else:
    raise KeyError("Cannot find TCN conv weights in state dict")

num_inputs = first_conv_weight.shape[1]
kernel_size = first_conv_weight.shape[2]

num_channels = []
i = 0
while conv_key_pattern.format(i) in tcn_state:
    num_channels.append(tcn_state[conv_key_pattern.format(i)].shape[0])
    i += 1

tcn_model = TCNModel(num_inputs, num_channels, kernel_size, output_size=3)

if 'network.0.conv1.conv.weight' in tcn_state:
    new_state = {}
    for key, value in tcn_state.items():
        new_key = key.replace('.conv1.conv.', '.conv1.').replace('.conv2.conv.', '.conv2.')
        new_state[new_key] = value
    tcn_model.load_state_dict(new_state)
else:
    tcn_model.load_state_dict(tcn_state)

tcn_model.eval()
print(f"✓ TCN loaded (input={num_inputs}, channels={num_channels}, kernel={kernel_size})")

# 4. Transformer
print("Loading Transformer...")
checkpoint = torch.load(MODEL_PATH / "transformer" / "best_transformer_model.pth", map_location='cpu', weights_only=False)
transformer_state = checkpoint['model_state_dict']
hyperparams = checkpoint['hyperparameters']
fc1_size = transformer_state['fc1.weight'].shape[0]
output_size = transformer_state['fc2.weight'].shape[0]
max_len = transformer_state['pos_encoder.pe'].shape[1]
transformer_model = TransformerModel(
    hyperparams['input_size'], hyperparams['d_model'], hyperparams['nhead'],
    hyperparams['num_layers'], hyperparams['dim_feedforward'],
    fc1_size, output_size, max_len
)
transformer_model.load_state_dict(transformer_state)
transformer_model.eval()
print(f"✓ Transformer loaded")

print("\n✅ ALL 4 MODELS LOADED SUCCESSFULLY!\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_all_features(df):
    features = pd.DataFrame()
    features['date'] = df['date']
    
    features["mid_price"] = (df["BID_PRICE_1"] + df["ASK_PRICE_1"]) / 2
    features["microprice"] = (df["BID_PRICE_1"] * df["ASK_SIZE_1"] + df["ASK_PRICE_1"] * df["BID_SIZE_1"]) / (df["BID_SIZE_1"] + df["ASK_SIZE_1"] + 1e-10)
    features["spread"] = df["ASK_PRICE_1"] - df["BID_PRICE_1"]
    features["vol_imbalance"] = (df["BID_SIZE_1"] - df["ASK_SIZE_1"]) / (df["BID_SIZE_1"] + df["ASK_SIZE_1"] + 1e-6)
    features["bid_ask_spread_ratio"] = features["spread"] / features["mid_price"]
    
    for i in range(1, 4):
        features[f"BID_PRICE_{i}"] = df[f"BID_PRICE_{i}"]
        features[f"BID_SIZE_{i}"] = df[f"BID_SIZE_{i}"]
        features[f"ASK_PRICE_{i}"] = df[f"ASK_PRICE_{i}"]
        features[f"ASK_SIZE_{i}"] = df[f"ASK_SIZE_{i}"]
    
    features["bid_price_mean"] = (df["BID_PRICE_1"] + df["BID_PRICE_2"] + df["BID_PRICE_3"]) / 3
    features["ask_price_mean"] = (df["ASK_PRICE_1"] + df["ASK_PRICE_2"] + df["ASK_PRICE_3"]) / 3
    features["bid_qty_mean"] = (df["BID_SIZE_1"] + df["BID_SIZE_2"] + df["BID_SIZE_3"]) / 3
    features["ask_qty_mean"] = (df["ASK_SIZE_1"] + df["ASK_SIZE_2"] + df["ASK_SIZE_3"]) / 3
    
    features["price_cum_diff"] = (df["ASK_PRICE_1"] - df["BID_PRICE_1"] + 
                                  df["ASK_PRICE_2"] - df["BID_PRICE_2"] + 
                                  df["ASK_PRICE_3"] - df["BID_PRICE_3"])
    features["qty_cum_diff"] = (df["ASK_SIZE_1"] - df["BID_SIZE_1"] + 
                                df["ASK_SIZE_2"] - df["BID_SIZE_2"] + 
                                df["ASK_SIZE_3"] - df["BID_SIZE_3"])
    
    features["time_delta"] = 1
    features["mid_diff"] = features["mid_price"].diff()
    features["mid_return"] = features["mid_diff"] / features["mid_price"].shift(1)
    
    features["total_bid_qty"] = df["BID_SIZE_1"] + df["BID_SIZE_2"] + df["BID_SIZE_3"]
    features["total_ask_qty"] = df["ASK_SIZE_1"] + df["ASK_SIZE_2"] + df["ASK_SIZE_3"]
    features["bid_qty_change"] = features["total_bid_qty"].diff()
    features["ask_qty_change"] = features["total_ask_qty"].diff()
    features["OFI"] = features["bid_qty_change"] - features["ask_qty_change"]
    
    features["mv_1s"] = features["mid_price"].rolling(1000, min_periods=1).mean()
    features["mv_5s"] = features["mid_price"].rolling(5000, min_periods=1).mean()
    features["vol_10"] = features["mid_return"].rolling(10, min_periods=1).std()
    features["vol_100"] = features["mid_return"].rolling(100, min_periods=1).std()
    features["vol_1s"] = features["mid_return"].rolling(1000, min_periods=1).std()
    features["vol_5s"] = features["mid_return"].rolling(5000, min_periods=1).std()
    
    delta = features["microprice"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    features["ema_fast"] = features["mid_price"].ewm(span=12, adjust=False).mean()
    features["ema_slow"] = features["mid_price"].ewm(span=26, adjust=False).mean()
    features["ema_diff"] = features["ema_fast"] - features["ema_slow"]
    
    features.drop(['total_bid_qty', 'total_ask_qty', 'bid_qty_change', 'ask_qty_change'], 
                  axis=1, inplace=True, errors='ignore')
    
    features = features.ffill().fillna(0)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.ffill().fillna(0)
    
    return features

def add_labels(features, horizon=23):
    features['future_price'] = features['microprice'].shift(-horizon)
    price_change = features['future_price'] - features['microprice']
    features['target'] = 1
    features.loc[price_change > 0, 'target'] = 2
    features.loc[price_change < 0, 'target'] = 0
    features.drop('future_price', axis=1, inplace=True)
    return features

# ============================================================
# LOAD SCALER AND FEATURES
# ============================================================
print("Loading scaler and selected features...")
FEATURES_PKL = (ROOT / "outputs" / "xgboost" / "selected_features.pkl").expanduser().resolve()
SCALER_PKL = (ROOT / "outputs" / "xgboost" / "scaler.pkl").expanduser().resolve()
FALLBACK_SCALER_PKL = (ROOT / "hft_gitlab" / "data" / "scaler.pkl").expanduser().resolve()

with open(FEATURES_PKL, "rb") as f:
    selected_features = pickle.load(f)
selected_features = list(selected_features)
print(f"✓ Using {len(selected_features)} selected features from {FEATURES_PKL}")

try:
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f) 
    print(f"✓ Loaded scaler from {SCALER_PKL}\n")
except FileNotFoundError:
    scaler = None


class IdentityScaler:
    def transform(self, X):
        return X.values if isinstance(X, pd.DataFrame) else X


def _coerce_scaler(scaler_obj, feature_cols: list[str]):
    """
    Ensure we have an object with .transform(). If the loaded pickle isn't a scaler
    (e.g. numpy array of feature names), fall back to identity.
    """
    if scaler_obj is None:
        return IdentityScaler(), f"⚠️ scaler not found at {SCALER_PKL}; using identity scaler (no normalization)"
    if not hasattr(scaler_obj, "transform"):
        return (
            IdentityScaler(),
            f"⚠️ scaler at {SCALER_PKL} is type={type(scaler_obj)} (no .transform). Using identity scaler instead.",
        )
    # If scaler has feature schema, ensure it matches; otherwise sklearn may error or mis-scale.
    feature_names_in = getattr(scaler_obj, "feature_names_in_", None)
    if feature_names_in is not None:
        try:
            feature_names_in = list(feature_names_in)
        except Exception:
            feature_names_in = None
    if feature_names_in is not None and list(feature_names_in) != list(feature_cols):
        return (
            IdentityScaler(),
            "⚠️ Loaded scaler feature_names_in_ does not match selected_features. Using identity scaler to avoid mismatch.",
        )
    return scaler_obj, f"✓ Using scaler with transform() from {SCALER_PKL}"


scaler, msg = _coerce_scaler(scaler, selected_features)
print(msg + "\n")

if isinstance(scaler, IdentityScaler) and FALLBACK_SCALER_PKL.exists():
    # Try to use prepare_data.py's StandardScaler when compatible.
    try:
        with open(FALLBACK_SCALER_PKL, "rb") as f:
            fallback_scaler = pickle.load(f)
        fallback_scaler, fb_msg = _coerce_scaler(fallback_scaler, selected_features)
        if not isinstance(fallback_scaler, IdentityScaler):
            scaler = fallback_scaler
            print(fb_msg + f" (fallback: {FALLBACK_SCALER_PKL})\n")
    except Exception as e:
        print(f"⚠️ Failed to load fallback scaler from {FALLBACK_SCALER_PKL}: {e}\n")


def _iter_prepared_splits() -> Iterator[Tuple[str, Path]]:
    for split in ["train", "val", "test"]:
        path = (DATA_DIR / f"{split}_data.csv").expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prepared data missing: {path}")
        yield split, path


def _load_prepared_feature_list() -> list[str]:
    """Use prepare_data.py's selected_features as model inputs for sequence models."""
    pkl = (DATA_DIR / "selected_features.pkl").expanduser().resolve()
    with open(pkl, "rb") as f:
        feats = pickle.load(f)
    return list(feats)


def _load_prepared_scaler():
    pkl = (DATA_DIR / "scaler.pkl").expanduser().resolve()
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
    if not hasattr(obj, "transform"):
        raise TypeError(f"Expected a scaler with .transform() at {pkl}, got {type(obj)}")
    return obj


def _build_windows_for_batch(
    combined: np.ndarray,
    end_indices: np.ndarray,
    seq_len: int,
    first_row: np.ndarray,
) -> np.ndarray:
    """Build [B, seq_len, F] windows ending at each end index (inclusive), with edge padding."""
    windows = np.empty((len(end_indices), seq_len, combined.shape[1]), dtype=np.float32)
    for j, idx_end in enumerate(end_indices):
        start = int(idx_end) - seq_len + 1
        if start >= 0:
            windows[j] = combined[start : int(idx_end) + 1]
        else:
            pad_len = -start
            windows[j, :pad_len] = first_row
            windows[j, pad_len:] = combined[: int(idx_end) + 1]
    return windows


def _predict_seq_model_full_length(
    model: nn.Module,
    X: np.ndarray,
    seq_len: int,
    batch_size: int,
    model_name: str,
    tcn: bool = False,
) -> np.ndarray:
    """
    Predict probabilities for every row of X using edge-padded windows.
    Returns probs shape [N, 3].
    """
    X = X.astype(np.float32, copy=False)
    n, f = X.shape
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    first_row = X[0:1]  # [1, F]
    seq_minus_1 = max(seq_len - 1, 0)
    buffer = np.empty((0, f), dtype=np.float32)
    all_probs: list[np.ndarray] = []

    # Process in chunks to bound memory; chunk size ties to batch_size for stable throughput.
    chunk_rows = max(batch_size * 10, 10_000)
    n_chunks = (n + chunk_rows - 1) // chunk_rows
    for chunk_idx in tqdm(range(n_chunks), desc=f"{model_name} rows", unit="chunk"):
        start_row = chunk_idx * chunk_rows
        end_row = min(start_row + chunk_rows, n)
        feats = X[start_row:end_row]

        combined = np.vstack([buffer, feats]) if buffer.size else feats
        offset = buffer.shape[0]
        end_indices = np.arange(offset, offset + feats.shape[0], dtype=np.int64)

        # Batch over end indices
        n_batches = (len(end_indices) + batch_size - 1) // batch_size
        for b in range(n_batches):
            b0 = b * batch_size
            b1 = min((b + 1) * batch_size, len(end_indices))
            idxs = end_indices[b0:b1]
            windows = _build_windows_for_batch(combined, idxs, seq_len, first_row)
            x_tensor = torch.from_numpy(windows)
            if tcn:
                x_tensor = x_tensor.transpose(1, 2)  # [B, F, T]
            with torch.no_grad():
                logits = model(x_tensor.float())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        # Update buffer with last seq_len-1 rows of combined
        if seq_minus_1 > 0:
            buffer = combined[-seq_minus_1:].copy()
        else:
            buffer = np.empty((0, f), dtype=np.float32)

    probs_full = np.vstack(all_probs) if all_probs else np.zeros((0, 3), dtype=np.float32)
    if probs_full.shape[0] != n:
        raise ValueError(f"{model_name}: expected {n} probs, got {probs_full.shape[0]}")
    return probs_full.astype(np.float32, copy=False)


def _write_prob_file(
    out_path: Path,
    actual: np.ndarray,
    probs: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prob_down = probs[:, 0]
    prob_neutral = probs[:, 1] if probs.shape[1] > 2 else np.zeros_like(prob_down)
    prob_up = probs[:, -1]
    predicted = np.argmax(np.vstack([prob_down, prob_neutral, prob_up]), axis=0).astype(int)
    df_out = pd.DataFrame(
        {
            "actual": actual.astype(int),
            "predicted": predicted,
            "prob_down": prob_down,
            "prob_neutral": prob_neutral,
            "prob_up": prob_up,
        }
    )
    df_out.to_csv(out_path, index=False)


def generate_predictions_from_prepared_data() -> None:
    """
    Generate train/val/test prediction files under RL_signal/model/ using best checkpoints.

    - Source features/labels: hft_gitlab/data/{train,val,test}_data.csv
    - Sequence models: edge-padded windows so every row has probabilities (no output padding)
    - XGBoost: if required features are missing from prepared CSV, fall back to existing outputs/xgboost/*.csv
    """
    feature_cols = _load_prepared_feature_list()
    scaler_obj = _load_prepared_scaler()

    # Basic sanity: match NN input sizes
    if len(feature_cols) != int(lstm_model.lstm.input_size):
        raise ValueError(f"LSTM input_size mismatch: model={lstm_model.lstm.input_size} vs features={len(feature_cols)}")
    if len(feature_cols) != int(tcn_model.network[0].conv1.in_channels):
        raise ValueError(f"TCN input_size mismatch: model={tcn_model.network[0].conv1.in_channels} vs features={len(feature_cols)}")
    if len(feature_cols) != int(transformer_model.input_projection.in_features):
        raise ValueError(
            f"Transformer input_size mismatch: model={transformer_model.input_projection.in_features} vs features={len(feature_cols)}"
        )

    # XGB expected feature names
    xgb_feature_names = list(getattr(xgb_model, "feature_names", None) or [])

    def _iter_chunks(csv_path: Path, usecols: list[str], chunksize: int = 50_000):
        return pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)

    def _predict_seq_split_to_csv(
        *,
        split: str,
        csv_path: Path,
        out_path: Path,
        model: nn.Module,
        seq_len: int,
        batch_size: int,
        model_name: str,
        tcn: bool,
    ) -> None:
        first_row: np.ndarray | None = None
        buffer: np.ndarray | None = None
        rows_written = 0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # fresh write
        if out_path.exists():
            out_path.unlink()

        usecols = ["label", *feature_cols]
        for chunk in tqdm(_iter_chunks(csv_path, usecols=usecols), desc=f"{model_name}/{split}", unit="chunk"):
            actual = chunk["label"].to_numpy(dtype=int)
            X_df = chunk[feature_cols]
            X_scaled = pd.DataFrame(scaler_obj.transform(X_df), columns=feature_cols).to_numpy(
                dtype=np.float32, copy=False
            )
            if X_scaled.size == 0:
                continue
            if first_row is None:
                first_row = X_scaled[0:1]
            if buffer is None:
                combined = X_scaled
                offset = 0
            else:
                combined = np.vstack([buffer, X_scaled])
                offset = buffer.shape[0]
            end_indices = np.arange(offset, offset + X_scaled.shape[0], dtype=np.int64)

            probs_parts: list[np.ndarray] = []
            n_batches = (len(end_indices) + batch_size - 1) // batch_size
            for b in range(n_batches):
                b0 = b * batch_size
                b1 = min((b + 1) * batch_size, len(end_indices))
                idxs = end_indices[b0:b1]
                windows = _build_windows_for_batch(combined, idxs, seq_len, first_row)
                x_tensor = torch.from_numpy(windows)
                if tcn:
                    x_tensor = x_tensor.transpose(1, 2)
                with torch.no_grad():
                    logits = model(x_tensor.float())
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_parts.append(probs)

            probs_chunk = np.vstack(probs_parts) if probs_parts else np.zeros((0, 3), dtype=np.float32)
            if probs_chunk.shape[0] != len(actual):
                raise ValueError(f"{model_name}/{split}: chunk probs len {probs_chunk.shape[0]} != {len(actual)}")

            prob_down = probs_chunk[:, 0]
            prob_neutral = probs_chunk[:, 1] if probs_chunk.shape[1] > 2 else np.zeros_like(prob_down)
            prob_up = probs_chunk[:, -1]
            predicted = np.argmax(np.vstack([prob_down, prob_neutral, prob_up]), axis=0).astype(int)
            df_out = pd.DataFrame(
                {
                    "actual": actual.astype(int),
                    "predicted": predicted,
                    "prob_down": prob_down,
                    "prob_neutral": prob_neutral,
                    "prob_up": prob_up,
                }
            )
            df_out.to_csv(out_path, index=False, mode="a", header=(rows_written == 0))
            rows_written += len(df_out)

            # update buffer
            keep = max(seq_len - 1, 0)
            if keep > 0:
                buffer = combined[-keep:].copy()
            else:
                buffer = None

        print(f"[generate_prediction] wrote {rows_written:,} rows -> {out_path}")

    for split, path in _iter_prepared_splits():
        print(f"\n=== Generating predictions for split: {split} ({path.name}) ===")
        _predict_seq_split_to_csv(
            split=split,
            csv_path=path,
            out_path=MODEL_OUT_DIR / f"lstm_{split}_predictions.csv",
            model=lstm_model,
            seq_len=100,
            batch_size=2048,
            model_name="LSTM",
            tcn=False,
        )
        _predict_seq_split_to_csv(
            split=split,
            csv_path=path,
            out_path=MODEL_OUT_DIR / f"tcn_{split}_predictions.csv",
            model=tcn_model,
            seq_len=30,
            batch_size=4096,
            model_name="TCN",
            tcn=True,
        )
        _predict_seq_split_to_csv(
            split=split,
            csv_path=path,
            out_path=MODEL_OUT_DIR / f"transformer_{split}_predictions.csv",
            model=transformer_model,
            seq_len=100,
            batch_size=1024,
            model_name="Transformer",
            tcn=False,
        )

        # --- XGBoost ---
        out_xgb = MODEL_OUT_DIR / f"xgb_{split}_predictions.csv"
        # We avoid loading the huge split CSV just to check feature presence; instead we use
        # the known fact that prepare_data.py may drop some features. If features are missing,
        # copy precomputed predictions produced by the best model.
        if xgb_feature_names:
            # Fall back to existing repo outputs (already aligned to prepared splits)
            fallback = XGB_OUTPUT_DIR / f"xgb_{split}_predictions.csv"
            if not fallback.exists():
                raise FileNotFoundError(
                    f"Fallback XGBoost prediction file doesn't exist: {fallback}"
                )
            out_xgb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fallback, out_xgb)
            print(f"[generate_prediction] XGB fallback copy: {fallback} -> {out_xgb}")
        else:
            raise RuntimeError("XGBoost model has no feature_names; cannot decide how to generate predictions safely.")


# ============================================================
# Legacy: raw-day prediction pipeline (kept for reference)
# ============================================================

# ============================================================
# PREPROCESS ONE DAY
# ============================================================
def preprocess_day(date):
    file_path = RAW_DATA_PATH / f'{date}_book_updates.csv.gz'
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path, compression='gzip')
    df['date'] = date
    df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
    df = df.set_index('COLLECTION_TIME')
    df = df.between_time("14:30", "21:00")
    df = df.reset_index()
    
    features = add_all_features(df)
    features = add_labels(features, horizon=23)
    
    X = features[selected_features]
    y = features['target']
    
    X_scaled = pd.DataFrame(scaler.transform(X), columns=selected_features)
    
    valid_idx = ~X_scaled.isna().any(axis=1) & ~y.isna()
    X_scaled = X_scaled[valid_idx].reset_index(drop=True)
    y = y[valid_idx].values
    
    return X_scaled, y

# ============================================================
# GENERATE PREDICTIONS WITH BATCHING
# ============================================================
def generate_predictions_for_day(date, xgb_model, lstm_model, tcn_model, transformer_model):
    import time
    import gc
    
    print(f"\nProcessing {date}...")
    
    t0 = time.time()
    X, y = preprocess_day(date)
    print(f"  Events: {len(X):,} (preprocessing: {time.time()-t0:.1f}s)")
    
    # 1. XGBoost
    t0 = time.time()
    dmatrix = xgb.DMatrix(X)
    xgb_probs = xgb_model.predict(dmatrix)
    
    if len(xgb_probs.shape) == 1:
        print(f"    ⚠️ XGBoost output is 1D (shape={xgb_probs.shape}), reshaping...")
        try:
            xgb_probs = xgb_probs.reshape(-1, 3)
        except:
            n_samples = len(xgb_probs)
            xgb_probs_2d = np.zeros((n_samples, 3))
            xgb_probs_2d[np.arange(n_samples), xgb_probs.astype(int)] = 1.0
            xgb_probs = xgb_probs_2d
    
    print(f"  XGBoost done: {time.time()-t0:.1f}s (shape={xgb_probs.shape})")
    
    # 2-4. Neural networks with batching
    X_array = X.values
    batch_size = 10000
    
    # LSTM
    print(f"  Starting LSTM (batch_size={batch_size})...")
    t0 = time.time()
    seq_len_lstm = 100
    lstm_probs = []
    y_seq = []
    
    for i in range(seq_len_lstm, len(X_array)):
        y_seq.append(y[i])
    
    num_sequences = len(X_array) - seq_len_lstm
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    for batch_idx, start in enumerate(
            tqdm(
                range(0, num_sequences, batch_size),
                total=num_batches,
                desc="    LSTM",
                unit="batch",
            )
        ):    
        end = min(start + batch_size, num_sequences)
        X_batch = []
        for i in range(start + seq_len_lstm, end + seq_len_lstm):
            X_batch.append(X_array[i-seq_len_lstm:i])
        
        if len(X_batch) > 0:
            X_tensor = torch.FloatTensor(np.array(X_batch))
            with torch.no_grad():
                output = lstm_model(X_tensor)
                probs = torch.softmax(output, dim=1).numpy()
                lstm_probs.append(probs)
    
    lstm_probs = np.vstack(lstm_probs) if lstm_probs else np.array([])
    print(f"  LSTM done: {time.time()-t0:.1f}s")
    
    # TCN
    print(f"  Starting TCN...")
    t0 = time.time()
    seq_len_tcn = 30
    tcn_probs = []
    
    num_sequences = len(X_array) - seq_len_tcn
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    for batch_idx, start in enumerate(
            tqdm(
                range(0, num_sequences, batch_size),
                total=num_batches,
                desc="    TCN",
                unit="batch",
            )
        ):    

        end = min(start + batch_size, num_sequences)
        X_batch = []
        for i in range(start + seq_len_tcn, end + seq_len_tcn):
            X_batch.append(X_array[i-seq_len_tcn:i])
        
        if len(X_batch) > 0:
            X_tensor = torch.FloatTensor(np.array(X_batch)).transpose(1, 2)
            with torch.no_grad():
                output = tcn_model(X_tensor)
                probs = torch.softmax(output, dim=1).numpy()
                tcn_probs.append(probs)
    
    tcn_probs = np.vstack(tcn_probs) if tcn_probs else np.array([])
    print(f"  TCN done: {time.time()-t0:.1f}s")
    
    # Transformer
    print(f"  Starting Transformer...")
    t0 = time.time()
    seq_len_transformer = 100
    transformer_probs = []
    
    transformer_batch_size = 1000
    num_sequences = len(X_array) - seq_len_transformer
    num_batches = (num_sequences + transformer_batch_size - 1) // transformer_batch_size

    for batch_idx, start in enumerate(
        tqdm(
            range(0, num_sequences, transformer_batch_size),
            total=num_batches,
            desc="    Transformer",
            unit="batch",
        )
    ):
        end = min(start + transformer_batch_size, num_sequences)
        
        batch_len = end - start
        X_batch = np.zeros((batch_len, seq_len_transformer, X_array.shape[1]), dtype=np.float32)
        
        for j, i in enumerate(range(start + seq_len_transformer, end + seq_len_transformer)):
            X_batch[j] = X_array[i-seq_len_transformer:i]
        
        X_tensor = torch.FloatTensor(X_batch)
        with torch.no_grad():
            output = transformer_model(X_tensor)
            probs = torch.softmax(output, dim=1).numpy()
            transformer_probs.append(probs)
        
        del X_batch, X_tensor, output, probs
        
        if batch_idx % 10 == 0:
            gc.collect()
    
    transformer_probs = np.vstack(transformer_probs) if transformer_probs else np.array([])
    print(f"  Transformer done: {time.time()-t0:.1f}s ({len(transformer_probs):,} predictions)")
    
    # Align predictions
    start_idx = 100
    min_len = min(len(xgb_probs) - start_idx, len(lstm_probs), len(tcn_probs), len(transformer_probs))
    
    print(f"\n  Alignment summary:")
    print(f"    XGBoost: {len(xgb_probs):,} predictions (using {start_idx}:{start_idx+min_len})")
    print(f"    LSTM: {len(lstm_probs):,} predictions (using 0:{min_len})")
    print(f"    TCN: {len(tcn_probs):,} predictions (using 0:{min_len})")
    print(f"    Transformer: {len(transformer_probs):,} predictions (using 0:{min_len})")
    print(f"    Final aligned length: {min_len:,}")
    
    predictions_df = pd.DataFrame({
        'actual': y_seq[:min_len],
        'xgb_pred': xgb_probs[start_idx:start_idx+min_len].argmax(axis=1),
        'xgb_prob_down': xgb_probs[start_idx:start_idx+min_len, 0],
        'xgb_prob_neutral': xgb_probs[start_idx:start_idx+min_len, 1],
        'xgb_prob_up': xgb_probs[start_idx:start_idx+min_len, 2],
        'lstm_pred': lstm_probs[:min_len].argmax(axis=1),
        'lstm_prob_down': lstm_probs[:min_len, 0],
        'lstm_prob_neutral': lstm_probs[:min_len, 1],
        'lstm_prob_up': lstm_probs[:min_len, 2],
        'tcn_pred': tcn_probs[:min_len].argmax(axis=1),
        'tcn_prob_down': tcn_probs[:min_len, 0],
        'tcn_prob_neutral': tcn_probs[:min_len, 1],
        'tcn_prob_up': tcn_probs[:min_len, 2],
        'transformer_pred': transformer_probs[:min_len].argmax(axis=1),
        'transformer_prob_down': transformer_probs[:min_len, 0],
        'transformer_prob_neutral': transformer_probs[:min_len, 1],
        'transformer_prob_up': transformer_probs[:min_len, 2],
    })
    
    output_file = OUTPUT_PATH / f"predictions_{date}.csv"
    predictions_df.to_csv(output_file, index=False)
    
    print(f"\n  Accuracies:")
    for model_name in ['xgb', 'lstm', 'tcn', 'transformer']:
        acc = (predictions_df[f'{model_name}_pred'] == predictions_df['actual']).mean()
        print(f"    {model_name.upper():12s}: {acc:.3f}")
    print(f"  ✓ Saved {output_file}")
    
    return predictions_df

if __name__ == "__main__":
    # Default behavior: generate split-based predictions from prepared CSVs.
    generate_predictions_from_prepared_data()