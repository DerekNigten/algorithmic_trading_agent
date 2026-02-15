import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# ===== CONFIGURATION =====
NUMBER_EVENTS_AHEAD = 23  # Change this easily
MAX_CORR = 0.85          # Change this easily

def main():
    # Setup paths
    data_dir = Path("~/Desktop/code/group_01_project/iex_data/book_snapshots").expanduser()
    output_dir = Path("~/Desktop/code/group_01_project/hft_gitlab/data").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File names
    days = [
        '20251020_book_updates.csv.gz',
        '20251021_book_updates.csv.gz',
        '20251022_book_updates.csv.gz',
        '20251023_book_updates.csv.gz',
        '20251024_book_updates.csv.gz'
    ]
    
    # Process each day
    print("\n" + "="*60)
    print("PROCESSING 5 DAYS OF DATA")
    print("="*60)
    processed_days = []
    for day_file in days:
        df_day = process_single_day(data_dir / day_file)
        processed_days.append(df_day)
    
    # Concatenate for feature selection (use all 5 days)
    print("\nCombining all days for feature selection...")
    df_all = pd.concat(processed_days, ignore_index=True)
    print(f"Total rows: {len(df_all):,}")
    
    # Feature selection
    selected_features = select_features(df_all, NUMBER_EVENTS_AHEAD, MAX_CORR)
    
    # Create labels for each day separately
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("="*60)
    
    for i, df_day in enumerate(processed_days):
        df_day["label"] = 1  # Default: no change
        df_day.loc[df_day["mid_price"].shift(-NUMBER_EVENTS_AHEAD) > df_day["mid_price"], "label"] = 2  # Up
        df_day.loc[df_day["mid_price"].shift(-NUMBER_EVENTS_AHEAD) < df_day["mid_price"], "label"] = 0  # Down
        df_day = df_day.dropna()
        processed_days[i] = df_day
    
    # Split
    train_df = pd.concat([processed_days[0], processed_days[1], processed_days[2]], ignore_index=True)
    val_df = processed_days[3].reset_index(drop=True)
    test_df = processed_days[4].reset_index(drop=True)
    
    print(f"Train (Days 1-3): {len(train_df):,} samples")
    print(f"Val (Day 4):      {len(val_df):,} samples")
    print(f"Test (Day 5):     {len(test_df):,} samples")
    
    # Normalize (fit on train only!)
    print("\nNormalizing features...")
    scaler = StandardScaler()
    train_df[selected_features] = scaler.fit_transform(train_df[selected_features])
    val_df[selected_features] = scaler.transform(val_df[selected_features])
    test_df[selected_features] = scaler.transform(test_df[selected_features])
    
    # Keep only selected features + label, and always retain mid_price for RL (price_col)
    columns_to_keep = selected_features + ['label']
    if 'mid_price' not in columns_to_keep:
        columns_to_keep.append('mid_price')
    train_df = train_df[columns_to_keep]
    val_df = val_df[columns_to_keep]
    test_df = test_df[columns_to_keep]
    
    # Save
    print("\nSaving files...")
    train_df.to_csv(output_dir / 'train_data.csv', index=False)
    val_df.to_csv(output_dir / 'val_data.csv', index=False)
    test_df.to_csv(output_dir / 'test_data.csv', index=False)
    
    with open(output_dir / 'selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Print label distributions
    print("\n" + "="*60)
    print("LABEL DISTRIBUTIONS")
    print("="*60)
    print("\nTrain:")
    print(train_df['label'].value_counts(normalize=True).sort_index())
    print("\nVal:")
    print(val_df['label'].value_counts(normalize=True).sort_index())
    print("\nTest:")
    print(test_df['label'].value_counts(normalize=True).sort_index())
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nFiles saved to: {output_dir}")
    print("  - train_data.csv")
    print("  - val_data.csv")
    print("  - test_data.csv")
    print("  - selected_features.pkl")
    print("  - scaler.pkl")


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def engineer_features(df):
    """Engineer features from raw order book data"""
    # Basic Level-1 features
    df["mid_price"] = (df["BID_PRICE_1"] + df["ASK_PRICE_1"]) / 2
    df["microprice"] = (df["BID_PRICE_1"] * df["ASK_SIZE_1"] + df["ASK_PRICE_1"] * df["BID_SIZE_1"]) / (df["BID_SIZE_1"] + df["ASK_SIZE_1"] + 1e-10)
    df["spread"] = df["ASK_PRICE_1"] - df["BID_PRICE_1"]
    df["vol_imbalance"] = (df["BID_SIZE_1"] - df["ASK_SIZE_1"]) / (df["BID_SIZE_1"] + df["ASK_SIZE_1"] + 1e-6)
    df["bid_ask_spread_ratio"] = df["spread"] / df["mid_price"]

    # Mean price/quantity across all 3 levels
    df["bid_price_mean"] = df[["BID_PRICE_1", "BID_PRICE_2", "BID_PRICE_3"]].mean(axis=1)
    df["ask_price_mean"] = df[["ASK_PRICE_1", "ASK_PRICE_2", "ASK_PRICE_3"]].mean(axis=1)
    df["bid_qty_mean"] = df[["BID_SIZE_1", "BID_SIZE_2", "BID_SIZE_3"]].mean(axis=1)
    df["ask_qty_mean"] = df[["ASK_SIZE_1", "ASK_SIZE_2", "ASK_SIZE_3"]].mean(axis=1)

    # Cumulative differences across 3 levels
    df["price_cum_diff"] = (df["ASK_PRICE_1"] - df["BID_PRICE_1"] + df["ASK_PRICE_2"] - df["BID_PRICE_2"] + df["ASK_PRICE_3"] - df["BID_PRICE_3"])
    df["qty_cum_diff"] = (df["ASK_SIZE_1"] - df["BID_SIZE_1"] + df["ASK_SIZE_2"] - df["BID_SIZE_2"] + df["ASK_SIZE_3"] - df["BID_SIZE_3"])

    # Time intervals between events
    df["time_delta"] = df.index.to_series().diff().dt.total_seconds()

    # Price momentum
    df["mid_diff"] = df["mid_price"].diff()
    df["mid_return"] = df["mid_diff"] / df["mid_price"].shift(1)

    # Order Flow Imbalance (OFI)
    df["total_bid_qty"] = df[["BID_SIZE_1", "BID_SIZE_2", "BID_SIZE_3"]].sum(axis=1)
    df["total_ask_qty"] = df[["ASK_SIZE_1", "ASK_SIZE_2", "ASK_SIZE_3"]].sum(axis=1)
    df["bid_qty_change"] = df["total_bid_qty"].diff()
    df["ask_qty_change"] = df["total_ask_qty"].diff()
    df["OFI"] = df["bid_qty_change"] - df["ask_qty_change"]

    # Moving averages
    df["mv_1s"] = df["mid_price"].rolling(1000, min_periods=1).mean()
    df["mv_5s"] = df["mid_price"].rolling(5000, min_periods=1).mean()

    # Volatility
    df["vol_10"] = df["mid_return"].rolling(10, min_periods=1).std()
    df["vol_100"] = df["mid_return"].rolling(100, min_periods=1).std()
    df["vol_1s"] = df["mid_return"].rolling(1000, min_periods=1).std()
    df["vol_5s"] = df["mid_return"].rolling(5000, min_periods=1).std()

    # RSI
    df["rsi_14"] = calculate_rsi(df["mid_price"], 14)

    # EMA
    df["ema_fast"] = df["mid_price"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["mid_price"].ewm(span=26, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
    
    return df

def process_single_day(file_path):
    """Load and process one day of data"""
    print(f"Processing {file_path.name}...")
    
    df = pd.read_csv(file_path, compression='gzip')
    df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
    df = df.set_index('COLLECTION_TIME')
    df = df.between_time("14:30", "21:00")
    
    df = engineer_features(df)
    
    print(f"  Rows: {len(df):,}")
    return df

def select_features(df, number_events_ahead, max_corr):
    """Select features using iterative correlation removal"""
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION: {number_events_ahead} events ahead, max_corr={max_corr}")
    print(f"{'='*60}")

    # Create label (temporary for feature selection)
    temp_df = df.copy()
    temp_df["label"] = 1  # Default: no change
    temp_df.loc[temp_df["mid_price"].shift(-number_events_ahead) > temp_df["mid_price"], "label"] = 2  # Up
    temp_df.loc[temp_df["mid_price"].shift(-number_events_ahead) < temp_df["mid_price"], "label"] = 0  # Down

    # Define all features
    feature_names = [
        'BID_PRICE_1', 'BID_SIZE_1', 'BID_PRICE_2', 'BID_SIZE_2', 'BID_PRICE_3', 'BID_SIZE_3',
        'ASK_PRICE_1', 'ASK_SIZE_1', 'ASK_PRICE_2', 'ASK_SIZE_2', 'ASK_PRICE_3', 'ASK_SIZE_3',
        'mid_price', 'microprice', 'spread', 'vol_imbalance', 'bid_ask_spread_ratio',
        'bid_price_mean', 'ask_price_mean', 'bid_qty_mean', 'ask_qty_mean', 'price_cum_diff',
        'qty_cum_diff', 'time_delta', 'mid_diff', 'mid_return', 'OFI', 'mv_1s', 'mv_5s', 
        'vol_10', 'vol_100', 'vol_1s', 'vol_5s', 'rsi_14', 'ema_fast', 'ema_slow', 'ema_diff'
    ]
    feature_names = [f for f in feature_names if f in temp_df.columns]

    temp_df = temp_df.dropna()

    # 為了加速，在非常大的資料集上只用一部分樣本來做 MI 與相關性計算
    max_samples = 200_000
    if len(temp_df) > max_samples:
        print(f"\nSubsampling from {len(temp_df):,} rows to {max_samples:,} rows for feature selection...")
        temp_df_sample = temp_df.sample(n=max_samples, random_state=42)
    else:
        temp_df_sample = temp_df

    # Compute MI scores
    print(f"\nComputing MI scores for {len(feature_names)} features on {len(temp_df_sample):,} samples...")
    X = temp_df_sample[feature_names].values
    y = temp_df_sample['label'].values
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
    
    mi_dict = dict(zip(feature_names, mi_scores))
    
    # Start with all features
    selected_features = feature_names.copy()
    
    print(f"Initial features: {len(selected_features)}")
    
    # Iterative removal
    iteration = 0
    while True:
        iteration += 1

        # Compute correlation matrix on the same subsample
        corr_matrix, _ = spearmanr(temp_df_sample[selected_features].values)
        
        # Find highest correlation above threshold
        max_correlation = -1
        remove_feat = None
        
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                corr_val = abs(corr_matrix[i, j])
                
                if corr_val > max_corr and corr_val > max_correlation:
                    max_correlation = corr_val
                    feat1 = selected_features[i]
                    feat2 = selected_features[j]
                    
                    # Remove the one with lower MI
                    if mi_dict[feat1] < mi_dict[feat2]:
                        remove_feat = feat1
                        keep_feat = feat2
                    else:
                        remove_feat = feat2
                        keep_feat = feat1
        
        # If no more high correlations, stop
        if remove_feat is None:
            break
        
        # Remove feature
        print(f"Iter {iteration}: Removing {remove_feat} (corr={max_correlation:.3f} with {keep_feat}, MI={mi_dict[remove_feat]:.4f})")
        selected_features.remove(remove_feat)
    
    print(f"\nFinal features: {len(selected_features)}")
    print(f"Removed: {len(feature_names) - len(selected_features)} features")

    print("\n" + "="*60)
    print("SELECTED FEATURES WITH MI SCORES (sorted by MI)")
    print("="*60)

    selected_mi = [(feat, mi_dict[feat]) for feat in selected_features]
    selected_mi.sort(key=lambda x: x[1], reverse=True)
    
    for feat, mi_score in selected_mi:
        print(f"{feat:25s} {mi_score:.6f}")
    
    return selected_features

if __name__ == '__main__':
    main()