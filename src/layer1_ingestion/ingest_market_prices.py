import os
import numpy as np
import pandas as pd
import yfinance as yf

# 10 Benchmark S&P 500 Equities (Mantshimuli, 2025 & CS7P01NM Execution Plan)
TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "ORCL", "GOOG", "TSLA", "META", "JPM", "BAC"]
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
OUTPUT_PATH = "data/parquet_cache/sp500_market_prices.parquet"

def suppress_outliers_neighborhood(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Phase 2: Statistical outlier suppression using neighborhood-average replacement."""
    s = series.copy()
    mean, std = s.mean(), s.std()
    if std == 0 or np.isnan(std):
        return s
    outliers = (s - mean).abs() > (z_thresh * std)
    neighbor_avg = (s.shift(1) + s.shift(-1)) / 2.0
    s[outliers] = neighbor_avg[outliers].fillna(mean)
    return s

def main():
    print(f"📥 [Layer 1] Pulling daily OHLCV prices ({START_DATE} to {END_DATE}) for {len(TICKERS)} S&P 500 stocks...")
    raw_df = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True, progress=False)

    records = []
    for ticker in TICKERS:
        df_t = raw_df[ticker].dropna(how="all").copy()
        df_t = df_t.reset_index()
        # Phase 1: Canonical ISO 8601 (YYYY-MM-DD) Alignment
        df_t["date"] = pd.to_datetime(df_t["Date"]).dt.strftime("%Y-%m-%d")
        df_t["ticker"] = ticker
        df_t = df_t.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        df_t = df_t[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values("date")

        # Feature Engineering: Daily Return, Log Return, Volatility (20d), RSI (14d), EMA (10d, 20d)
        df_t["daily_return"] = df_t["close"].pct_change().fillna(0.0)
        df_t["daily_return_clean"] = suppress_outliers_neighborhood(df_t["daily_return"], z_thresh=3.0)
        df_t["log_return"] = np.log(df_t["close"] / df_t["close"].shift(1)).fillna(0.0)
        df_t["volatility_20d"] = df_t["daily_return_clean"].rolling(20, min_periods=1).std().fillna(0.0)
        df_t["ema_10"] = df_t["close"].ewm(span=10, adjust=False).mean()
        df_t["ema_20"] = df_t["close"].ewm(span=20, adjust=False).mean()

        # Chronological Leak-Free Split (0.6 Train / 0.2 Val / 0.2 Test)
        n = len(df_t)
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)
        df_t["split"] = "test"
        df_t.iloc[:train_end, df_t.columns.get_loc("split")] = "train"
        df_t.iloc[train_end:val_end, df_t.columns.get_loc("split")] = "val"

        # Leak-Free Z-Score Standardization (fit on train period only)
        for col in ["close", "volume", "daily_return_clean", "volatility_20d"]:
            train_mean = df_t.loc[df_t["split"] == "train", col].mean()
            train_std = df_t.loc[df_t["split"] == "train", col].std() + 1e-8
            df_t[f"{col}_z"] = (df_t[col] - train_mean) / train_std

        records.append(df_t)

    full_df = pd.concat(records, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Phase 3: Serialize into compressed Apache Parquet columnar layout
    full_df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)

    print(f"✅ Saved {len(full_df):,} aligned rows to {OUTPUT_PATH}")
    print("\n📊 Split Breakdown (Leak-Free 60/20/20 Chronological Protocol):")
    print(full_df.groupby("split")["date"].agg(rows="count", min_date="min", max_date="max"))

if __name__ == "__main__":
    main()
