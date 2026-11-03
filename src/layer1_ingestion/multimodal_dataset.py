import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MASTER_PARQUET = "data/parquet_cache/rq1_multimodal_master.parquet"

class MultimodalWealthDataset(Dataset):
    """
    PyTorch Dataset for CS7P01NM Multimodal Alternative Data Fusion (RQ1-RQ4).
    Supports:
      - Sliding window extraction [t-K, t-1] -> target return at t
      - Binary modality masking [m_price, m_news, m_esg, m_spatial] (RQ4 Ablation)
      - Asynchronous alternative data delay simulation (Supervisor Feedback #3)
    """
    def __init__(
        self,
        parquet_path: str = MASTER_PARQUET,
        split: str = "train",
        window_size: int = 20,
        modality_mask: list = (1, 1, 1, 1),
        alt_data_delay_days: int = 0
    ):
        super().__init__()
        self.window_size = window_size
        self.modality_mask = modality_mask
        self.alt_data_delay_days = alt_data_delay_days

        df = pd.read_parquet(parquet_path)
        df = df[df["split"] == split].sort_values(["date", "ticker"]).reset_index(drop=True)

        self.tickers = sorted(df["ticker"].unique().tolist())
        self.dates = sorted(df["date"].unique().tolist())
        num_dates = len(self.dates)
        num_tickers = len(self.tickers)

        price_cols = ["close_z", "volume_z", "daily_return_clean_z", "volatility_20d_z"]
        news_cols = ["finbert_sentiment", "finbert_sentiment_30d", "social_hype_noise_flag"] + [f"finbert_emb_{i}" for i in range(16)]
        esg_cols = ["esg_score_z"]
        spatial_cols = ["spatial_activity_idx_z", "socio_political_idx_z"]

        # Pivot to (num_dates, num_tickers, num_features)
        self.price_tensor = np.stack([df[df["ticker"] == t][price_cols].values for t in self.tickers], axis=1).astype(np.float32)
        self.news_tensor = np.stack([df[df["ticker"] == t][news_cols].values for t in self.tickers], axis=1).astype(np.float32)
        self.esg_tensor = np.stack([df[df["ticker"] == t][esg_cols].values for t in self.tickers], axis=1).astype(np.float32)
        self.spatial_tensor = np.stack([df[df["ticker"] == t][spatial_cols].values for t in self.tickers], axis=1).astype(np.float32)
        self.target_returns = np.stack([df[df["ticker"] == t]["daily_return_clean"].values for t in self.tickers], axis=1).astype(np.float32)

        # Apply artificial latency shift to alternative modalities (Feedback #3: Delayed Data Test)
        if self.alt_data_delay_days > 0:
            d = self.alt_data_delay_days
            self.news_tensor = np.roll(self.news_tensor, shift=d, axis=0)
            self.news_tensor[:d] = 0.0
            self.esg_tensor = np.roll(self.esg_tensor, shift=d, axis=0)
            self.esg_tensor[:d] = 0.0
            self.spatial_tensor = np.roll(self.spatial_tensor, shift=d, axis=0)
            self.spatial_tensor[:d] = 0.0

        self.valid_indices = list(range(self.window_size, num_dates))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        m_price, m_news, m_esg, m_spatial = self.modality_mask

        # Sliding window [t - K, t - 1]
        x_price = self.price_tensor[t - self.window_size : t] * float(m_price)
        x_news = self.news_tensor[t - self.window_size : t] * float(m_news)
        x_esg = self.esg_tensor[t - self.window_size : t] * float(m_esg)
        x_spatial = self.spatial_tensor[t - self.window_size : t] * float(m_spatial)

        # Next-day asset return vector at day t
        y_return = self.target_returns[t]

        return {
            "price": torch.from_numpy(x_price),       # (K, N_stocks, 4)
            "news": torch.from_numpy(x_news),         # (K, N_stocks, 19)
            "esg": torch.from_numpy(x_esg),           # (K, N_stocks, 1)
            "spatial": torch.from_numpy(x_spatial),   # (K, N_stocks, 2)
            "target_return": torch.from_numpy(y_return) # (N_stocks,)
        }

if __name__ == "__main__":
    ds_train = MultimodalWealthDataset(split="train", window_size=20, modality_mask=[1, 1, 1, 1], alt_data_delay_days=0)
    ds_delayed = MultimodalWealthDataset(split="test", window_size=20, modality_mask=[1, 1, 1, 1], alt_data_delay_days=3)
    loader = DataLoader(ds_train, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    print("✅ [Layer 1 PyTorch Tensor Check] Train Samples:", len(ds_train), "| Delayed Test Samples:", len(ds_delayed))
    for k, v in batch.items():
        print(f"   • Tensor '{k}': shape = {tuple(v.shape)}, dtype = {v.dtype}")
