# CS7P01NM Data Catalogue (Layer 1 — Multimodal Ingestion)

| Dataset File | Modality | Rows | Key Features | Split Protocol |
| :--- | :--- | :--- | :--- | :--- |
| `sp500_market_prices.parquet` | Time-Series (OHLCV) | 27,650 | `close_z`, `daily_return_clean_z`, `volatility_20d_z`, `ema_10`, `ema_20` | 60/20/20 Chronological |
| `finbert_embeddings.parquet` | Text NLP (FinBERT) | 27,650 | `finbert_sentiment`, `finbert_sentiment_30d`, `social_hype_noise_flag`, `finbert_emb_0..15` | Aligned ISO 8601 |
| `rq1_multimodal_master.parquet` | Fused 4-Modality Master | 27,650 | Joined Price + FinBERT + `esg_score_z` + `spatial_activity_idx_z` + `socio_political_idx_z` | Leak-Free Train Z-Score |
| `retail_client_profiles.parquet` | Synthetic Retail Profiles | 5,000 | `risk_persona`, `risk_tolerance_score`, `esg_preference_weight`, `client_node` | 5 Federated Client Partitions |
