# Multimodal Alternative Data Fusion for Holistic Personal Wealth Management for Retail Investors Using AI

[![Module](https://img.shields.io/badge/Module-CS7P01NM%20%signals-blue)](https://londonmet.ac.uk)
[![Degree](https://img.shields.io/badge/Degree-MSc%20Artificial%20Intelligence-green)](https://londonmet.ac.uk)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blueviolet)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains the official Proof of Concept (PoC) implementation for the MSc Research Project titled **"Multimodal Alternative Data Fusion for Holistic Personal Wealth Management for Retail Investors Using AI"**. 

Traditional robo-advisors and financial AI systems are predominantly tailored for institutional actors, leaving retail investors suffering from severe information asymmetry and reliant on static single-modality tools. Furthermore, alternative data streams often exhibit a high noise-to-signal ratio (due to spam and bot manipulation) and asynchronous data latency. This research implements a privacy-preserving, multimodal deep learning framework that seamlessly fuses heterogeneous alternative data sources—such as **FinBERT textual sentiment/news**, **MSCI ESG disclosures**, **socio-political/spatial indices**, and **traditional time-series financial prices**—into a unified wealth management platform backed by **Federated Learning (Flower)** and **Differential Privacy (Opacus)**.

* **Author:** Sasitha Disanayaka (LMU ID: 25024566)
* **Supervisor:** Dr. Kaneeka Vidanage
* **Institution:** London Metropolitan University / NEXT Campus (MSc Artificial Intelligence)

---

## 🔍 Research Questions (RQ1 – RQ4)
* **RQ1 (Data Fusion):** How can multimodal alternative data streams be effectively fused to generate holistic and personalised wealth management insights for retail investors?
* **RQ2 (AI Architecture):** What architectural configurations of cross-modal attention and temporal fusion generate optimal predictive performance for personalised portfolio recommendations?
* **RQ3 (Privacy & Compliance):** How can federated learning and differential privacy mechanisms ($\epsilon = 3.0, \delta = 10^{-5}$) be integrated to ensure robust data privacy while maintaining recommendation quality?
* **RQ4 (Evaluation & XAI):** To what extent does the multimodal AI framework outperform single-modality baselines on standard financial performance metrics (Sharpe Ratio, Maximum Drawdown, Sortino Ratio), and how do SHAP explanations build user trust?

---

## 🏛️ 5-Layer System Architecture
1. **Layer 1 — Data Ingestion & Alignment (`src/layer1_ingestion/`):** Aggregates S&P 500 OHLCV prices, Reuters/FinBERT sentiment, MSCI ESG ratings, spatial indicators, and SDV CTGAN synthetic retail client profiles under canonical ISO 8601 (`YYYY-MM-DD`) UTC timestamps and Apache Parquet columnar storage.
2. **Layer 2 — Modality Encoders (`src/layer2_encoders/`):** Utilizes Temporal Fusion Transformers (TFT) for time-series prices, FinBERT for text sentiment, and 1D-CNNs for spatial/ESG tabular indices.
3. **Layer 3 — Multimodal Fusion (`src/layer3_fusion/`):** Implements gated Cross-Modal Multi-Head Attention to filter out social media noise and align heterogeneous modalities into a unified contextual embedding.
4. **Layer 4 — Recommendation & Explainability (`src/layer4_rl_xai/`):** Deploys a Deep Deterministic Policy Gradient (DDPG) reinforcement learning portfolio agent with transaction cost penalties and `shap`/`lime` explainability overlays displayed via Streamlit.
5. **Layer 5 — Privacy & Compliance (`src/layer5_fl_dp/`):** Configures Flower (`flwr`) federated learning across 5 simulated retail client nodes integrated with Opacus differential privacy ($(\epsilon, \delta)$ accounting).

---

## 📁 Repository Directory Structure
```text
msc_cs7p01nm_poc/
├── data/
│   ├── raw/                 # Raw downloaded API data feeds
│   ├── processed/           # Processed preview CSV exports
│   ├── parquet_cache/       # High-speed compressed Apache Parquet files
│   ├── synthetic_sdv/       # SDV CTGAN synthetic retail client profiles
│   └── DATA_CATALOGUE.md    # Complete dataset inventory and descriptions
├── src/
│   ├── layer1_ingestion/    # Data pull, ISO 8601 alignment, PyTorch Dataset
│   ├── layer2_encoders/     # TFT, FinBERT projection, and 1D-CNN encoders
│   ├── layer3_fusion/       # Gated Cross-Modal Multi-Head Attention module
│   ├── layer4_rl_xai/       # Gymnasium RL portfolio environment & SHAP XAI
│   ├── layer5_fl_dp/        # Flower federated server/client & Opacus DP engine
│   └── dashboard/           # Interactive Streamlit evaluation interface
├── notebooks/               # Exploratory Data Analysis & baseline experiments
├── thesis/
│   ├── tables/              # Exported CSV performance & ablation tables
│   └── figures/             # 300-DPI publication-grade EDA and evaluation charts
├── Dockerfile               # Python 3.11 container configuration
├── docker-compose.yml       # Local orchestration for dev and Streamlit dashboard
├── requirements.txt         # Locked Python dependency versions
└── README.md                # Project documentation and reproduction guide