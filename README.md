# 🕵️ Social Media Manipulation Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Clustering-purple?style=for-the-badge)
![IsolationForest](https://img.shields.io/badge/Isolation%20Forest-Anomaly%20Detection-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

**An end-to-end machine learning pipeline for detecting coordinated inauthentic behavior, bot accounts, and information manipulation on social media platforms.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Pipeline & Methodology](#-pipeline--methodology)
  - [Task 1 — Bot & Anomaly Detection](#task-1--bot--anomaly-detection)
  - [Task 2 — Feature Engineering](#task-2--feature-engineering)
  - [Task 3 — Influence Score Estimation](#task-3--influence-score-estimation)
  - [Task 4 — Manipulation Mapping](#task-4--manipulation-mapping)
  - [Task 5 — Jury Inference](#task-5--jury-inference)
- [Repository Structure](#-repository-structure)
- [Data](#-data)
- [Models](#-models)
- [Outputs & Results](#-outputs--results)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Technical Deep Dive](#-technical-deep-dive)
- [Key Findings](#-key-findings)

---

## 🔍 Project Overview

This project implements a **multi-task anomaly detection and manipulation scoring system** for large-scale social media datasets. It combines unsupervised learning (HDBSCAN clustering, Isolation Forest), feature engineering over author-level and post-level behavioral signals, and an inference pipeline capable of scoring new accounts and posts in production.

The system was designed to answer a critical question:

> *"Given millions of social media posts and author metadata, which accounts are behaving inauthentically — and how do we quantify, cluster, and explain that behavior?"*

The pipeline produces per-author manipulation scores, per-post anomaly scores, interpretable cluster profiles, and jury-ready predictions — all from raw data with no ground-truth labels.

---

## 🎯 Problem Statement

Modern disinformation campaigns on social media platforms are characterized by:

- **Coordinated inauthentic behavior** — networks of accounts posting similar content at similar times
- **Bot-like activity patterns** — superhuman posting frequencies, unnatural engagement ratios
- **Keyword & narrative injection** — systematic amplification of specific terms across languages/platforms
- **Cross-platform manipulation** — same actors operating simultaneously on multiple networks

Traditional rule-based systems fail to scale and adapt. This project uses a **fully unsupervised ML approach** that learns the statistical fingerprint of normal vs. anomalous behavior directly from data.

---

## 🏗️ System Architecture

```
Raw Data (posts + author metadata)
           │
           ▼
┌─────────────────────────┐
│   Feature Engineering   │  ← src/features.py
│  (Post-level + Author)  │     notebooks/02_features.ipynb
└────────────┬────────────┘
             │
     ┌───────┴───────┐
     │               │
     ▼               ▼
┌─────────┐    ┌──────────────┐
│ HDBSCAN │    │  Isolation   │   ← src/hdbscan_cluster.py
│Clustering│   │   Forest     │      src/model.py
│ (Task 1) │   │ (Task 1 & 3) │
└────┬─────┘   └──────┬───────┘
     │                │
     ▼                ▼
┌──────────────────────────┐
│   Manipulation Mapping   │  ← notebooks/04_manipulation_map.ipynb
│  (Cluster Annotation +   │
│   Language/Platform Map) │
└────────────┬─────────────┘
             │
             ▼
┌─────────────────────────┐
│    Inference Engine     │  ← src/inference.py
│  (Score New Accounts)   │     notebooks/05_inference.ipynb
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Jury Predictions    │  ← outputs/jury_predictions.csv
│  (Final Scored Output)  │
└─────────────────────────┘
```

---

## 🔬 Pipeline & Methodology

### Task 1 — Bot & Anomaly Detection

**Goal:** Identify individual accounts showing statistically anomalous behavioral patterns.

**Approach:**
1. Extract numerical behavioral features at the **author level** (posting frequency, engagement rate, follower/following ratio, activity timestamps, etc.)
2. Apply **MinMax scaling** (`models/task_1/minmax_scaler.pkl`) and **standard scaling** (`models/task_1/scaler.pkl`) for normalization
3. Train an **Isolation Forest** (`models/task_1/isolation_forest.pkl`) — an ensemble of random trees that isolates anomalies by how few splits are required to separate them from the bulk of data
   - Anomaly score ∈ [-1, 1]: values close to -1 indicate high anomaly
4. Produce `outputs/author_scores.csv` — a scored list of all authors with their anomaly signal

**Why Isolation Forest?**
- No assumption about the shape of the normal distribution
- Scales to millions of records efficiently (O(n log n))
- Handles high-dimensional feature spaces without distance metrics
- Unsupervised — no labeled "bot" data required

---

### Task 2 — Feature Engineering

**Goal:** Build rich, informative feature sets from raw post and author data.

**Key Feature Categories:**

| Category | Features |
|---|---|
| **Temporal** | Posts per hour/day, posting interval variance, burst detection |
| **Content** | Text length, keyword density, duplicate ratio, hashtag count |
| **Social Graph** | Follower/following ratio, engagement rate, reply/retweet ratio |
| **Cross-platform** | Platform activity distribution, multi-platform presence flag |
| **Linguistic** | Language diversity, non-native language usage ratio |
| **Network** | Co-occurrence with flagged accounts, keyword co-sharing |

**Outputs:**
- `data/processed/post_features.parquet` — post-level features (~185 MB, high cardinality)
- `data/processed/author_features.parquet` — author-level aggregated features (~69 MB)
- `data/processed/dup_lookup.parquet` — duplicate content lookup table
- `data/processed/kw_lookup.parquet` — keyword frequency/distribution lookup

---

### Task 3 — Influence Score Estimation

**Goal:** Beyond detecting anomalies, quantify the **degree of influence** each account exerts — separating passive bots from active amplifiers.

**Approach:**
1. Build an **influence-specific feature set** (reach, amplification rate, network centrality proxies)
2. Apply **Isolation Forest** specifically tuned for influence estimation (`models/task_3/inf_isolation_forest.pkl`)
3. Use **clip bounds** (`models/task_3/inf_clip_bounds.npy`) to handle extreme outliers in influence metrics
4. Scale scores to [0, 1] via MinMax scaler (`models/task_3/inf_minmax_scaler.pkl`)

This produces a continuous **influence-manipulation score** rather than a binary flag, enabling ranking and prioritization.

---

### Task 4 — Manipulation Mapping

**Goal:** Cluster detected manipulators into interpretable groups and map them by language, platform, and behavioral profile.

**Approach:**
1. **HDBSCAN Clustering** (`models/task_1/hdbscan.pkl`) — Hierarchical Density-Based Spatial Clustering of Applications with Noise
   - HDBSCAN outperforms K-Means here because manipulation clusters are **non-spherical, variable-density, and noisy**
   - Points that don't fit any cluster are labeled as "noise" (cluster = -1), which is valuable information
   - HDBSCAN scaler: `models/task_1/hdbscan_scaler.pkl`
2. Cluster annotation: each cluster receives a **behavioral profile label** (`outputs/cluster_summary_annotated.csv`)
3. Generate visualizations:
   - Language distribution map across clusters
   - Platform-language heatmap
   - Score distribution per cluster
   - Noise analysis (what didn't cluster and why)

**Key Outputs:**
- `outputs/post_clusters.parquet` — every post assigned to a cluster
- `outputs/cluster_summary.csv` — raw cluster statistics
- `outputs/cluster_summary_annotated.csv` — human-annotated cluster profiles
- `outputs/top_bots.csv` — highest-confidence flagged accounts

---

### Task 5 — Jury Inference

**Goal:** Apply the trained pipeline to a held-out jury input and produce final scored predictions.

**Approach (`src/inference.py`):**
1. Load `data/processed/jury_input_example.csv` (new, unseen accounts)
2. Run the full feature extraction pipeline
3. Apply all saved scalers and models in order
4. Produce `outputs/jury_predictions.csv` — final predictions with scores, cluster assignments, and manipulation category labels

The inference module is **fully production-ready** — new data can be scored without retraining.

---

## 📁 Repository Structure

```
.
├── data/
│   └── processed/
│       ├── author_features.parquet   # Author-level behavioral features
│       ├── post_features.parquet     # Post-level features
│       ├── dup_lookup.parquet        # Duplicate content lookup
│       ├── kw_lookup.parquet         # Keyword distribution lookup
│       ├── jury_input_example.csv    # Example input for inference
│       └── otocsv.py                 # CSV utility script
│
├── models/
│   ├── task_1/
│   │   ├── hdbscan.pkl               # Trained HDBSCAN clustering model
│   │   ├── hdbscan_scaler.pkl        # Scaler for HDBSCAN input
│   │   ├── isolation_forest.pkl      # Anomaly detection model
│   │   ├── minmax_scaler.pkl         # MinMax normalization
│   │   └── scaler.pkl                # Standard scaler
│   └── task_3/
│       ├── inf_isolation_forest.pkl  # Influence-specific anomaly model
│       ├── inf_minmax_scaler.pkl     # Scaler for influence features
│       ├── inf_scaler.pkl            # Standard scaler for influence
│       └── inf_clip_bounds.npy       # Outlier clip bounds array
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_features.ipynb             # Feature engineering pipeline
│   ├── 03_if_scorer.ipynb            # Isolation Forest training & scoring
│   ├── 04_manipulation_map.ipynb     # HDBSCAN clustering & visualization
│   └── 05_inference.ipynb            # End-to-end inference pipeline
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Data loading utilities
│   ├── features.py                   # Feature engineering functions
│   ├── hdbscan_cluster.py            # HDBSCAN training & prediction
│   ├── inference.py                  # Production inference engine
│   └── model.py                      # Model training utilities
│
├── outputs/
│   ├── author_scores.csv             # All authors with anomaly scores
│   ├── top_bots.csv                  # Highest-confidence flagged accounts
│   ├── jury_predictions.csv          # Final jury output
│   ├── post_clusters.parquet         # Posts with cluster assignments
│   ├── post_scores.parquet           # Per-post anomaly scores
│   ├── cluster_summary.csv           # Raw cluster statistics
│   ├── cluster_summary_annotated.csv # Annotated cluster profiles
│   └── fig_*.png                     # Visualization outputs (8 figures)
│
├── proje_snapshot/
│   └── snapshot.txt                  # Project state snapshot
│
├── ARCHITECTURE.md                   # Detailed architecture documentation
├── CLAUDE.md                         # AI assistant context
├── README.md                         # This file
├── pyproject.toml                    # Project dependencies
└── uv.lock                           # Locked dependency versions
```

---

## 📊 Data

### Input Data

| File | Size | Description |
|---|---|---|
| `author_features.parquet` | ~69 MB | Per-author behavioral aggregate features |
| `post_features.parquet` | ~185 MB | Per-post content and engagement features |
| `dup_lookup.parquet` | ~646 KB | Duplicate/near-duplicate content index |
| `kw_lookup.parquet` | ~1.2 MB | Keyword frequency distribution table |
| `jury_input_example.csv` | ~56 KB | Example inference input |

> **Note:** All large files are tracked via **Git LFS** (`git-lfs`). Run `git lfs pull` after cloning to download binary data.

### Data Schema (Author Features — Key Fields)

```
author_id           — Unique author identifier
platform            — Source platform
language            — Detected language
posts_total         — Total post count
posts_per_day       — Average daily posting rate
engagement_rate     — (likes + shares + comments) / followers
follower_ratio      — followers / following
burst_score         — Measure of unnatural activity bursts
dup_content_ratio   — Proportion of duplicate/near-duplicate posts
kw_injection_score  — Keyword manipulation signal
```

---

## 🤖 Models

### Isolation Forest (Task 1 & 3)

```python
# Conceptual configuration
IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.05,   # ~5% assumed anomaly rate
    random_state=42
)
```

- **Purpose:** Unsupervised anomaly detection on author behavioral features
- **Output:** Anomaly score per author/post, decision boundary at 0.0

### HDBSCAN (Task 1 — Clustering)

```python
# Conceptual configuration
HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom'
)
```

- **Purpose:** Identify distinct behavioral archetypes among flagged accounts
- **Output:** Cluster labels + soft cluster probabilities

### Preprocessing Chain

```
Raw Features
    → StandardScaler (zero mean, unit variance)
    → MinMaxScaler (bounded to [0, 1])
    → Outlier clipping (Task 3 only)
    → Model Input
```

---

## 📈 Outputs & Results

### Generated Figures

| Figure | Description |
|---|---|
| `fig_manipulation_score_dist.png` | Distribution of manipulation scores across all authors |
| `fig_manipulation_category_pie.png` | Breakdown of manipulation categories |
| `fig_manipulation_feature_profile.png` | Feature importance profile per cluster |
| `fig_manipulation_language_map.png` | Geographic/linguistic distribution of flagged accounts |
| `fig_manipulation_platform_map.png` | Platform distribution of manipulation activity |
| `fig_manipulation_lang_platform_heatmap.png` | Language × Platform co-occurrence heatmap |
| `fig_manipulation_noise_analysis.png` | Analysis of HDBSCAN noise points |
| `inference_advanced_summary.png` | Full inference pipeline summary dashboard |
| `inference_realistic_summary.png` | Realistic-scenario prediction summary |
| `inference_correlations.png` | Feature correlation matrix for inference inputs |
| `inference_signals.png` | Key detection signal visualization |
| `model_report_v2.png` | Model performance and calibration report |

### Score Outputs

- **`author_scores.csv`** — Every author in the dataset scored; sortable by manipulation probability
- **`top_bots.csv`** — Pre-filtered highest-confidence anomalous accounts
- **`jury_predictions.csv`** — Final output for external evaluation/jury review
- **`post_scores.parquet`** — Post-level granularity scores for deep-dive analysis

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)
- Git LFS (`git lfs install`)

### 1. Clone the Repository

```bash
git clone https://github.com/Cullinan-coder/<repo-name>.git
cd <repo-name>

# Pull LFS files (models, parquet data)
git lfs pull
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Verify Installation

```bash
python -c "from src.inference import run_inference; print('✓ Setup complete')"
```

---

## 🚀 Usage

### Run Full Inference on New Data

```python
from src.inference import run_inference

# Score a new dataset
predictions = run_inference(
    input_path="data/processed/jury_input_example.csv",
    output_path="outputs/my_predictions.csv"
)
```

### Score Authors Directly

```python
from src.model import load_models, score_authors
import pandas as pd

# Load pre-trained models
models = load_models("models/task_1/")

# Load features
author_features = pd.read_parquet("data/processed/author_features.parquet")

# Score
scores = score_authors(author_features, models)
print(scores[scores['anomaly_score'] < -0.1].head(20))
```

### Run Clustering on New Features

```python
from src.hdbscan_cluster import cluster_posts
import pandas as pd

post_features = pd.read_parquet("data/processed/post_features.parquet")
clusters = cluster_posts(post_features, model_path="models/task_1/hdbscan.pkl")
```

### Notebooks (Recommended for Exploration)

```bash
# Start Jupyter
jupyter lab

# Run in order:
# 01_eda.ipynb          → Understand the data
# 02_features.ipynb     → Reproduce feature engineering
# 03_if_scorer.ipynb    → Train/evaluate Isolation Forest
# 04_manipulation_map.ipynb → Cluster & visualize
# 05_inference.ipynb    → End-to-end scoring
```

---

## 🔧 Technical Deep Dive

### Why Unsupervised Learning?

Ground-truth labels for "bot" or "manipulator" accounts are:
1. **Expensive** — require expert human annotation
2. **Biased** — labeled datasets reflect past known tactics, not novel ones
3. **Platform-specific** — labels from Twitter don't transfer to Telegram

Isolation Forest and HDBSCAN learn the **statistical structure** of normal vs. anomalous behavior without needing labeled examples.

### HDBSCAN vs. K-Means for This Problem

| Property | K-Means | HDBSCAN |
|---|---|---|
| Cluster shape | Spherical only | Arbitrary |
| Outlier handling | Assigns to nearest cluster | Labels as noise (-1) |
| Number of clusters | Must specify k | Discovered automatically |
| Density sensitivity | None | Core feature |
| Scalability | O(nk) | O(n log n) |

Manipulation campaigns form **irregular, dense sub-communities** — HDBSCAN is the natural choice.

### Isolation Forest: How It Works

```
1. Draw a random feature
2. Draw a random split value within feature range
3. Recurse until point is isolated
4. Anomaly score = average path length across all trees
   → Short path = anomaly (easy to isolate = unusual)
   → Long path = normal (hard to isolate = typical)
```

### Parquet Format Rationale

All large intermediate datasets use Apache Parquet:
- **10-50x smaller** than equivalent CSV
- **Columnar storage** — reading only needed features is fast
- **Schema preservation** — types, nullability are stored
- **Pandas/Polars native** — zero-overhead loading

### Git LFS for Binary Files

Models (`.pkl`) and large parquets are stored in Git LFS:
```
models/task_1/hdbscan.pkl         164 MB
models/task_1/isolation_forest.pkl  37 MB
outputs/post_clusters.parquet     202 MB
outputs/post_scores.parquet       158 MB
```
This keeps the repository fast to clone while preserving full model reproducibility.

---

## 🔑 Key Findings

From the analysis visualizations and cluster summaries:

- **Coordinated posting networks** were identified across multiple platforms sharing identical keyword injection patterns
- **Language-platform clustering** revealed distinct regional manipulation campaigns (visible in `fig_manipulation_lang_platform_heatmap.png`)
- **High-influence anomalous accounts** (Task 3) represent a small fraction (~2-5%) of flagged accounts but disproportionate reach
- **HDBSCAN noise points** (~15-25% of anomalous accounts) represent genuinely unique/isolated bad actors not belonging to any coordinated network
- The **feature profile visualization** (`fig_manipulation_feature_profile.png`) shows that `burst_score`, `dup_content_ratio`, and `kw_injection_score` are the strongest discriminators

---

## 📄 License

See `LICENSE` for full terms.

---

## 🙏 Acknowledgments

Built with:
- [scikit-learn](https://scikit-learn.org/) — Isolation Forest implementation
- [hdbscan](https://hdbscan.readthedocs.io/) — Density-based clustering
- [pandas](https://pandas.pydata.org/) + [pyarrow](https://arrow.apache.org/) — Data processing
- [uv](https://github.com/astral-sh/uv) — Dependency management

---

<div align="center">

*Built for detecting coordinated inauthentic behavior at scale.*

</div>
