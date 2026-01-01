# Fuzzy Membership Functions from K-means Clustering

A complete implementation of K-means clustering with automatic fuzzy membership function construction on the **Wine Quality** dataset.

## Project Overview

This project demonstrates:
- **K-means clustering** implementation from scratch (with k-means++ initialization)
- **Automatic construction** of membership functions (triangular, Gaussian, trapezoidal)
- **Multi-criteria cluster selection** (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Visualization** of membership functions with panel plots
- **External validation** using wine quality scores

## Dataset

**Wine Quality (Red)** from UCI Machine Learning Repository:
- 1,599 samples
- 11 physicochemical features (fixed acidity, volatile acidity, citric acid, etc.)
- Quality score (3-8) used for validation only, not clustering

## Installation

```bash
# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the complete pipeline:

```bash
python main.py
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 5 | Number of clusters |
| `--seed` | 42 | Random seed for reproducibility |
| `--n-init` | 10 | Number of K-means restarts |
| `--max-iter` | 300 | Maximum iterations |
| `--skip-sweep` | False | Skip elbow analysis |
| `--mf-type` | triangular | MF type: `triangular`, `gaussian`, `trapezoidal` |
| `--compare-mf` | False | Generate comparison of all MF types |

### Examples

```bash
# Default (triangular MFs, k=5)
python main.py

# Gaussian membership functions
python main.py --mf-type gaussian

# Compare all MF types with panel plots
python main.py --compare-mf

# Different k value
python main.py --k 4 --seed 123
```

## Project Structure

```
fuzzy-sets/
├── src/
│   ├── data.py        # Data loading, preprocessing, scaling
│   ├── kmeans.py      # K-means implementation from scratch
│   ├── fuzzy_mf.py    # Fuzzy membership function construction
│   └── viz.py         # Visualization utilities
├── main.py            # End-to-end pipeline
├── data/              # Dataset (auto-downloaded)
├── outputs/           # Clustering results, MF parameters
├── figures/           # All plots (PNG + PDF)
└── README.md
```

## Data Transformation Pipeline

### Stage 1: Raw Data Loading

```
CSV → DataFrame → NumPy array (float64)
```

- Input: `winequality-red.csv` (semicolon-separated)
- Extract 11 numeric feature columns
- Separate quality column for validation (not used in clustering)

### Stage 2: Missing Value Handling

```
X_raw → Check NaN → Impute with column mean if needed
```

| Step | Action |
|------|--------|
| 1. Detect | Count NaN values: `np.isnan(X_raw).sum()` |
| 2. Compute | Column means (ignoring NaN): `np.nanmean(X_raw, axis=0)` |
| 3. Impute | Replace NaN with column mean |

**Why mean imputation?**
- Simple and fast
- Preserves sample size (no row dropping)
- K-means requires complete data (no NaN allowed)

**Wine Quality dataset:** Has **0 missing values** — no imputation needed.

### Stage 3: Z-Score Standardization

```
X_std = (X_raw - μ) / σ
```

| Feature | Original Scale | After Standardization |
|---------|----------------|----------------------|
| fixed acidity | 4.6 – 15.9 g/L | μ≈0, σ=1 |
| density | 0.990 – 1.004 g/cm³ | μ≈0, σ=1 |
| alcohol | 8.4 – 14.9 % | μ≈0, σ=1 |

**Why standardize?** Features have vastly different scales. K-means uses Euclidean distance — without standardization, high-variance features dominate.

### Stage 4: K-means Clustering

```
X_std (standardized) → K-means → labels, centers_std
```

- Clustering performed in **standardized space**
- Equal contribution from all features

### Stage 5: Inverse Transform

```
centers_raw = centers_std × σ + μ
```

Convert cluster centers back to **original units** for human interpretation.

### Stage 6: Membership Function Construction

```
centers_raw → Sort per feature → MF parameters (in original units)
```

MFs are defined in original units so they're interpretable:
- "Alcohol = 11.5%" → μ = 0.8 in "High" set ✓
- Not "Alcohol = 1.02 std" (meaningless)

### Pipeline Summary

```
┌─────────────────┐
│   Raw CSV       │  Original units (g/L, %, etc.)
└────────┬────────┘
         ▼
┌─────────────────┐
│  X_raw (N×11)   │  Missing value check
└────────┬────────┘
         ▼
┌─────────────────┐
│  Standardize    │  X_std = (X - μ) / σ
└────────┬────────┘
         ▼
┌─────────────────┐
│   K-means       │  Clustering in standardized space
└────────┬────────┘
         ▼
┌─────────────────┐
│ Inverse Transform│  centers_raw = centers_std × σ + μ
└────────┬────────┘
         ▼
┌─────────────────┐
│  Fuzzy MFs      │  Built from centers_raw (original units)
└─────────────────┘
```

## Methodology

### 1. K-means Clustering

- **Initialization**: k-means++ for stable convergence
- **Distance metric**: Euclidean distance
- **Convergence**: Max center shift < tolerance or max iterations reached
- **Multiple restarts**: Best of n_init runs (lowest inertia)

### 2. Cluster Selection Metrics

| Metric | Optimization | Description |
|--------|--------------|-------------|
| Inertia (SSE) | Lower ↓ | Sum of squared distances to centers |
| Silhouette | Higher ↑ | Cluster separation quality [-1, 1] |
| Davies-Bouldin | Lower ↓ | Ratio of within/between cluster distances |
| Calinski-Harabasz | Higher ↑ | Variance ratio criterion |

### 3. Membership Function Types

| Type | Parameters | Shape | Best For |
|------|------------|-------|----------|
| **Triangular** | (left, peak, right) | Sharp peak | Simple, interpretable |
| **Gaussian** | (mean, sigma) | Bell curve | Smooth, continuous data |
| **Trapezoidal** | (a, b, c, d) | Flat top | "Definitely in category" |

### 4. MF Construction from Centers

For each feature j:
1. Extract j-th coordinate of each cluster center
2. Sort centers: s₁ < s₂ < ... < sₖ
3. Compute midpoints: bᵢ = (sᵢ + sᵢ₊₁) / 2
4. Define k MFs with boundaries at midpoints

## Outputs

### Data Files (`outputs/`)

| File | Description |
|------|-------------|
| `feature_stats.csv` | Feature statistics (n_missing, mean, std, min, max) |
| `centers_raw.csv` | Cluster centers in original units |
| `centers_std.csv` | Cluster centers in standardized space |
| `labels.npy` | Cluster assignment for each sample |
| `kmeans_metrics_by_k.csv` | All clustering metrics for k sweep |
| `membership_params_*.json` | MF parameters for each type |
| `cluster_quality_summary.csv` | Quality analysis per cluster |
| `run_config.json` | All hyperparameters for reproducibility |

### Figures (`figures/`)

**Data Exploration:**
| File | Description |
|------|-------------|
| `feature_distributions.png/pdf` | Histograms with mean/median |
| `feature_boxplots.png/pdf` | Standardized box plots |
| `feature_summary_stats.png/pdf` | Mean, std, min, max panels |
| `feature_correlation.png/pdf` | Correlation heatmap |

**Cluster Analysis:**
| File | Description |
|------|-------------|
| `clusters_pca.png/pdf` | PCA 2D projection of clusters |
| `clusters_pairwise.png/pdf` | Pairwise scatter plots |
| `cluster_profiles_radar.png/pdf` | Radar chart of cluster profiles |
| `cluster_centers_heatmap.png/pdf` | Cluster center values |
| `cluster_metrics_panel.png/pdf` | 2×2 panel of k-selection metrics |
| `quality_by_cluster.png/pdf` | Wine quality by cluster |

**Membership Functions:**
| File | Description |
|------|-------------|
| `all_features_*.png/pdf` | All 11 features for each MF type |
| `mf_panel_*.png/pdf` | 3-way MF type comparison |
| `membership_*.png/pdf` | Individual feature MF plots |
| `all_figures.pdf` | Combined PDF with all visualizations |

## Reproducibility

- Fixed random seed (default: 42)
- All hyperparameters logged in `outputs/run_config.json`
- Python version compatibility: ≥ 3.12

## License

MIT License
