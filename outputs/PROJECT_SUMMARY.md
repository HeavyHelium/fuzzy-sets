# Project Summary: Fuzzy Membership Functions from K-means Clustering

## Project Topic: C
**"Develop a software module that applies K-means clustering to a database. Using the discovered clusters and their corresponding cluster centers, automatically define membership functions for each feature of the data. Visualize the resulting membership functions. Illustrate the module's operation by applying it to a well-known dataset."**

---

## 1. Application of Fuzzy Logic for Pattern Extraction

This project demonstrates how fuzzy logic can be applied to extract meaningful patterns from a database. By combining **K-means clustering** with **automatic fuzzy membership function construction**, we transform crisp cluster boundaries into soft, gradual membership degrees that better represent the inherent uncertainty in real-world data.

### Key Insight
Traditional K-means assigns each data point to exactly one cluster (hard assignment). Our approach extends this by constructing overlapping membership functions, allowing a wine sample to have partial membership in multiple fuzzy sets (e.g., "somewhat high alcohol" with μ=0.7 AND "medium alcohol" with μ=0.3).

---

## 2. Dataset Description

**Wine Quality (Red)** — UCI Machine Learning Repository

| Property | Value |
|----------|-------|
| Samples | 1,599 (≥1000 ✓) |
| Features | 11 (≥5 ✓) |
| Source | Physicochemical tests on Portuguese red wines |

### Features
1. Fixed acidity (g/L tartaric acid)
2. Volatile acidity (g/L acetic acid)
3. Citric acid (g/L)
4. Residual sugar (g/L)
5. Chlorides (g/L sodium chloride)
6. Free sulfur dioxide (mg/L)
7. Total sulfur dioxide (mg/L)
8. Density (g/cm³)
9. pH
10. Sulphates (g/L potassium sulphate)
11. Alcohol (% vol)

Quality score (3-8) is included for external validation but NOT used in clustering.

---

## 3. Data Transformation Pipeline

### Stage 1: Raw Data Loading
- Input: CSV file (semicolon-separated)
- Output: NumPy array `X_raw` (1599 × 11) in original units

### Stage 2: Missing Value Handling

| Step | Action | Code |
|------|--------|------|
| 1. Detect | Count NaN values | `np.isnan(X_raw).sum()` |
| 2. Compute | Column means (ignoring NaN) | `np.nanmean(X_raw, axis=0)` |
| 3. Impute | Replace NaN with column mean | `X_raw[mask, j] = col_means[j]` |

**Why mean imputation?**
- Simple and computationally fast
- Preserves sample size (no row dropping)
- K-means requires complete data (no NaN allowed in distance calculations)
- Appropriate when missing data is minimal and random (MCAR)

**Alternative strategies (not implemented):**
- Median imputation — robust to outliers
- KNN imputation — uses similar samples
- Drop rows — if very few missing values

**Wine Quality dataset:** Has **0 missing values** — imputation not triggered, but the check ensures robustness for other datasets.

### Stage 3: Z-Score Standardization
```
X_std = (X_raw - μ) / σ
```

| Feature | Original Range | After Standardization |
|---------|----------------|----------------------|
| fixed acidity | 4.6 – 15.9 g/L | μ≈0, σ=1 |
| density | 0.990 – 1.004 g/cm³ | μ≈0, σ=1 |
| alcohol | 8.4 – 14.9 % | μ≈0, σ=1 |

**Why standardize?** Features have vastly different scales. Without standardization, high-variance features dominate Euclidean distance.

### Stage 4: K-means Clustering
- Performed in **standardized space** for equal feature contribution
- Output: cluster labels and centers in standardized units

### Stage 5: Inverse Transform
```
centers_raw = centers_std × σ + μ
```
Convert centers back to original units for interpretation.

### Stage 6: Membership Function Construction
- Use `centers_raw` (original units) for interpretable MFs
- "Alcohol = 11.5%" → μ = 0.8 in "High" set ✓

---

## 4. Algorithm Implementation

### 4.1 K-means Clustering (from scratch)

**File:** `src/kmeans.py`

- **Initialization:** k-means++ for stable convergence
- **Distance metric:** Euclidean distance
- **Convergence criteria:** Max center shift < tolerance (1e-4) OR max iterations (300)
- **Multiple restarts:** Best of n_init=10 runs (lowest inertia)

### 4.2 Membership Function Types

**File:** `src/fuzzy_mf.py`

Three membership function types are implemented:

| Type | Parameters | Shape | Formula |
|------|------------|-------|---------|
| **Triangular** | (left, peak, right) | /\\ | Linear slopes |
| **Gaussian** | (mean, sigma) | Bell | exp(-((x-μ)²)/(2σ²)) |
| **Trapezoidal** | (a, b, c, d) | /‾\\ | Flat top region |

### MF Construction Algorithm

For each feature j:
1. Extract j-th coordinate of each center: {c₁ⱼ, c₂ⱼ, ..., cₖⱼ}
2. Sort centers: s₁ < s₂ < ... < sₖ
3. Compute midpoints: bᵢ = (sᵢ + sᵢ₊₁) / 2
4. Define k MFs with boundaries at midpoints

---

## 5. Cluster Selection: Multi-Criteria Analysis

A sweep over k ∈ {3, 4, 5, 6, 7, 8} was performed with **four clustering metrics**:

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---------|------------|----------------|-------------------|
| 3 | 12,630 | 0.189 | 1.77 | **313.3** |
| **4** | 11,294 | **0.205** | 1.51 | 296.3 |
| 5 | 10,155 | 0.190 | 1.46 | 291.7 |
| **6** | 9,361 | 0.194 | **1.40** | 280.0 |
| 7 | 8,645 | 0.193 | 1.40 | 274.5 |
| 8 | 8,305 | 0.151 | 1.54 | 254.1 |

### Metric Interpretation

| Metric | Best k | Optimization | Description |
|--------|--------|--------------|-------------|
| Silhouette | k=4 | Higher ↑ | Cluster separation quality |
| Davies-Bouldin | k=6 | Lower ↓ | Within/between cluster ratio |
| Calinski-Harabasz | k=3 | Higher ↑ | Variance ratio criterion |
| Elbow | k=4-5 | — | Diminishing returns point |

**Selection rationale:** Metrics disagree (common in real data). **k=5** provides a good balance between cluster quality and interpretability.

---

## 6. Results and Analysis

### 6.1 Cluster Characteristics (k=5)

| Cluster | Size | Mean Quality | Key Characteristics |
|---------|------|--------------|---------------------|
| 0 | 28 | 5.36 | Outliers: very high chlorides (0.36), high sulphates (1.28) |
| 1 | 335 | 5.33 | High SO₂ (total: 91.2 mg/L), moderate alcohol (9.8%) |
| 2 | 312 | **6.15** | **Best quality:** High alcohol (11.75%), low volatile acidity (0.45) |
| 3 | 373 | 5.95 | High fixed acidity (10.6), high citric acid (0.49), good alcohol (10.6%) |
| 4 | 551 | 5.33 | Largest cluster: Low citric acid (0.11), high volatile acidity (0.65) |

### 6.2 Membership Function Comparison

Example: **Alcohol Content** with different MF types

| Type | Set 5 (High Alcohol) | Characteristics |
|------|---------------------|-----------------|
| Triangular | peak=11.75%, range 11.2–14.9 | Sharp boundaries |
| Gaussian | mean=11.75%, σ=0.28 | Smooth, infinite tails |
| Trapezoidal | flat 11.6–11.9%, slopes to 11.2 and 14.9 | Definite "high" region |

### 6.3 External Validation with Quality

Cluster 2 (highest mean quality = 6.15) is characterized by:
- **Higher alcohol** (cluster peak at 11.75%)
- **Lower volatile acidity** (unpleasant vinegar taste)
- **Higher pH** (less acidic, smoother taste)

This aligns with wine quality research showing that alcohol content and low volatile acidity are positive quality indicators.

---

## 7. Visualizations

### Data Exploration Plots

| Plot Type | Files | Description |
|-----------|-------|-------------|
| **Feature Distributions** | `feature_distributions.png/pdf` | Histograms with mean/median for all 11 features |
| **Feature Box Plots** | `feature_boxplots.png/pdf` | Standardized box plots to compare spreads/outliers |
| **Summary Statistics** | `feature_summary_stats.png/pdf` | 2×2 panel: mean, std, min, max per feature |
| **Correlation Matrix** | `feature_correlation.png/pdf` | Pearson correlations between all feature pairs |

### Cluster Visualization Plots

| Plot Type | Files | Description |
|-----------|-------|-------------|
| **PCA Projection** | `clusters_pca.png/pdf` | 2D scatter plot (PC1=28%, PC2=18% variance) |
| **Pairwise Scatter** | `clusters_pairwise.png/pdf` | Key feature pairs colored by cluster |
| **Cluster Profiles** | `cluster_profiles_radar.png/pdf` | Radar chart comparing cluster fingerprints |
| **Centers Heatmap** | `cluster_centers_heatmap.png/pdf` | Heatmap of cluster center values |

### Membership Function Plots

| Plot Type | Files | Description |
|-----------|-------|-------------|
| **Cluster Metrics Panel** | `cluster_metrics_panel.png/pdf` | 2×2 grid comparing all 4 k-selection metrics |
| **All Features per MF Type** | `all_features_*.png/pdf` | 3×4 grid of all 11 features per MF type |
| **MF Type Comparison** | `mf_panel_*.png/pdf` | Side-by-side comparison of 3 MF types |
| **Individual Features** | `membership_*.png/pdf` | Detailed view with histogram overlay |
| **Quality by Cluster** | `quality_by_cluster.png/pdf` | External validation with wine quality |

### Data Artifacts (`outputs/`)

| File | Description |
|------|-------------|
| `feature_stats.csv` | Feature statistics (n_missing, mean, std, min, max) |
| `centers_raw.csv` | Cluster centers in original units (k×d) |
| `centers_std.csv` | Cluster centers in standardized space |
| `labels.npy` | Cluster assignment for each sample |
| `kmeans_metrics_by_k.csv` | All 4 metrics for k sweep |
| `membership_params_*.json` | MF parameters for each type (triangular, gaussian, trapezoidal) |
| `cluster_quality_summary.csv` | Quality statistics per cluster |
| `run_config.json` | All hyperparameters for reproducibility |

---

## 8. Reproducibility

All experiments are fully reproducible:

```bash
# Activate environment and run
source .venv/bin/activate

# Default (triangular, k=5)
python main.py

# Gaussian MFs
python main.py --mf-type gaussian

# Compare all MF types
python main.py --compare-mf

# Different k
python main.py --k 4
```

**Environment:**
- Python ≥ 3.12
- numpy, pandas, matplotlib, seaborn

---

## 9. Conclusion

This project successfully demonstrates **Topic C** by:

1. ✅ Implementing K-means clustering from scratch (with k-means++ initialization)
2. ✅ Implementing **three types** of membership functions (triangular, Gaussian, trapezoidal)
3. ✅ Performing **multi-criteria cluster selection** (Silhouette, Davies-Bouldin, Calinski-Harabasz)
4. ✅ Visualizing all membership functions with **panel plots** (11 features × 5 fuzzy sets × 3 types)
5. ✅ Applying to a well-known dataset (UCI Wine Quality, 1599×11)
6. ✅ Providing comprehensive **data transformation documentation**
7. ✅ Generating **exploratory data analysis** visualizations (distributions, correlations, box plots)
8. ✅ Creating **cluster visualizations** (PCA projection, radar profiles, pairwise scatter)

### Key Contributions Beyond Topic C

| Feature | Topic A Coverage |
|---------|------------------|
| Multi-criteria k selection | Comparative analysis of 4 clustering metrics |
| Cluster visualizations | PCA, radar charts, pairwise scatter plots |
| Panel visualizations | Side-by-side comparison of methods |
| Data exploration | Distributions, correlations, outlier detection |

### Summary Statistics

| Category | Count |
|----------|-------|
| Total visualizations | ~40 plots (PNG + PDF) |
| Data artifacts | 10 CSV/JSON files |
| MF types implemented | 3 (triangular, Gaussian, trapezoidal) |
| Cluster metrics | 4 (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz) |
| Features analyzed | 11 |
| Samples processed | 1,599 |

The fuzzy membership functions provide a natural way to interpret cluster boundaries as gradual linguistic categories, enabling soft classification like "this wine has HIGH alcohol content with membership degree 0.85."

---

## References

1. Cortez, P., et al. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553.
2. UCI Machine Learning Repository: Wine Quality Dataset
3. Bezdek, J.C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms. Springer.
