"""
Visualization utilities for K-means and fuzzy membership functions.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .fuzzy_mf import triangular_mf, gaussian_mf, trapezoidal_mf, evaluate_mf


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Increase all font sizes by 2 points
plt.rcParams.update({
    'font.size': 12,           # base font size (default ~10)
    'axes.titlesize': 14,      # title font size
    'axes.labelsize': 13,      # axis label font size
    'xtick.labelsize': 11,     # x-tick label font size
    'ytick.labelsize': 11,     # y-tick label font size
    'legend.fontsize': 11,     # legend font size
    'figure.titlesize': 16,    # suptitle font size
})


def plot_inertia_vs_k(
    inertia_df: pd.DataFrame,
    output_path: Path = Path("figures/inertia_vs_k.png")
) -> None:
    """Plot inertia (elbow curve) vs number of clusters."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(inertia_df["k"], inertia_df["inertia"], "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=12)
    ax.set_title("K-means Elbow Curve", fontsize=14, fontweight="bold")
    ax.set_xticks(inertia_df["k"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved elbow plot to {output_path}")


def plot_cluster_metrics_panel(
    metrics_df: pd.DataFrame,
    output_path: Path = Path("figures/cluster_metrics_panel.png")
) -> None:
    """Plot all clustering metrics in a 2x2 panel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Inertia (Elbow)
    ax = axes[0, 0]
    ax.plot(metrics_df["k"], metrics_df["inertia"], "o-", linewidth=2, markersize=8, color="#e74c3c")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Inertia (SSE)", fontsize=11)
    ax.set_title("Elbow Curve (lower = tighter clusters)", fontsize=12, fontweight="bold")
    ax.set_xticks(metrics_df["k"])
    ax.grid(True, alpha=0.3)
    
    # Silhouette Score
    ax = axes[0, 1]
    colors = ["#27ae60" if s == metrics_df["silhouette"].max() else "#3498db" for s in metrics_df["silhouette"]]
    ax.bar(metrics_df["k"], metrics_df["silhouette"], color=colors, edgecolor="black")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score (higher = better separation)", fontsize=12, fontweight="bold")
    ax.set_xticks(metrics_df["k"])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Davies-Bouldin Index
    ax = axes[1, 0]
    colors = ["#27ae60" if d == metrics_df["davies_bouldin"].min() else "#9b59b6" for d in metrics_df["davies_bouldin"]]
    ax.bar(metrics_df["k"], metrics_df["davies_bouldin"], color=colors, edgecolor="black")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Davies-Bouldin Index", fontsize=11)
    ax.set_title("Davies-Bouldin Index (lower = better)", fontsize=12, fontweight="bold")
    ax.set_xticks(metrics_df["k"])
    ax.grid(True, alpha=0.3, axis="y")
    
    # Calinski-Harabasz Index
    ax = axes[1, 1]
    colors = ["#27ae60" if c == metrics_df["calinski_harabasz"].max() else "#f39c12" for c in metrics_df["calinski_harabasz"]]
    ax.bar(metrics_df["k"], metrics_df["calinski_harabasz"], color=colors, edgecolor="black")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Calinski-Harabasz Index", fontsize=11)
    ax.set_title("Calinski-Harabasz Index (higher = better)", fontsize=12, fontweight="bold")
    ax.set_xticks(metrics_df["k"])
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Cluster Number Selection: Multi-Criteria Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved cluster metrics panel to {output_path}")


def plot_mf_types_panel(
    feature: str,
    all_mf_params: dict,
    X_raw: np.ndarray,
    feature_idx: int,
    output_dir: Path = Path("figures")
) -> None:
    """
    Plot all MF types for a feature in a single panel.
    
    Args:
        feature: Feature name
        all_mf_params: Dict with keys "triangular", "gaussian", "trapezoidal"
        X_raw: Original data
        feature_idx: Index of feature
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feat_values = X_raw[:, feature_idx]
    feat_min, feat_max = feat_values.min(), feat_values.max()
    padding = (feat_max - feat_min) * 0.05
    x_range = np.linspace(feat_min - padding, feat_max + padding, 500)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    mf_types = ["triangular", "gaussian", "trapezoidal"]
    titles = ["Triangular (sharp)", "Gaussian (smooth)", "Trapezoidal (flat top)"]
    
    for ax, mf_type, title in zip(axes, mf_types, titles):
        params = all_mf_params[mf_type]
        mfs = params[feature]
        k = len(mfs)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
        
        for i, mf in enumerate(mfs):
            mu = evaluate_mf(x_range, mf)
            ax.plot(x_range, mu, linewidth=2, color=colors[i])
            ax.fill_between(x_range, mu, alpha=0.15, color=colors[i])
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("μ", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Membership Function Comparison: {feature}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    safe_name = feature.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"mf_panel_{safe_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved MF panel: {output_path}")


def plot_feature_statistics(
    X_raw: np.ndarray,
    feature_names: list[str],
    output_dir: Path = Path("figures")
) -> None:
    """
    Plot comprehensive statistics for the dataset features.
    Creates a multi-panel figure with distributions, box plots, and summary stats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_features = len(feature_names)
    
    # Figure 1: Feature distributions (histograms)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        values = X_raw[:, idx]
        
        ax.hist(values, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(values.mean(), color="#e74c3c", linestyle="--", linewidth=2, label=f"μ={values.mean():.2f}")
        ax.axvline(np.median(values), color="#27ae60", linestyle=":", linewidth=2, label=f"med={np.median(values):.2f}")
        
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(axis="both", labelsize=8)
    
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Feature Distributions with Mean (red) and Median (green)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "feature_distributions.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved feature distributions to {output_dir}/feature_distributions.png")
    
    # Figure 2: Box plots (normalized for comparison)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Normalize for visualization (z-score)
    X_norm = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    
    bp = ax.boxplot(X_norm, labels=feature_names, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_features))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Standardized Value (z-score)", fontsize=11)
    ax.set_title("Feature Box Plots (Standardized for Comparison)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_boxplots.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "feature_boxplots.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved feature box plots to {output_dir}/feature_boxplots.png")
    
    # Figure 3: Summary statistics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean
    ax = axes[0, 0]
    bars = ax.barh(feature_names, X_raw.mean(axis=0), color="#3498db", edgecolor="black")
    ax.set_xlabel("Mean Value", fontsize=10)
    ax.set_title("Feature Means (Original Scale)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Std
    ax = axes[0, 1]
    bars = ax.barh(feature_names, X_raw.std(axis=0), color="#e74c3c", edgecolor="black")
    ax.set_xlabel("Standard Deviation", fontsize=10)
    ax.set_title("Feature Standard Deviations", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Min
    ax = axes[1, 0]
    bars = ax.barh(feature_names, X_raw.min(axis=0), color="#27ae60", edgecolor="black")
    ax.set_xlabel("Minimum Value", fontsize=10)
    ax.set_title("Feature Minimums", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Max
    ax = axes[1, 1]
    bars = ax.barh(feature_names, X_raw.max(axis=0), color="#9b59b6", edgecolor="black")
    ax.set_xlabel("Maximum Value", fontsize=10)
    ax.set_title("Feature Maximums", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.suptitle("Feature Summary Statistics", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_summary_stats.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "feature_summary_stats.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved feature summary stats to {output_dir}/feature_summary_stats.png")
    
    # Figure 4: Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = np.corrcoef(X_raw.T)
    
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    # Add correlation values
    for i in range(n_features):
        for j in range(n_features):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "feature_correlation.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved feature correlation matrix to {output_dir}/feature_correlation.png")


def plot_all_features_panel(
    membership_params: dict,
    X_raw: np.ndarray,
    feature_names: list[str],
    output_dir: Path = Path("figures")
) -> None:
    """
    Plot all features' membership functions in a single multi-panel figure.
    Creates a grid showing all 11 features for the given MF type.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mf_type = membership_params.get("mf_type", "triangular")
    n_features = len(feature_names)
    
    # Calculate grid size (e.g., 3x4 for 11 features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        mfs = membership_params[feature]
        k = len(mfs)
        
        feat_values = X_raw[:, idx]
        feat_min, feat_max = feat_values.min(), feat_values.max()
        padding = (feat_max - feat_min) * 0.05
        x_range = np.linspace(feat_min - padding, feat_max + padding, 300)
        
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
        
        for i, mf in enumerate(mfs):
            mu = evaluate_mf(x_range, mf)
            ax.plot(x_range, mu, linewidth=1.5, color=colors[i])
            ax.fill_between(x_range, mu, alpha=0.12, color=colors[i])
        
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel("μ", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    type_title = mf_type.capitalize()
    plt.suptitle(f"{type_title} Membership Functions — All Features (k={k})", 
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    output_path = output_dir / f"all_features_{mf_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved all-features panel: {output_path}")


def plot_membership_function(
    feature: str,
    membership_params: dict,
    X_raw: np.ndarray,
    feature_idx: int,
    output_dir: Path = Path("figures"),
    show_histogram: bool = True,
    suffix: str = ""
) -> None:
    """
    Plot membership functions for a single feature.
    
    Args:
        feature: Feature name
        membership_params: Dictionary of MF parameters
        X_raw: Original feature matrix for histogram
        feature_idx: Index of feature in X_raw
        output_dir: Directory to save figure
        show_histogram: Whether to overlay histogram
        suffix: Optional suffix for filename (e.g., "_gaussian")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mfs = membership_params[feature]
    k = len(mfs)
    mf_type = membership_params.get("mf_type", "triangular")
    
    # Feature range
    feat_values = X_raw[:, feature_idx]
    feat_min, feat_max = feat_values.min(), feat_values.max()
    padding = (feat_max - feat_min) * 0.05
    x_range = np.linspace(feat_min - padding, feat_max + padding, 500)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Color palette for membership functions
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    
    # Plot each membership function
    for i, mf in enumerate(mfs):
        mu = evaluate_mf(x_range, mf)
        label = f"Set {i+1} (cluster {mf['center_idx']})"
        ax1.plot(x_range, mu, linewidth=2.5, color=colors[i], label=label)
        ax1.fill_between(x_range, mu, alpha=0.15, color=colors[i])
        
        # Mark the center
        if mf_type == "triangular":
            center = mf["peak"]
        elif mf_type == "gaussian":
            center = mf["mean"]
        elif mf_type == "trapezoidal":
            center = (mf["b"] + mf["c"]) / 2
        else:
            center = mf.get("peak", mf.get("mean", 0))
        ax1.axvline(center, color=colors[i], linestyle="--", alpha=0.5, linewidth=1)
    
    ax1.set_xlabel(feature, fontsize=12)
    ax1.set_ylabel("Membership Degree (μ)", fontsize=12)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlim(x_range[0], x_range[-1])
    
    # Overlay histogram on secondary axis
    if show_histogram:
        ax2 = ax1.twinx()
        ax2.hist(feat_values, bins=30, alpha=0.25, color="gray", density=True)
        ax2.set_ylabel("Density", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_ylim(bottom=0)
    
    type_label = mf_type.capitalize()
    ax1.set_title(f"{type_label} Membership Functions: {feature}", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    
    # Clean filename
    safe_name = feature.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"membership_{safe_name}{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved membership plot: {output_path}")


def plot_mf_comparison(
    feature: str,
    triangular_params: dict,
    gaussian_params: dict,
    trapezoidal_params: dict,
    X_raw: np.ndarray,
    feature_idx: int,
    output_dir: Path = Path("figures")
) -> None:
    """
    Plot comparison of different membership function types for a single feature.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature range
    feat_values = X_raw[:, feature_idx]
    feat_min, feat_max = feat_values.min(), feat_values.max()
    padding = (feat_max - feat_min) * 0.05
    x_range = np.linspace(feat_min - padding, feat_max + padding, 500)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    params_list = [
        (triangular_params, "Triangular", "Sharp peaks, linear"),
        (gaussian_params, "Gaussian", "Smooth, bell-shaped"),
        (trapezoidal_params, "Trapezoidal", "Flat top region"),
    ]
    
    for ax, (params, title, desc) in zip(axes, params_list):
        mfs = params[feature]
        k = len(mfs)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
        
        for i, mf in enumerate(mfs):
            mu = evaluate_mf(x_range, mf)
            ax.plot(x_range, mu, linewidth=2, color=colors[i], label=f"Set {i+1}")
            ax.fill_between(x_range, mu, alpha=0.15, color=colors[i])
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("μ", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"{title}\n({desc})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
    
    plt.suptitle(f"Membership Function Comparison: {feature}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    safe_name = feature.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"mf_comparison_{safe_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"mf_comparison_{safe_name}.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Saved MF comparison: {output_path}")


def plot_all_membership_functions(
    membership_params: dict,
    X_raw: np.ndarray,
    feature_names: list[str],
    output_dir: Path = Path("figures"),
    suffix: str = ""
) -> None:
    """Plot membership functions for all features."""
    for idx, feature in enumerate(feature_names):
        plot_membership_function(
            feature, membership_params, X_raw, idx, output_dir, suffix=suffix
        )


def plot_quality_by_cluster(
    labels: np.ndarray,
    y_quality: np.ndarray,
    output_path: Path = Path("figures/quality_by_cluster.png")
) -> None:
    """Plot quality distribution per cluster."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    k = len(np.unique(labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    data = pd.DataFrame({"Cluster": labels, "Quality": y_quality})
    sns.boxplot(data=data, x="Cluster", y="Quality", hue="Cluster", ax=axes[0], palette="viridis", legend=False)
    axes[0].set_title("Quality Distribution by Cluster", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Wine Quality")
    
    # Mean quality per cluster
    cluster_means = data.groupby("Cluster")["Quality"].agg(["mean", "std", "count"])
    cluster_means = cluster_means.reset_index()
    
    bars = axes[1].bar(
        cluster_means["Cluster"],
        cluster_means["mean"],
        yerr=cluster_means["std"],
        capsize=5,
        color=plt.cm.viridis(np.linspace(0.2, 0.8, k)),
        edgecolor="black"
    )
    axes[1].set_title("Mean Quality by Cluster (±1 std)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Mean Quality")
    
    # Add count labels on bars
    for bar, count in zip(bars, cluster_means["count"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"n={count}",
            ha="center",
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved quality by cluster plot to {output_path}")
    
    return cluster_means


def plot_clusters_2d(
    X_std: np.ndarray,
    labels: np.ndarray,
    centers_std: np.ndarray,
    feature_names: list[str],
    output_dir: Path = Path("figures")
) -> None:
    """
    Visualize clusters using PCA projection to 2D.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k = len(centers_std)
    
    # Simple PCA implementation (no sklearn dependency)
    # Center the data
    X_centered = X_std - X_std.mean(axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to 2D
    X_pca = X_centered @ eigenvectors[:, :2]
    centers_pca = (centers_std - X_std.mean(axis=0)) @ eigenvectors[:, :2]
    
    # Variance explained
    var_explained = eigenvalues[:2] / eigenvalues.sum() * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    
    for i in range(k):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], alpha=0.6, s=30, label=f"Cluster {i} (n={mask.sum()})")
    
    # Plot centers
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c="red", marker="X", s=200, edgecolor="black", linewidth=2, 
               label="Centers", zorder=10)
    
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=11)
    ax.set_title(f"Cluster Visualization (PCA Projection, k={k})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "clusters_pca.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Saved PCA cluster plot to {output_dir}/clusters_pca.png")


def plot_clusters_pairwise(
    X_raw: np.ndarray,
    labels: np.ndarray,
    centers_raw: np.ndarray,
    feature_names: list[str],
    output_dir: Path = Path("figures"),
    features_to_plot: list[str] = None
) -> None:
    """
    Plot pairwise scatter plots for selected features, colored by cluster.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k = len(centers_raw)
    
    # Default to key features if not specified
    if features_to_plot is None:
        features_to_plot = ["alcohol", "volatile acidity", "sulphates", "pH"]
    
    # Get indices
    indices = [feature_names.index(f) for f in features_to_plot if f in feature_names]
    n_features = len(indices)
    
    if n_features < 2:
        print("Need at least 2 features for pairwise plot")
        return
    
    fig, axes = plt.subplots(n_features, n_features, figsize=(12, 12))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram per cluster
                for c in range(k):
                    mask = labels == c
                    ax.hist(X_raw[mask, idx_i], bins=20, alpha=0.5, color=colors[c], density=True)
                ax.set_ylabel("Density" if j == 0 else "")
            else:
                # Off-diagonal: scatter
                for c in range(k):
                    mask = labels == c
                    ax.scatter(X_raw[mask, idx_j], X_raw[mask, idx_i], 
                               c=[colors[c]], alpha=0.5, s=15)
                
                # Plot centers
                ax.scatter(centers_raw[:, idx_j], centers_raw[:, idx_i],
                           c="red", marker="X", s=100, edgecolor="black", linewidth=1.5, zorder=10)
            
            # Labels
            if i == n_features - 1:
                ax.set_xlabel(features_to_plot[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(features_to_plot[i], fontsize=9)
            
            ax.tick_params(axis="both", labelsize=7)
    
    # Add legend
    legend_elements = [plt.scatter([], [], c=[colors[i]], label=f"Cluster {i}") for i in range(k)]
    legend_elements.append(plt.scatter([], [], c="red", marker="X", s=100, label="Centers"))
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9)
    
    plt.suptitle(f"Pairwise Cluster Scatter Plots (k={k})", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "clusters_pairwise.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "clusters_pairwise.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Saved pairwise cluster plot to {output_dir}/clusters_pairwise.png")


def plot_cluster_profiles(
    centers_raw: np.ndarray,
    feature_names: list[str],
    X_raw: np.ndarray,
    output_dir: Path = Path("figures")
) -> None:
    """
    Plot radar/spider chart showing cluster profiles.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k, n_features = centers_raw.shape
    
    # Normalize centers to [0, 1] for comparison
    feat_min = X_raw.min(axis=0)
    feat_max = X_raw.max(axis=0)
    centers_norm = (centers_raw - feat_min) / (feat_max - feat_min + 1e-8)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    
    for i in range(k):
        values = centers_norm[i].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Cluster Profiles (Normalized, k={k})", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_profiles_radar.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "cluster_profiles_radar.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Saved cluster profiles radar chart to {output_dir}/cluster_profiles_radar.png")


def plot_cluster_centers_heatmap(
    centers_raw: np.ndarray,
    feature_names: list[str],
    output_path: Path = Path("figures/cluster_centers_heatmap.png")
) -> None:
    """Plot heatmap of cluster centers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize for visualization
    centers_norm = (centers_raw - centers_raw.min(axis=0)) / (
        centers_raw.max(axis=0) - centers_raw.min(axis=0) + 1e-8
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        centers_norm,
        annot=centers_raw.round(2),
        fmt="",
        xticklabels=feature_names,
        yticklabels=[f"Cluster {i}" for i in range(len(centers_raw))],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Normalized Value"}
    )
    
    ax.set_title("Cluster Centers (values shown, colors normalized)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved cluster centers heatmap to {output_path}")


def generate_all_pdfs(
    membership_params: dict,
    X_raw: np.ndarray,
    feature_names: list[str],
    inertia_df: pd.DataFrame,
    centers_raw: np.ndarray,
    labels: np.ndarray,
    y_quality: np.ndarray,
    output_dir: Path = Path("figures")
) -> None:
    """Generate combined PDF with ALL existing PNG figures."""
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combined PDF with all figures - collect all existing PNGs
    combined_pdf_path = output_dir / "all_figures.pdf"
    
    # Define the order of figures for the combined PDF
    # First, the key analysis plots in logical order
    priority_order = [
        # Data exploration
        "feature_distributions.png",
        "feature_boxplots.png",
        "feature_summary_stats.png",
        "feature_correlation.png",
        # Cluster selection
        "inertia_vs_k.png",
        "cluster_metrics_panel.png",
        # Cluster visualization
        "clusters_pca.png",
        "clusters_pairwise.png",
        "cluster_profiles_radar.png",
        "cluster_centers_heatmap.png",
        "quality_by_cluster.png",
        # All features panels (per MF type)
        "all_features_triangular.png",
        "all_features_gaussian.png",
        "all_features_trapezoidal.png",
    ]
    
    # Get all PNG files
    all_pngs = sorted(output_dir.glob("*.png"))
    
    # Build ordered list: priority files first, then remaining files
    ordered_pngs = []
    remaining_pngs = set(all_pngs)
    
    for priority_file in priority_order:
        full_path = output_dir / priority_file
        if full_path in remaining_pngs:
            ordered_pngs.append(full_path)
            remaining_pngs.remove(full_path)
    
    # Add MF panel comparisons next
    mf_panels = sorted([p for p in remaining_pngs if p.name.startswith("mf_panel_")])
    for p in mf_panels:
        ordered_pngs.append(p)
        remaining_pngs.remove(p)
    
    # Add individual membership plots
    membership_plots = sorted([p for p in remaining_pngs if p.name.startswith("membership_")])
    for p in membership_plots:
        ordered_pngs.append(p)
        remaining_pngs.remove(p)
    
    # Add any remaining files
    ordered_pngs.extend(sorted(remaining_pngs))
    
    # Create combined PDF from all PNGs
    with PdfPages(combined_pdf_path) as pdf:
        for png_path in ordered_pngs:
            try:
                # Load image and add to PDF
                img = Image.open(png_path)
                
                # Create figure with appropriate size
                dpi = 100
                fig_width = img.width / dpi
                fig_height = img.height / dpi
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.imshow(img)
                ax.axis("off")
                plt.tight_layout(pad=0)
                
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not add {png_path.name} to PDF: {e}")
    
    print(f"Saved combined PDF with {len(ordered_pngs)} figures to {combined_pdf_path}")
    
    # Also generate individual PDFs for any PNGs that don't have one
    for png_path in all_pngs:
        pdf_path = png_path.with_suffix(".pdf")
        if not pdf_path.exists():
            try:
                img = Image.open(png_path)
                dpi = 100
                fig_width = img.width / dpi
                fig_height = img.height / dpi
                
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.imshow(img)
                ax.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create PDF for {png_path.name}: {e}")
    
    print(f"Saved individual PDFs to {output_dir}/")

