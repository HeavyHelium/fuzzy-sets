"""
K-means clustering implementation from scratch.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def euclidean_distance(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances from each point to each center.
    
    Args:
        X: Data points (N, d)
        centers: Cluster centers (k, d)
    
    Returns:
        distances: Distance matrix (N, k)
    """
    # Use broadcasting: (N, 1, d) - (1, k, d) -> (N, k, d) -> sum -> (N, k)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances


def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initialize cluster centers using k-means++ algorithm.
    
    Args:
        X: Data points (N, d)
        k: Number of clusters
        rng: Random number generator
    
    Returns:
        centers: Initial cluster centers (k, d)
    """
    n_samples, n_features = X.shape
    centers = np.empty((k, n_features), dtype=X.dtype)
    
    # Choose first center uniformly at random
    idx = rng.integers(n_samples)
    centers[0] = X[idx]
    
    # Choose remaining centers with probability proportional to D(x)^2
    for i in range(1, k):
        # Compute distances to nearest existing center
        distances = euclidean_distance(X, centers[:i])
        min_distances = distances.min(axis=1)
        
        # Square distances for probability weighting
        probs = min_distances ** 2
        probs /= probs.sum()
        
        # Sample next center
        idx = rng.choice(n_samples, p=probs)
        centers[i] = X[idx]
    
    return centers


def assign_clusters(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each point to nearest cluster center."""
    distances = euclidean_distance(X, centers)
    return distances.argmin(axis=1)


def update_centers(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute centers as mean of assigned points."""
    n_features = X.shape[1]
    centers = np.zeros((k, n_features), dtype=X.dtype)
    
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            centers[i] = X[mask].mean(axis=0)
        else:
            # Empty cluster: keep previous center (handled by caller)
            centers[i] = np.nan
    
    return centers


def compute_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Compute sum of squared distances to assigned centers."""
    distances = euclidean_distance(X, centers)
    assigned_distances = distances[np.arange(len(labels)), labels]
    return float((assigned_distances ** 2).sum())


def kmeans_single(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Run single K-means clustering.
    
    Args:
        X: Data points (N, d)
        k: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance (max center shift)
        rng: Random number generator
    
    Returns:
        centers: Final cluster centers (k, d)
        labels: Cluster assignments (N,)
        inertia: Sum of squared distances
        n_iter: Number of iterations performed
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize with k-means++
    centers = kmeans_plusplus_init(X, k, rng)
    
    for iteration in range(max_iter):
        # Assign points to clusters
        labels = assign_clusters(X, centers)
        
        # Update centers
        new_centers = update_centers(X, labels, k)
        
        # Handle empty clusters by reinitializing from random points
        empty_mask = np.isnan(new_centers).any(axis=1)
        if empty_mask.any():
            n_empty = empty_mask.sum()
            random_idx = rng.choice(len(X), size=n_empty, replace=False)
            new_centers[empty_mask] = X[random_idx]
        
        # Check convergence
        center_shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
        centers = new_centers
        
        if center_shift < tol:
            break
    
    # Final assignment and inertia
    labels = assign_clusters(X, centers)
    inertia = compute_inertia(X, labels, centers)
    
    return centers, labels, inertia, iteration + 1


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
    seed: int = 42
) -> dict:
    """
    K-means clustering with multiple restarts.
    
    Args:
        X: Data points (N, d)
        k: Number of clusters
        max_iter: Maximum iterations per run
        tol: Convergence tolerance
        n_init: Number of random restarts
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
            - centers: Best cluster centers (k, d)
            - labels: Best cluster assignments (N,)
            - inertia: Best (lowest) inertia
            - n_iter: Iterations for best run
            - cluster_sizes: Number of points per cluster
    """
    rng = np.random.default_rng(seed)
    
    best_centers = None
    best_labels = None
    best_inertia = np.inf
    best_n_iter = 0
    
    for init_idx in range(n_init):
        centers, labels, inertia, n_iter = kmeans_single(
            X, k, max_iter, tol, rng
        )
        
        if inertia < best_inertia:
            best_centers = centers
            best_labels = labels
            best_inertia = inertia
            best_n_iter = n_iter
    
    # Compute cluster sizes
    cluster_sizes = np.bincount(best_labels, minlength=k)
    
    return {
        "centers": best_centers,
        "labels": best_labels,
        "inertia": best_inertia,
        "n_iter": best_n_iter,
        "cluster_sizes": cluster_sizes,
    }


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score for clustering quality.
    
    Measures how similar points are to their own cluster compared to other clusters.
    Range: [-1, 1], higher is better.
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    if k == 1 or k >= n_samples:
        return 0.0
    
    # Compute pairwise distances
    distances = euclidean_distance(X, X)
    
    silhouette_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        # a(i) = mean distance to points in same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        if same_cluster.sum() > 0:
            a_i = distances[i, same_cluster].mean()
        else:
            a_i = 0
        
        # b(i) = min mean distance to points in other clusters
        b_i = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = labels == label
                if other_cluster.sum() > 0:
                    mean_dist = distances[i, other_cluster].mean()
                    b_i = min(b_i, mean_dist)
        
        if b_i == np.inf:
            b_i = 0
        
        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0
    
    return float(silhouette_values.mean())


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute Davies-Bouldin index for clustering quality.
    
    Measures the average similarity between clusters.
    Lower is better (0 = perfect separation).
    """
    k = len(centers)
    
    if k == 1:
        return 0.0
    
    # Compute within-cluster scatter (average distance to center)
    scatter = np.zeros(k)
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            cluster_points = X[mask]
            distances_to_center = np.sqrt(((cluster_points - centers[i]) ** 2).sum(axis=1))
            scatter[i] = distances_to_center.mean()
    
    # Compute between-cluster distances
    center_distances = euclidean_distance(centers, centers)
    
    # Compute DB index
    db_values = np.zeros(k)
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j and center_distances[i, j] > 0:
                ratio = (scatter[i] + scatter[j]) / center_distances[i, j]
                max_ratio = max(max_ratio, ratio)
        db_values[i] = max_ratio
    
    return float(db_values.mean())


def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute Calinski-Harabasz index (Variance Ratio Criterion).
    
    Ratio of between-cluster variance to within-cluster variance.
    Higher is better.
    """
    n_samples, n_features = X.shape
    k = len(centers)
    
    if k == 1 or k >= n_samples:
        return 0.0
    
    # Overall centroid
    overall_center = X.mean(axis=0)
    
    # Between-cluster dispersion (weighted by cluster size)
    between_cluster = 0
    for i in range(k):
        n_i = (labels == i).sum()
        between_cluster += n_i * ((centers[i] - overall_center) ** 2).sum()
    
    # Within-cluster dispersion
    within_cluster = 0
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            within_cluster += ((X[mask] - centers[i]) ** 2).sum()
    
    if within_cluster == 0:
        return 0.0
    
    # CH index
    ch = (between_cluster / (k - 1)) / (within_cluster / (n_samples - k))
    return float(ch)


def sweep_k(
    X: np.ndarray,
    k_range: range = range(3, 9),
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sweep over different values of k and record multiple clustering metrics.
    
    Returns:
        DataFrame with columns: k, inertia, silhouette, davies_bouldin, calinski_harabasz
    """
    results = []
    
    for k in k_range:
        print(f"Running K-means with k={k}...")
        result = kmeans(X, k, max_iter, tol, n_init, seed)
        
        # Compute additional metrics
        sil = silhouette_score(X, result["labels"])
        db = davies_bouldin_index(X, result["labels"], result["centers"])
        ch = calinski_harabasz_index(X, result["labels"], result["centers"])
        
        results.append({
            "k": k,
            "inertia": result["inertia"],
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
        })
        print(f"  Inertia: {result['inertia']:.2f}, Silhouette: {sil:.3f}, DB: {db:.3f}, CH: {ch:.1f}")
    
    return pd.DataFrame(results)


def save_kmeans_results(
    result: dict,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    output_dir: Path = Path("outputs")
) -> None:
    """Save K-means results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Centers in standardized space
    centers_std_df = pd.DataFrame(
        result["centers"],
        columns=feature_names
    )
    centers_std_df.index.name = "cluster"
    centers_std_df.to_csv(output_dir / "centers_std.csv")
    
    # Centers in original units (inverse transform)
    centers_raw = result["centers"] * stds + means
    centers_raw_df = pd.DataFrame(
        centers_raw,
        columns=feature_names
    )
    centers_raw_df.index.name = "cluster"
    centers_raw_df.to_csv(output_dir / "centers_raw.csv")
    
    # Labels
    np.save(output_dir / "labels.npy", result["labels"])
    
    # Summary
    summary = {
        "k": int(len(result["centers"])),
        "inertia": float(result["inertia"]),
        "n_iter": int(result["n_iter"]),
        "cluster_sizes": result["cluster_sizes"].tolist(),
    }
    with open(output_dir / "kmeans_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved K-means results to {output_dir}")
    
    return centers_raw

