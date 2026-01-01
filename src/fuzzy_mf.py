"""
Fuzzy membership function construction from K-means cluster centers.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def triangular_mf(x: np.ndarray, left: float, peak: float, right: float) -> np.ndarray:
    """
    Evaluate triangular membership function.
    
    Shape: /\\ (sharp peak)
    
    Args:
        x: Input values to evaluate
        left: Left base of triangle (μ=0)
        peak: Peak of triangle (μ=1)
        right: Right base of triangle (μ=0)
    
    Returns:
        Membership values in [0, 1]
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    
    # Left slope: (x - L) / (P - L) for L < x <= P
    left_mask = (x > left) & (x <= peak)
    if peak > left:
        result[left_mask] = (x[left_mask] - left) / (peak - left)
    
    # Right slope: (R - x) / (R - P) for P < x < R
    right_mask = (x > peak) & (x < right)
    if right > peak:
        result[right_mask] = (right - x[right_mask]) / (right - peak)
    
    # Peak
    result[x == peak] = 1.0
    
    return result


def gaussian_mf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """
    Evaluate Gaussian membership function.
    
    Shape: Bell curve (smooth, no sharp corners)
    
    μ(x) = exp(-((x - mean)² / (2σ²)))
    
    Args:
        x: Input values to evaluate
        mean: Center of the Gaussian (μ=1)
        sigma: Standard deviation (controls width)
    
    Returns:
        Membership values in [0, 1]
    """
    x = np.asarray(x)
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def trapezoidal_mf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Evaluate trapezoidal membership function.
    
    Shape: /‾‾\\ (flat top between b and c)
    
    Args:
        x: Input values to evaluate
        a: Left foot (μ=0)
        b: Left shoulder (μ=1 starts)
        c: Right shoulder (μ=1 ends)
        d: Right foot (μ=0)
    
    Returns:
        Membership values in [0, 1]
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    
    # Rising edge: a < x < b
    rising = (x > a) & (x < b)
    if b > a:
        result[rising] = (x[rising] - a) / (b - a)
    
    # Flat top: b <= x <= c
    flat = (x >= b) & (x <= c)
    result[flat] = 1.0
    
    # Falling edge: c < x < d
    falling = (x > c) & (x < d)
    if d > c:
        result[falling] = (d - x[falling]) / (d - c)
    
    return result


def construct_membership_functions(
    centers_raw: np.ndarray,
    X_raw: np.ndarray,
    feature_names: list[str],
    mf_type: str = "triangular"
) -> dict:
    """
    Construct membership functions for each feature using cluster centers.
    
    Supported types:
    - "triangular": Sharp peak, linear slopes
    - "gaussian": Smooth bell curve
    - "trapezoidal": Flat top region
    
    For each feature j:
    1. Extract j-th coordinate of each center
    2. Sort the centers
    3. Compute parameters based on MF type
    
    Args:
        centers_raw: Cluster centers in original units (k, d)
        X_raw: Original feature matrix (N, d) for min/max bounds
        feature_names: List of feature names
        mf_type: Type of membership function ("triangular", "gaussian", "trapezoidal")
    
    Returns:
        Dictionary mapping feature name -> list of MF parameters
    """
    k, d = centers_raw.shape
    membership_params = {"mf_type": mf_type}
    
    for j, feature in enumerate(feature_names):
        # Get center values for this feature
        center_values = centers_raw[:, j]
        
        # Sort and get original indices
        sorted_indices = np.argsort(center_values)
        sorted_centers = center_values[sorted_indices]
        
        # Feature bounds
        feat_min = X_raw[:, j].min()
        feat_max = X_raw[:, j].max()
        
        # Compute midpoints between adjacent sorted centers
        midpoints = (sorted_centers[:-1] + sorted_centers[1:]) / 2
        
        mfs = []
        for i in range(k):
            original_cluster_idx = int(sorted_indices[i])
            center = sorted_centers[i]
            
            # Determine boundaries
            if i == 0:
                left_bound = feat_min
                right_bound = midpoints[0] if k > 1 else feat_max
            elif i == k - 1:
                left_bound = midpoints[-1]
                right_bound = feat_max
            else:
                left_bound = midpoints[i - 1]
                right_bound = midpoints[i]
            
            if mf_type == "triangular":
                mfs.append({
                    "type": "triangular",
                    "left": float(left_bound),
                    "peak": float(center),
                    "right": float(right_bound),
                    "center_idx": original_cluster_idx,
                    "sorted_idx": i,
                })
            
            elif mf_type == "gaussian":
                # Sigma based on distance to nearest boundary
                # Use ~2 sigma to reach near-zero at boundaries
                sigma = min(center - left_bound, right_bound - center) / 2.0
                sigma = max(sigma, 1e-6)  # Avoid zero sigma
                
                mfs.append({
                    "type": "gaussian",
                    "mean": float(center),
                    "sigma": float(sigma),
                    "center_idx": original_cluster_idx,
                    "sorted_idx": i,
                })
            
            elif mf_type == "trapezoidal":
                # Create flat top around center (±10% of range)
                width = right_bound - left_bound
                flat_width = width * 0.2  # 20% flat top
                
                mfs.append({
                    "type": "trapezoidal",
                    "a": float(left_bound),
                    "b": float(center - flat_width / 2),
                    "c": float(center + flat_width / 2),
                    "d": float(right_bound),
                    "center_idx": original_cluster_idx,
                    "sorted_idx": i,
                })
            
            else:
                raise ValueError(f"Unknown MF type: {mf_type}. Use 'triangular', 'gaussian', or 'trapezoidal'.")
        
        membership_params[feature] = mfs
    
    return membership_params


def evaluate_mf(x: np.ndarray, mf_params: dict) -> np.ndarray:
    """Evaluate a single membership function given its parameters."""
    mf_type = mf_params.get("type", "triangular")
    
    if mf_type == "triangular":
        return triangular_mf(x, mf_params["left"], mf_params["peak"], mf_params["right"])
    elif mf_type == "gaussian":
        return gaussian_mf(x, mf_params["mean"], mf_params["sigma"])
    elif mf_type == "trapezoidal":
        return trapezoidal_mf(x, mf_params["a"], mf_params["b"], mf_params["c"], mf_params["d"])
    else:
        raise ValueError(f"Unknown MF type: {mf_type}")


def evaluate_membership(
    x: float | np.ndarray,
    feature: str,
    membership_params: dict
) -> np.ndarray:
    """
    Evaluate all membership functions for a feature at given values.
    
    Args:
        x: Input value(s)
        feature: Feature name
        membership_params: Dictionary of MF parameters
    
    Returns:
        Membership values for each fuzzy set (shape: (k,) or (n_values, k))
    """
    x = np.atleast_1d(x)
    mfs = membership_params[feature]
    k = len(mfs)
    
    result = np.zeros((len(x), k))
    for i, mf in enumerate(mfs):
        result[:, i] = evaluate_mf(x, mf)
    
    return result.squeeze() if len(x) == 1 else result


def fuzzify_sample(
    sample: np.ndarray,
    feature_names: list[str],
    membership_params: dict
) -> dict:
    """
    Fuzzify a single sample, returning membership values for all features.
    
    Args:
        sample: Feature values for one sample (d,)
        feature_names: List of feature names
        membership_params: Dictionary of MF parameters
    
    Returns:
        Dictionary mapping feature name -> membership values (k,)
    """
    result = {}
    for j, feature in enumerate(feature_names):
        result[feature] = evaluate_membership(sample[j], feature, membership_params)
    return result


def save_membership_params(
    membership_params: dict,
    output_path: Path = Path("outputs/membership_params.json")
) -> None:
    """Save membership function parameters to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(membership_params, f, indent=2)
    
    print(f"Saved membership parameters to {output_path}")


def load_membership_params(path: Path = Path("outputs/membership_params.json")) -> dict:
    """Load membership function parameters from JSON."""
    with open(path) as f:
        return json.load(f)


def membership_params_to_dataframe(membership_params: dict) -> pd.DataFrame:
    """Convert membership parameters to a flat DataFrame for easy viewing."""
    rows = []
    mf_type = membership_params.get("mf_type", "triangular")
    
    for feature, mfs in membership_params.items():
        if feature == "mf_type":
            continue
        for mf in mfs:
            row = {
                "feature": feature,
                "type": mf.get("type", mf_type),
                "sorted_idx": mf["sorted_idx"],
                "center_idx": mf["center_idx"],
            }
            # Add type-specific parameters
            if mf.get("type", mf_type) == "triangular":
                row.update({"left": mf["left"], "peak": mf["peak"], "right": mf["right"]})
            elif mf.get("type", mf_type) == "gaussian":
                row.update({"mean": mf["mean"], "sigma": mf["sigma"]})
            elif mf.get("type", mf_type) == "trapezoidal":
                row.update({"a": mf["a"], "b": mf["b"], "c": mf["c"], "d": mf["d"]})
            rows.append(row)
    return pd.DataFrame(rows)

