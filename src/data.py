"""
Data loading, preprocessing, and scaling for Wine Quality dataset.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# Wine Quality dataset URL (UCI ML Repository)
WINE_RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def download_wine_data(data_dir: Path = Path("data")) -> Path:
    """Download the Wine Quality (red) dataset if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / "winequality-red.csv"
    
    if not filepath.exists():
        print(f"Downloading Wine Quality dataset to {filepath}...")
        urllib.request.urlretrieve(WINE_RED_URL, filepath)
        print("Download complete.")
    else:
        print(f"Dataset already exists at {filepath}")
    
    return filepath


def load_wine_data(filepath: Path | str) -> pd.DataFrame:
    """Load the wine quality CSV file."""
    df = pd.read_csv(filepath, sep=";")
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
    return df


def preprocess_wine_data(
    df: pd.DataFrame,
    quality_col: str = "quality"
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Preprocess the wine quality dataset.
    
    Returns:
        X_raw: Feature matrix (N, d) in original units (after imputation)
        y_quality: Quality labels for optional analysis
        feature_names: List of feature column names
        X_original: Original feature matrix before imputation (for missing value stats)
    """
    # Separate features from quality label
    feature_cols = [col for col in df.columns if col != quality_col]
    
    X_raw = df[feature_cols].values.astype(np.float64)
    X_original = X_raw.copy()  # Keep copy before imputation
    y_quality = df[quality_col].values if quality_col in df.columns else None
    
    # Check for missing values per feature
    missing_per_feature = np.isnan(X_raw).sum(axis=0)
    n_missing = missing_per_feature.sum()
    
    print(f"\nMissing value analysis:")
    print(f"  Total missing: {n_missing} / {X_raw.size} ({100*n_missing/X_raw.size:.2f}%)")
    
    if n_missing > 0:
        print(f"\n  Per-feature breakdown:")
        for j, (feat, count) in enumerate(zip(feature_cols, missing_per_feature)):
            pct = 100 * count / X_raw.shape[0]
            status = f"{count} ({pct:.1f}%)" if count > 0 else "0 (complete)"
            print(f"    {feat}: {status}")
        
        print("\n  Imputation: Replacing NaN with column means...")
        col_means = np.nanmean(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            mask = np.isnan(X_raw[:, j])
            X_raw[mask, j] = col_means[j]
        print("  Imputation complete.")
    else:
        print("  All features complete (no imputation needed).")
    
    print(f"\nFeature matrix shape: {X_raw.shape}")
    print(f"Features: {feature_cols}")
    
    return X_raw, y_quality, feature_cols, X_original


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using z-score normalization.
    
    Returns:
        X_std: Standardized feature matrix
        means: Column means used for standardization
        stds: Column standard deviations used for standardization
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    
    # Avoid division by zero for constant features
    stds[stds == 0] = 1.0
    
    X_std = (X - means) / stds
    
    return X_std, means, stds


def inverse_standardize(X_std: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Convert standardized values back to original scale."""
    return X_std * stds + means


def save_feature_stats(
    X_raw: np.ndarray,
    X_std: np.ndarray,
    feature_names: list[str],
    output_path: Path = Path("outputs/feature_stats.csv"),
    X_original: np.ndarray = None
) -> None:
    """Save feature statistics (mean/std before and after scaling)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count missing values (use X_original if provided, before imputation)
    if X_original is not None:
        missing_counts = np.isnan(X_original).sum(axis=0)
        missing_pct = 100 * missing_counts / X_original.shape[0]
    else:
        missing_counts = np.zeros(len(feature_names))
        missing_pct = np.zeros(len(feature_names))
    
    stats = pd.DataFrame({
        "feature": feature_names,
        "n_missing": missing_counts.astype(int),
        "pct_missing": missing_pct.round(2),
        "raw_mean": X_raw.mean(axis=0),
        "raw_std": X_raw.std(axis=0),
        "raw_min": X_raw.min(axis=0),
        "raw_max": X_raw.max(axis=0),
        "std_mean": X_std.mean(axis=0),
        "std_std": X_std.std(axis=0),
    })
    
    stats.to_csv(output_path, index=False)
    print(f"Saved feature statistics to {output_path}")


def load_and_prepare_data(
    data_dir: Path = Path("data"),
    output_dir: Path = Path("outputs")
) -> dict:
    """
    Full data loading and preprocessing pipeline.
    
    Returns dict with:
        - X_raw: Original feature matrix (after imputation)
        - X_std: Standardized feature matrix  
        - y_quality: Quality labels
        - feature_names: List of feature names
        - means: Standardization means
        - stds: Standardization stds
    """
    # Download if needed
    filepath = download_wine_data(data_dir)
    
    # Load
    df = load_wine_data(filepath)
    
    # Preprocess (returns X_original for missing value tracking)
    X_raw, y_quality, feature_names, X_original = preprocess_wine_data(df)
    
    # Standardize
    X_std, means, stds = standardize(X_raw)
    
    # Save stats (include missing value counts from X_original)
    save_feature_stats(X_raw, X_std, feature_names, output_dir / "feature_stats.csv", X_original)
    
    return {
        "X_raw": X_raw,
        "X_std": X_std,
        "y_quality": y_quality,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
    }

