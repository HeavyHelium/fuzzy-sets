#!/usr/bin/env python3
"""
K-means Clustering with Automatic Fuzzy Membership Functions
=============================================================

End-to-end pipeline for the Wine Quality dataset:
1. Load and preprocess data
2. Run K-means clustering (with k sweep for elbow analysis)
3. Construct triangular fuzzy membership functions from cluster centers
4. Visualize membership functions and cluster analysis
5. Save all outputs and generate summary statistics

Usage:
    python main.py [--k K] [--seed SEED]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_and_prepare_data, inverse_standardize
from src.kmeans import kmeans, sweep_k, save_kmeans_results
from src.fuzzy_mf import (
    construct_membership_functions,
    save_membership_params,
    membership_params_to_dataframe,
)
from src.viz import (
    plot_inertia_vs_k,
    plot_all_membership_functions,
    plot_quality_by_cluster,
    plot_cluster_centers_heatmap,
    generate_all_pdfs,
    plot_mf_comparison,
    plot_cluster_metrics_panel,
    plot_mf_types_panel,
    plot_all_features_panel,
    plot_feature_statistics,
    plot_clusters_2d,
    plot_clusters_pairwise,
    plot_cluster_profiles,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-means clustering with fuzzy membership functions on Wine Quality"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of clusters (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n-init", type=int, default=10,
        help="Number of K-means restarts (default: 10)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=300,
        help="Maximum iterations for K-means (default: 300)"
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the k sweep (elbow analysis)"
    )
    parser.add_argument(
        "--mf-type", type=str, default="triangular",
        choices=["triangular", "gaussian", "trapezoidal"],
        help="Type of membership function (default: triangular)"
    )
    parser.add_argument(
        "--compare-mf", action="store_true",
        help="Generate comparison plots of all MF types"
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def analyze_clusters_with_quality(
    labels: np.ndarray,
    y_quality: np.ndarray,
    output_dir: Path
) -> pd.DataFrame:
    """Analyze cluster composition with respect to wine quality."""
    df = pd.DataFrame({
        "cluster": labels,
        "quality": y_quality,
    })
    
    summary = df.groupby("cluster")["quality"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).round(3)
    
    summary.to_csv(output_dir / "cluster_quality_summary.csv")
    print("\nCluster Quality Summary:")
    print(summary.to_string())
    
    return summary


def main():
    args = parse_args()
    
    # Paths
    data_dir = Path("data")
    output_dir = Path("outputs")
    figures_dir = Path("figures")
    
    # =========================================================================
    print_section("1. DATA LOADING AND PREPROCESSING")
    # =========================================================================
    
    data = load_and_prepare_data(data_dir, output_dir)
    
    X_raw = data["X_raw"]
    X_std = data["X_std"]
    y_quality = data["y_quality"]
    feature_names = data["feature_names"]
    means = data["means"]
    stds = data["stds"]
    
    print(f"\nDataset summary:")
    print(f"  - Samples (N): {X_raw.shape[0]}")
    print(f"  - Features (d): {X_raw.shape[1]}")
    print(f"  - Feature names: {feature_names}")
    
    # Generate data statistics visualizations
    print("\nGenerating feature statistics plots...")
    plot_feature_statistics(X_raw, feature_names, figures_dir)
    
    # =========================================================================
    print_section("2. K-MEANS CLUSTERING")
    # =========================================================================
    
    # Elbow analysis (k sweep) with multiple metrics
    if not args.skip_sweep:
        print("\nRunning k sweep with multiple clustering metrics...")
        metrics_df = sweep_k(
            X_std,
            k_range=range(3, 9),
            n_init=args.n_init,
            seed=args.seed
        )
        metrics_df.to_csv(output_dir / "kmeans_metrics_by_k.csv", index=False)
        
        # Plot individual elbow curve
        plot_inertia_vs_k(metrics_df, figures_dir / "inertia_vs_k.png")
        
        # Plot comprehensive metrics panel (2x2)
        plot_cluster_metrics_panel(metrics_df, figures_dir / "cluster_metrics_panel.png")
        
        # Find best k by each metric
        print("\n  Best k by metric:")
        print(f"    Silhouette (max):        k={metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']}")
        print(f"    Davies-Bouldin (min):    k={metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']}")
        print(f"    Calinski-Harabasz (max): k={metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']}")
        
        inertia_df = metrics_df  # For compatibility
    
    # Run K-means with chosen k
    print(f"\nRunning K-means with k={args.k}...")
    result = kmeans(
        X_std,
        k=args.k,
        max_iter=args.max_iter,
        n_init=args.n_init,
        seed=args.seed
    )
    
    print(f"\nK-means Results:")
    print(f"  - Inertia: {result['inertia']:.2f}")
    print(f"  - Iterations: {result['n_iter']}")
    print(f"  - Cluster sizes: {result['cluster_sizes'].tolist()}")
    
    # Save results and get centers in original units
    centers_raw = save_kmeans_results(
        result, feature_names, means, stds, output_dir
    )
    
    # Cluster visualizations
    print("\nGenerating cluster visualizations...")
    plot_cluster_centers_heatmap(centers_raw, feature_names, figures_dir / "cluster_centers_heatmap.png")
    plot_clusters_2d(X_std, result["labels"], result["centers"], feature_names, figures_dir)
    plot_clusters_pairwise(X_raw, result["labels"], centers_raw, feature_names, figures_dir)
    plot_cluster_profiles(centers_raw, feature_names, X_raw, figures_dir)
    
    # =========================================================================
    print_section("3. FUZZY MEMBERSHIP FUNCTION CONSTRUCTION")
    # =========================================================================
    
    print(f"\nConstructing {args.mf_type} membership functions from cluster centers...")
    
    membership_params = construct_membership_functions(
        centers_raw, X_raw, feature_names, mf_type=args.mf_type
    )
    
    # Save parameters
    save_membership_params(membership_params, output_dir / f"membership_params_{args.mf_type}.json")
    
    # Also save as CSV for easy viewing
    mf_df = membership_params_to_dataframe(membership_params)
    mf_df.to_csv(output_dir / f"membership_params_{args.mf_type}.csv", index=False)
    
    # Print example
    print(f"\nExample {args.mf_type} membership function parameters (first feature):")
    first_feature = feature_names[0]
    for i, mf in enumerate(membership_params[first_feature]):
        if args.mf_type == "triangular":
            print(f"  Set {i+1}: left={mf['left']:.3f}, peak={mf['peak']:.3f}, right={mf['right']:.3f}")
        elif args.mf_type == "gaussian":
            print(f"  Set {i+1}: mean={mf['mean']:.3f}, sigma={mf['sigma']:.3f}")
        elif args.mf_type == "trapezoidal":
            print(f"  Set {i+1}: a={mf['a']:.3f}, b={mf['b']:.3f}, c={mf['c']:.3f}, d={mf['d']:.3f}")
    
    # Generate comparison of all MF types if requested
    if args.compare_mf:
        print("\nGenerating MF type comparison for all three types...")
        triangular_params = construct_membership_functions(centers_raw, X_raw, feature_names, "triangular")
        gaussian_params = construct_membership_functions(centers_raw, X_raw, feature_names, "gaussian")
        trapezoidal_params = construct_membership_functions(centers_raw, X_raw, feature_names, "trapezoidal")
        
        # Save all types
        save_membership_params(triangular_params, output_dir / "membership_params_triangular.json")
        save_membership_params(gaussian_params, output_dir / "membership_params_gaussian.json")
        save_membership_params(trapezoidal_params, output_dir / "membership_params_trapezoidal.json")
        
        # Combine into dict for panel plots
        all_mf_params = {
            "triangular": triangular_params,
            "gaussian": gaussian_params,
            "trapezoidal": trapezoidal_params,
        }
        
        # Generate panel plots for key features (side-by-side comparison)
        key_features = ["alcohol", "volatile acidity", "fixed acidity", "pH", "sulphates"]
        print(f"\n  Generating MF type panel plots for: {key_features}")
        for feature in key_features:
            if feature in feature_names:
                idx = feature_names.index(feature)
                plot_mf_types_panel(feature, all_mf_params, X_raw, idx, figures_dir)
        
        # Generate all-features panel for each MF type
        print("\n  Generating all-features panel plots per MF type...")
        plot_all_features_panel(triangular_params, X_raw, feature_names, figures_dir)
        plot_all_features_panel(gaussian_params, X_raw, feature_names, figures_dir)
        plot_all_features_panel(trapezoidal_params, X_raw, feature_names, figures_dir)
    
    # =========================================================================
    print_section("4. VISUALIZATION")
    # =========================================================================
    
    print(f"\nGenerating {args.mf_type} membership function plots...")
    suffix = f"_{args.mf_type}" if args.mf_type != "triangular" else ""
    plot_all_membership_functions(
        membership_params, X_raw, feature_names, figures_dir, suffix=suffix
    )
    
    # =========================================================================
    print_section("5. QUALITY ANALYSIS (External Validation)")
    # =========================================================================
    
    if y_quality is not None:
        analyze_clusters_with_quality(result["labels"], y_quality, output_dir)
        plot_quality_by_cluster(result["labels"], y_quality, figures_dir / "quality_by_cluster.png")
    
    # =========================================================================
    print_section("6. PDF GENERATION")
    # =========================================================================
    
    print("\nGenerating PDF versions of all figures...")
    generate_all_pdfs(
        membership_params=membership_params,
        X_raw=X_raw,
        feature_names=feature_names,
        inertia_df=inertia_df if not args.skip_sweep else pd.DataFrame({"k": [args.k], "inertia": [result["inertia"]]}),
        centers_raw=centers_raw,
        labels=result["labels"],
        y_quality=y_quality,
        output_dir=figures_dir
    )
    
    # =========================================================================
    print_section("7. SUMMARY")
    # =========================================================================
    
    # Save run configuration
    config = {
        "k": args.k,
        "seed": args.seed,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "n_samples": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "feature_names": feature_names,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Pipeline complete!")
    print(f"\nOutputs saved to:")
    print(f"  - Data: {output_dir}/")
    print(f"  - Figures: {figures_dir}/")
    print(f"\nKey files:")
    print(f"  - Cluster centers: {output_dir}/centers_raw.csv")
    print(f"  - Membership parameters: {output_dir}/membership_params.json")
    print(f"  - Quality analysis: {output_dir}/cluster_quality_summary.csv")
    print(f"  - Membership plots: {figures_dir}/membership_*.png")
    print(f"  - Combined PDF: {figures_dir}/all_figures.pdf")


if __name__ == "__main__":
    main()
