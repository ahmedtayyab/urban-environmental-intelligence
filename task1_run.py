"""Task 1: Dimensionality Challenge - Execute and Visualize."""
import pandas as pd
from pathlib import Path
from src.task1_pca import analyze_pca, plot_pca_scatter_clean, plot_loading_heatmap, plot_loading_arrows


def main():
    """Execute Task 1: PCA analysis on environmental variables."""
    
    project_root = Path(__file__).parent
    processed_data_path = project_root / "data" / "processed" / "prepared_data.parquet"
    raw_features_path = project_root / "data" / "processed" / "raw_features.parquet"
    viz_dir = project_root / "visualizations" / "task1"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TASK 1: THE DIMENSIONALITY CHALLENGE - PCA ANALYSIS")
    print("="*70)
    
    # Load processed data
    print("\nLoading processed data...")
    df_processed = pd.read_parquet(processed_data_path)
    df_raw_features = pd.read_parquet(raw_features_path)
    
    print(f"Data shape: {df_processed.shape}")
    print(f"Features: {df_processed.columns.tolist()}")
    
    # Run PCA analysis
    print("\n" + "-"*70)
    print("DIMENSIONALITY REDUCTION: PCA")
    print("-"*70)
    
    pca_df, results = analyze_pca(df_processed, df_raw_features)
    
    # Extract results
    loadings = results["loadings"]
    explained_var = results["explained_variance"]
    
    # Create visualizations
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    # 1. PCA Scatter plot (main viz)
    print("\n1. PCA Scatter Plot: Zone Clustering")
    plot_pca_scatter_clean(
        pca_df,
        explained_var,
        title="PCA: Industrial vs Residential Environmental Profiles",
        filepath=str(viz_dir / "01_pca_scatter_zones.png")
    )
    
    # 2. Loadings heatmap
    print("\n2. PCA Loadings Heatmap: Variable Contributions")
    plot_loading_heatmap(
        loadings,
        title="PCA Loadings: How Variables Drive Principal Components",
        filepath=str(viz_dir / "02_pca_loadings_heatmap.png")
    )
    
    # 3. Biplot (data + variables)
    print("\n3. PCA Biplot: Data Points & Variable Vectors")
    plot_loading_arrows(
        pca_df,
        loadings,
        explained_var,
        title="PCA Biplot: Zones & Environmental Variable Directions",
        filepath=str(viz_dir / "03_pca_biplot.png")
    )
    
    # Analysis summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY: KEY FINDINGS")
    print("="*70)
    
    print("\nExplained Variance by Component:")
    for pc, var in explained_var.items():
        print(f"  {pc}: {var:.1%}")
    
    print("\nVariable Contributions (Loadings):")
    print("\nPC1 (Primary driver):")
    pc1_loadings = loadings["PC1"].abs().sort_values(ascending=False)
    for var, load in pc1_loadings.items():
        direction = "positive" if loadings.loc[var, "PC1"] > 0 else "negative"
        print(f"  {var:15s}: {loadings.loc[var, 'PC1']:+.3f} ({direction})")
    
    print("\nPC2 (Secondary driver):")
    pc2_loadings = loadings["PC2"].abs().sort_values(ascending=False)
    for var, load in pc2_loadings.items():
        direction = "positive" if loadings.loc[var, "PC2"] > 0 else "negative"
        print(f"  {var:15s}: {loadings.loc[var, 'PC2']:+.3f} ({direction})")
    
    # Zone separation analysis
    print("\n" + "-"*70)
    print("ZONE CLUSTERING ANALYSIS")
    print("-"*70)
    
    industrial = pca_df[pca_df["zone"] == "Industrial"]
    residential = pca_df[pca_df["zone"] == "Residential"]
    
    print(f"\nIndustrial zones (n={len(industrial)}):")
    print(f"  PC1 mean: {industrial['PC1'].mean():.3f} ± {industrial['PC1'].std():.3f}")
    print(f"  PC2 mean: {industrial['PC2'].mean():.3f} ± {industrial['PC2'].std():.3f}")
    
    print(f"\nResidential zones (n={len(residential)}):")
    print(f"  PC1 mean: {residential['PC1'].mean():.3f} ± {residential['PC1'].std():.3f}")
    print(f"  PC2 mean: {residential['PC2'].mean():.3f} ± {residential['PC2'].std():.3f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION & JUSTIFICATION")
    print("="*70)
    
    print("""
PCA (Principal Component Analysis) was chosen because:

1. DIMENSIONALITY REDUCTION: Transforms 6 correlated variables into 2
   uncorrelated principal components while preserving maximum variance.

2. LINEAR COMBINATIONS: PCs are weighted combinations of original variables,
   revealing the "axes of greatest variation" in the data.

3. LOADINGS INTERPRETATION: High absolute loadings indicate which original
   variables drive each principal component:
   - PC1 likely dominated by pollution metrics (PM2.5, PM10, NO2)
   - PC2 likely driven by weather variables (Temperature, Humidity)

4. ZONE SEPARATION: Clear clustering shows Industrial vs Residential zones
   have distinct environmental profiles, validating the zone classification.

5. DATA-INK RATIO: Scatter plot uses only essential ink (points, axes, labels)
   with no decorative elements, following Edward Tufte principles.
""")
    
    print("\n" + "="*70)
    print("TASK 1 COMPLETE")
    print("="*70)
    print(f"✓ Visualizations saved to: {viz_dir}")
    print(f"✓ PCA Scatter plot: 01_pca_scatter_zones.png")
    print(f"✓ Loadings heatmap: 02_pca_loadings_heatmap.png")
    print(f"✓ Biplot: 03_pca_biplot.png")
    
    return pca_df, results


if __name__ == "__main__":
    pca_df, results = main()
