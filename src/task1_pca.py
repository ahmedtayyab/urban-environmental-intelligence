"""Task 1: Dimensionality Reduction with PCA."""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from src.visualization import setup_figure, VizConfig


class PCAAnalyzer:
    """PCA-based dimensionality reduction and analysis."""
    
    def __init__(self, n_components: int = 2):
        """Initialize PCA with specified number of components."""
        self.n_components = n_components
        self.pca = None
        self.scaler = None
        self.feature_names = None
        self.explained_variance_ratio = None
    
    def fit(self, df: pd.DataFrame, feature_cols: list) -> "PCAAnalyzer":
        """
        Fit PCA on feature columns.
        
        Parameters:
        - df: Input DataFrame
        - feature_cols: List of column names to use
        """
        self.feature_names = feature_cols
        X = df[feature_cols].values
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        return self
    
    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Transform data using fitted PCA."""
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        
        return pca_df
    
    def get_loadings(self) -> pd.DataFrame:
        """
        Get PCA loadings (contributions of original variables).
        
        Returns DataFrame with variables as rows, PCs as columns.
        """
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
            index=self.feature_names
        )
        
        return loadings
    
    def get_explained_variance(self) -> Dict[str, float]:
        """Get cumulative explained variance for each PC."""
        cumsum = np.cumsum(self.explained_variance_ratio)
        result = {}
        for i, val in enumerate(cumsum):
            result[f"PC{i+1}"] = val
        return result


def analyze_pca(df_processed: pd.DataFrame, 
                raw_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run PCA analysis on processed data.
    
    Parameters:
    - df_processed: Standardized feature data
    - raw_df: Optional raw features for reference
    
    Returns:
    - PCA-transformed DataFrame with zone labels
    - Analysis results dictionary
    """
    feature_cols = ["PM2.5", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]
    
    print("Running PCA analysis...")
    
    # Initialize and fit PCA
    analyzer = PCAAnalyzer(n_components=2)
    analyzer.fit(df_processed, feature_cols)
    
    # Transform data
    pca_df = analyzer.transform(df_processed, feature_cols)
    
    # Add metadata
    pca_df["zone"] = df_processed["zone"].values
    pca_df["location_id"] = df_processed["location_id"].values
    
    if raw_df is not None:
        pca_df["PM2.5_raw"] = raw_df["PM2.5"].values
    
    # Get analysis results
    loadings = analyzer.get_loadings()
    explained_var = analyzer.get_explained_variance()
    
    print(f"PC1 explained variance: {explained_var['PC1']:.1%}")
    print(f"PC2 explained variance: {explained_var['PC2']:.1%}")
    print(f"Total explained variance: {explained_var['PC2']:.1%}")
    
    print("\nPCA Loadings (variable contributions):")
    print(loadings)
    
    results = {
        "analyzer": analyzer,
        "loadings": loadings,
        "explained_variance": explained_var,
        "feature_cols": feature_cols
    }
    
    return pca_df, results


def plot_pca_scatter_clean(pca_df: pd.DataFrame,
                           explained_var: Dict[str, float],
                           title: str = "PCA: Environmental Variables by Zone",
                           filepath: str = None) -> plt.Figure:
    """
    Plot PCA results with zones highlighted, minimal chart junk.
    """
    fig, ax = setup_figure(figsize=(9, 6))
    
    zones = pca_df["zone"].unique()
    colors = {"Industrial": "#e74c3c", "Residential": "#3498db"}
    
    for zone in zones:
        mask = pca_df["zone"] == zone
        ax.scatter(
            pca_df.loc[mask, "PC1"],
            pca_df.loc[mask, "PC2"],
            alpha=0.5,
            s=35,
            label=zone,
            color=colors.get(zone, None),
            edgecolors="none"
        )
    
    # Format variance labels
    pc1_var = explained_var.get("PC1", 0.0)
    pc2_var = explained_var.get("PC2", 0.0)
    
    ax.set_xlabel(f"PC1 ({pc1_var:.1%} variance)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pc2_var:.1%} variance)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
    
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_loading_heatmap(loadings_df: pd.DataFrame,
                         title: str = "PCA Loadings: Variable Contributions",
                         filepath: str = None) -> plt.Figure:
    """
    Plot PCA loadings as heatmap.
    
    Shows which original variables drive the principal components.
    Positive values: variable increases with PC
    Negative values: variable decreases with PC
    """
    fig, ax = setup_figure(figsize=(6, 4))
    
    # Create heatmap
    sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Loading"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=False
    )
    
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Principal Component", fontsize=10, fontweight="bold")
    ax.set_ylabel("Environmental Variable", fontsize=10, fontweight="bold")
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_loading_arrows(pca_df: pd.DataFrame,
                       loadings_df: pd.DataFrame,
                       explained_var: Dict[str, float],
                       title: str = "PCA Biplot: Variables & Zones",
                       scale_factor: float = 3.5,
                       filepath: str = None) -> plt.Figure:
    """
    Biplot showing both data points and variable vectors.
    
    Arrows indicate direction and magnitude of original variables
    in PC space.
    """
    fig, ax = setup_figure(figsize=(10, 7))
    
    # Plot zones
    zones = pca_df["zone"].unique()
    colors = {"Industrial": "#e74c3c", "Residential": "#3498db"}
    
    for zone in zones:
        mask = pca_df["zone"] == zone
        ax.scatter(
            pca_df.loc[mask, "PC1"],
            pca_df.loc[mask, "PC2"],
            alpha=0.3,
            s=20,
            label=zone,
            color=colors.get(zone, None),
            edgecolors="none"
        )
    
    # Plot loading vectors (arrows)
    for var in loadings_df.index:
        pc1_load = loadings_df.loc[var, "PC1"] * scale_factor
        pc2_load = loadings_df.loc[var, "PC2"] * scale_factor
        
        ax.arrow(
            0, 0, pc1_load, pc2_load,
            head_width=0.15,
            head_length=0.15,
            fc="black",
            ec="black",
            linewidth=1,
            alpha=0.7
        )
        
        # Label arrow
        ax.text(
            pc1_load * 1.15,
            pc2_load * 1.15,
            var,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none")
        )
    
    pc1_var = explained_var.get("PC1", 0.0)
    pc2_var = explained_var.get("PC2", 0.0)
    
    ax.set_xlabel(f"PC1 ({pc1_var:.1%} variance)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pc2_var:.1%} variance)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
    
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig
