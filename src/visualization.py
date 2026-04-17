"""Visualization utilities with emphasis on data-ink ratio."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Configure matplotlib for minimal ink
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["lines.linewidth"] = 0.8


class VizConfig:
    """Configuration for clean, honest visualizations."""
    
    # Sequential colormap for data integrity
    CMAP_SEQUENTIAL = "YlOrRd"
    
    # Dark background for reduced chart junk
    STYLE = "whitegrid"
    
    # Minimal grid settings
    GRID_ALPHA = 0.2
    GRID_WIDTH = 0.3


def setup_figure(figsize: Tuple[int, int] = (10, 6)):
    """Create a clean figure with minimal chart junk."""
    sns.set_style(VizConfig.STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Reduce grid opacity
    ax.grid(alpha=VizConfig.GRID_ALPHA, linewidth=VizConfig.GRID_WIDTH)
    
    return fig, ax


def plot_pca_scatter(pca_df: pd.DataFrame, 
                     pc1: str = "PC1", pc2: str = "PC2",
                     hue: str = "zone",
                     title: str = "PCA: Environmental Variables",
                     filepath: str = None) -> plt.Figure:
    """
    Plot PCA results with zones highlighted.
    
    Parameters:
    - pca_df: DataFrame with PC1, PC2, and zone columns
    - hue: Column to color by (typically 'zone')
    - title: Plot title
    - filepath: Save to file if provided
    """
    fig, ax = setup_figure(figsize=(8, 6))
    
    zones = pca_df[hue].unique()
    colors = {"Industrial": "#e74c3c", "Residential": "#3498db"}
    
    for zone in zones:
        mask = pca_df[hue] == zone
        ax.scatter(
            pca_df.loc[mask, pc1],
            pca_df.loc[mask, pc2],
            alpha=0.6,
            s=30,
            label=zone,
            color=colors.get(zone, None),
            edgecolors="none"
        )
    
    ax.set_xlabel(f"{pc1} (Explained Variance: TBD%)", fontsize=10)
    ax.set_ylabel(f"{pc2} (Explained Variance: TBD%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best", frameon=False)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    return fig


def plot_loading_heatmap(loadings: pd.DataFrame,
                         title: str = "PCA Loadings",
                         filepath: str = None) -> plt.Figure:
    """
    Plot PCA loadings as heatmap to show variable contributions.
    
    Parameters:
    - loadings: DataFrame with variables as rows and PCs as columns
    - title: Plot title
    - filepath: Save to file if provided
    """
    fig, ax = setup_figure(figsize=(6, 4))
    
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Loading"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )
    
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Environmental Variable", fontsize=10)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    return fig


def plot_heatmap_temporal(df: pd.DataFrame,
                         value_col: str = "PM2.5",
                         time_col: str = "date",
                         station_col: str = "location_id",
                         title: str = "Temporal PM2.5 Patterns",
                         filepath: str = None,
                         max_stations: int = 50) -> plt.Figure:
    """
    Plot high-density temporal data as heatmap (raster plot).
    
    One row = one station, columns = time periods
    """
    df = df.copy()
    
    # Limit stations for readability
    if len(df[station_col].unique()) > max_stations:
        top_stations = df[station_col].value_counts().head(max_stations).index
        df = df[df[station_col].isin(top_stations)]
    
    # Create pivot table
    pivot = df.pivot_table(
        index=station_col,
        columns=time_col,
        values=value_col,
        aggfunc="mean"
    )
    
    fig, ax = setup_figure(figsize=(14, 6))
    
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=VizConfig.CMAP_SEQUENTIAL,
        interpolation="nearest"
    )
    
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Station ID", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{value_col} (µg/m³)", fontsize=9)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    return fig


def plot_distribution_histogram(df: pd.DataFrame,
                                value_col: str = "PM2.5",
                                bins: int = 50,
                                title: str = "PM2.5 Distribution (Peaks)",
                                filepath: str = None) -> plt.Figure:
    """
    Histogram showing distribution peaks.
    """
    fig, ax = setup_figure(figsize=(8, 5))
    
    ax.hist(df[value_col], bins=bins, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel(f"{value_col} (µg/m³)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Add percentile line
    p99 = df[value_col].quantile(0.99)
    ax.axvline(p99, color="red", linestyle="--", linewidth=1.5, label=f"99th percentile: {p99:.1f}")
    ax.legend(loc="upper right", frameon=False)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    return fig


def plot_distribution_logscale(df: pd.DataFrame,
                              value_col: str = "PM2.5",
                              bins: int = 50,
                              title: str = "PM2.5 Distribution (Tails, log scale)",
                              filepath: str = None) -> plt.Figure:
    """
    Histogram with log scale to reveal tail behavior.
    """
    fig, ax = setup_figure(figsize=(8, 5))
    
    ax.hist(df[value_col], bins=bins, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    
    ax.set_xlabel(f"{value_col} (µg/m³)", fontsize=10)
    ax.set_ylabel("Frequency (log scale)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    # Add percentile line
    p99 = df[value_col].quantile(0.99)
    ax.axvline(p99, color="blue", linestyle="--", linewidth=1.5, label=f"99th percentile: {p99:.1f}")
    ax.legend(loc="upper right", frameon=False)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    return fig


def plot_small_multiples(df: pd.DataFrame,
                        value_col: str = "PM2.5",
                        region_col: str = "city",
                        pop_col: str = "population",
                        title: str = "Pollution vs Population by Region",
                        filepath: str = None,
                        max_regions: int = 12) -> plt.Figure:
    """
    Small multiples approach for 3-variable comparison.
    """
    regions = df[region_col].unique()[:max_regions]
    
    n_cols = 3
    n_rows = (len(regions) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, region in enumerate(regions):
        ax = axes[idx]
        region_data = df[df[region_col] == region]
        
        scatter = ax.scatter(
            region_data[pop_col],
            region_data[value_col],
            c=region_data[value_col],
            cmap=VizConfig.CMAP_SEQUENTIAL,
            alpha=0.6,
            s=50,
            edgecolors="none"
        )
        
        ax.set_xlabel("Population Density", fontsize=8)
        ax.set_ylabel(f"{value_col} (µg/m³)", fontsize=8)
        ax.set_title(region, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.2, linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Hide unused subplots
    for idx in range(len(regions), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
    return fig
