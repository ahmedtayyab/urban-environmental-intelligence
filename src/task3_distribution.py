"""Task 3: Distribution Modeling and Tail Integrity Analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class DistributionAnalyzer:
    """Analyze PM2.5 distributions with focus on tail behavior."""
    
    def __init__(self, data: np.ndarray):
        """Initialize with PM2.5 values."""
        self.data = data.astype(float)
        self.data = self.data[~np.isnan(self.data)]  # Remove NaNs
    
    def calculate_percentiles(self) -> Dict[str, float]:
        """Calculate key percentiles."""
        return {
            "p01": np.percentile(self.data, 1),
            "p05": np.percentile(self.data, 5),
            "p25": np.percentile(self.data, 25),
            "p50": np.percentile(self.data, 50),  # Median
            "p75": np.percentile(self.data, 75),
            "p95": np.percentile(self.data, 95),
            "p99": np.percentile(self.data, 99),
            "p99.9": np.percentile(self.data, 99.9),
        }
    
    def fit_distributions(self) -> Dict[str, Dict]:
        """
        Fit multiple distributions to detect best fit.
        
        Tests: normal, gamma (more robust for environmental data)
        """
        results = {}
        
        # Normal distribution
        mu, sigma = stats.norm.fit(self.data)
        try:
            ks_stat, ks_pval = stats.kstest(self.data, "norm", args=(mu, sigma))
        except:
            ks_stat, ks_pval = np.nan, np.nan
        results["normal"] = {
            "params": {"mu": mu, "sigma": sigma},
            "ks_stat": ks_stat,
            "ks_pval": ks_pval
        }
        
        # Gamma distribution (better for right-skewed environmental data)
        try:
            shape, loc, scale = stats.gamma.fit(self.data)
            ks_stat, ks_pval = stats.kstest(self.data, "gamma", args=(shape, loc, scale))
            results["gamma"] = {
                "params": {"shape": shape, "scale": scale, "loc": loc},
                "ks_stat": ks_stat,
                "ks_pval": ks_pval
            }
        except Exception as e:
            results["gamma"] = {"error": str(e)}
        
        return results


def select_industrial_zone(df: pd.DataFrame,
                           zone_col: str = "zone",
                           value_col: str = "PM2.5") -> Tuple[np.ndarray, str]:
    """
    Select a representative industrial zone with substantial data.
    
    Returns: (PM2.5 array, location_name)
    """
    # Filter for industrial zones
    industrial = df[df[zone_col] == "Industrial"].copy()
    
    # Find location with most data points
    location_counts = industrial.groupby("location_id").size()
    best_location = location_counts.idxmax()
    
    zone_data = industrial[industrial["location_id"] == best_location][value_col].values
    
    return zone_data, f"Industrial Zone (Location {best_location})"


def plot_distribution_peaks(data: np.ndarray,
                           bins: int = 40,
                           title: str = "PM2.5 Distribution: Peak Optimization",
                           percentiles: Dict = None,
                           filepath: str = None) -> plt.Figure:
    """
    Histogram optimized for revealing PEAKS (main distribution shape).
    
    Linear scale, moderate bin count for main distribution clarity.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Histogram with linear scale
    n, bins_edges, patches = ax.hist(
        data,
        bins=bins,
        color="#3498db",
        alpha=0.75,
        edgecolor="#2c3e50",
        linewidth=0.5
    )
    
    ax.set_xlabel("PM2.5 (µg/m³)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    
    # Add percentile lines
    if percentiles:
        ax.axvline(percentiles["p50"], color="#e74c3c", linestyle="--", 
                  linewidth=1.5, label=f"Median: {percentiles['p50']:.1f}", alpha=0.8)
        ax.axvline(percentiles["p95"], color="#f39c12", linestyle="--", 
                  linewidth=1.5, label=f"95th %ile: {percentiles['p95']:.1f}", alpha=0.8)
        ax.axvline(percentiles["p99"], color="#c0392b", linestyle="--", 
                  linewidth=2, label=f"99th %ile: {percentiles['p99']:.1f}", alpha=0.9)
    
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.3, axis="y")
    
    # Statistics box
    stats_text = f"n={len(data)}\nMean={np.mean(data):.1f}\nStd={np.std(data):.1f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment="top", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_distribution_tails(data: np.ndarray,
                           bins: int = 80,
                           title: str = "PM2.5 Distribution: Tail Optimization (Log Scale)",
                           percentiles: Dict = None,
                           filepath: str = None) -> plt.Figure:
    """
    Histogram optimized for revealing TAILS (rare, extreme values).
    
    Log scale on y-axis, more bins to show tail granularity.
    Reveals rare extreme events that linear plots obscure.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Histogram with log scale
    n, bins_edges, patches = ax.hist(
        data,
        bins=bins,
        color="#e74c3c",
        alpha=0.75,
        edgecolor="#2c3e50",
        linewidth=0.5
    )
    
    ax.set_yscale("log")
    ax.set_xlabel("PM2.5 (µg/m³)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (log scale)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    
    # Add percentile lines
    if percentiles:
        ax.axvline(percentiles["p50"], color="#3498db", linestyle="--", 
                  linewidth=1.5, label=f"Median: {percentiles['p50']:.1f}", alpha=0.8)
        ax.axvline(percentiles["p95"], color="#f39c12", linestyle="--", 
                  linewidth=1.5, label=f"95th %ile: {percentiles['p95']:.1f}", alpha=0.8)
        ax.axvline(percentiles["p99"], color="#c0392b", linestyle="--", 
                  linewidth=2, label=f"99th %ile: {percentiles['p99']:.1f}", alpha=0.9)
        
        # Shade extreme region
        ax.axvspan(percentiles["p99"], data.max(), alpha=0.1, color="red")
    
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.3, axis="y")
    
    # Statistics box
    extreme_count = (data > percentiles["p99"]).sum() if percentiles else 0
    stats_text = f"n={len(data)}\nMean={np.mean(data):.1f}\nStd={np.std(data):.1f}\nExtreme (>P99): {extreme_count}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment="top", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_qq_plot(data: np.ndarray,
                title: str = "Q-Q Plot: Normality Assessment",
                filepath: str = None) -> plt.Figure:
    """
    Q-Q plot to assess distribution fit and tail behavior.
    
    Deviations from the line indicate non-normality.
    Upward curve at tail indicates heavier upper tail than normal.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(data, dist="norm", plot=ax)
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Theoretical Quantiles", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sample Quantiles (PM2.5)", fontsize=10, fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.3)
    
    # Add annotation
    ax.text(0.05, 0.95, "Upward curve in tail\nindicates heavier tail\nthan normal distribution",
           transform=ax.transAxes, fontsize=9,
           verticalalignment="top",
           bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3, edgecolor="none"))
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_tail_focus(data: np.ndarray,
                   percentile_start: float = 95,
                   title: str = "Extreme Events: P95+ Tail Detail",
                   filepath: str = None) -> plt.Figure:
    """
    Focused histogram on extreme tail (P95 and above).
    
    Zooms into the dangerous region where extreme hazard events occur.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    p95 = np.percentile(data, percentile_start)
    tail_data = data[data >= p95]
    
    ax.hist(
        tail_data,
        bins=30,
        color="#c0392b",
        alpha=0.8,
        edgecolor="#7f1d12",
        linewidth=0.5
    )
    
    ax.set_xlabel("PM2.5 (µg/m³)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    
    # Mark key thresholds
    p99 = np.percentile(data, 99)
    p999 = np.percentile(data, 99.9)
    extreme_threshold = 200  # Extreme hazard threshold
    
    ax.axvline(p99, color="#3498db", linestyle="--", linewidth=1.5, 
              label=f"99th %ile: {p99:.0f}", alpha=0.8)
    if extreme_threshold in data or (data > extreme_threshold).any():
        ax.axvline(extreme_threshold, color="#e74c3c", linestyle="--", linewidth=2, 
                  label=f"Hazard threshold: {extreme_threshold}", alpha=0.9)
    
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.3, axis="y")
    
    tail_stats = f"Extreme events (>{p95:.0f}): {len(tail_data)}\nMax: {tail_data.max():.1f}"
    ax.text(0.05, 0.95, tail_stats, transform=ax.transAxes,
           fontsize=9, verticalalignment="top",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig
