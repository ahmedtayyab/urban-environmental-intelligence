"""Task 4: Visual Integrity Audit - 3D Rejection & Alt Solutions."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class ThreeDAnalyzer:
    """Analyze why 3D charts violate data integrity principles."""
    
    @staticmethod
    def calculate_lie_factor(true_value: float, visual_value: float) -> float:
        """
        Calculate Lie Factor per Tufte's definition.
        
        Lie Factor = (visual magnitude) / (data magnitude)
        - 1.0 = honest representation
        - >1.5 = exaggeration (misleading)
        - <0.5 = minimization (misleading)
        """
        if true_value == 0:
            return np.nan
        return visual_value / true_value
    
    @staticmethod
    def analyze_3d_distortion() -> Dict:
        """Analyze distortion effects in 3D bar charts."""
        return {
            "perspective_distortion": {
                "description": "Back values appear smaller than front values",
                "impact": "5-40% visual distortion depending on viewing angle",
                "visibility_issue": "Blocks view of back bars entirely"
            },
            "occlusion": {
                "description": "Front bars occlude (hide) back bars",
                "impact": "Impossible to compare all values simultaneously",
                "lie_factor_range": "2.0 to 5.0+ (severe)"
            },
            "color_gradients": {
                "description": "3D shading creates false luminance gradients",
                "impact": "Brain perceives shaded sides as separate data",
                "confusion": "Is the darkness representing value or just shading?"
            },
            "axis_length_distortion": {
                "description": "3D perspective makes axis lengths appear non-linear",
                "impact": "Equal intervals don't look equal",
                "lie_factor_range": "1.5 to 3.0"
            }
        }


def prepare_regional_data(df: pd.DataFrame,
                         max_regions: int = 8) -> pd.DataFrame:
    """
    Prepare data for 3-variable visualization.
    
    Creates: region, pollution_level, population_density aggregates
    """
    df = df.copy()
    
    # Ensure PM2.5 is numeric
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["PM2.5"])
    
    # Create synthetic population density linked to zone type
    df["population_density"] = (
        df["zone"].map({"Industrial": 150, "Residential": 200}) +
        np.random.normal(0, 20, len(df))
    )
    
    # Aggregate by location
    regional = df.groupby("location_id").agg({
        "PM2.5": "mean",
        "zone": lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        "population_density": "mean",
    }).reset_index(drop=True)
    
    regional.columns = ["pollution", "zone", "pop_density"]
    regional["region_id"] = range(1, len(regional) + 1)
    regional["region"] = [f"Region {i}" for i in regional["region_id"]]
    
    # Ensure numeric types
    regional["pollution"] = pd.to_numeric(regional["pollution"], errors="coerce")
    regional["pop_density"] = pd.to_numeric(regional["pop_density"], errors="coerce")
    regional = regional.dropna(subset=["pollution", "pop_density"])
    
    # Select top regions for clarity
    regional = regional.nlargest(max_regions, "pollution").reset_index(drop=True)
    
    return regional


def plot_3d_bar_chart_demo(data: pd.DataFrame,
                          filepath: str = None):
    """
    PROPOSAL: 3D bar chart (to be REJECTED).
    
    Demonstrates the problems with 3D visualization.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    x = np.arange(len(data))
    y = data["pop_density"].values
    z = data["pollution"].values
    
    # Create 3D bars
    xpos, ypos = np.meshgrid(x, [0]*len(data))
    dx = 0.4 * np.ones_like(xpos[0])
    dy = 0.4 * np.ones_like(ypos[0])
    dz = z
    
    colors = plt.cm.YlOrRd(z / z.max())
    ax.bar3d(x, y, 0, dx, dy, dz, color=colors, shade=True, alpha=0.8)
    
    ax.set_xlabel("Region", fontsize=10, fontweight="bold")
    ax.set_ylabel("Population Density", fontsize=10, fontweight="bold")
    ax.set_zlabel("PM2.5 Pollution", fontsize=10, fontweight="bold")
    ax.set_title("REJECTED: 3D Bar Chart Proposal\n(Violates Data Integrity)", 
                fontsize=11, fontweight="bold", color="red")
    
    ax.set_xticks(x)
    ax.set_xticklabels(data["region"].values, rotation=45, ha="right")
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_bivariate_mapping(data: pd.DataFrame,
                          x_col: str = "pop_density",
                          y_col: str = "pollution",
                          color_col: str = "pollution",
                          title: str = "Bivariate Mapping: Pollution vs Population Density × Region",
                          cmap: str = "YlOrRd",
                          filepath: str = None) -> plt.Figure:
    """
    SOLUTION 1: Bivariate Mapping (Bubble/Scatter chart).
    
    3 variables encoded:
    - X-axis: Population Density
    - Y-axis: Pollution Level
    - Color: Pollution Level (redundant but strengthens perception)
    - Labels: Region names
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Create scatter plot
    scatter = ax.scatter(
        data[x_col],
        data[y_col],
        c=data[color_col],
        s=200,
        cmap=cmap,
        alpha=0.7,
        edgecolors="#2c3e50",
        linewidth=1.5
    )
    
    # Add region labels
    for idx, row in data.iterrows():
        ax.annotate(
            row["region"],
            (row[x_col], row[y_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            alpha=0.8
        )
    
    ax.set_xlabel("Population Density (people/km²)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PM2.5 Pollution (µg/m³)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("PM2.5 (µg/m³)", fontsize=10, fontweight="bold")
    
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_small_multiples(data: pd.DataFrame,
                        x_col: str = "pop_density",
                        y_col: str = "pollution",
                        color_col: str = "zone",
                        title: str = "Small Multiples: Pollution × Population across Regions",
                        max_cols: int = 4,
                        filepath: str = None) -> plt.Figure:
    """
    SOLUTION 2: Small Multiples approach.
    
    One mini-chart per region:
    - X: Population Density
    - Y: Pollution
    - Color: Zone type (Industrial/Residential)
    - Title: Region name
    
    Enables detailed comparison within familiar layout.
    """
    n_regions = len(data)
    n_cols = min(max_cols, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten()
    
    colors = {"Industrial": "#e74c3c", "Residential": "#3498db"}
    
    for idx, (ax_idx, (_, row)) in enumerate(zip(range(len(axes)), data.iterrows())):
        ax = axes[ax_idx]
        
        # Single bar showing pollution with color for zone
        zone = row["zone"]
        bar_color = colors.get(zone, "#95a5a6")
        
        ax.barh(
            [0],
            [row[y_col]],
            color=bar_color,
            alpha=0.7,
            edgecolor="#2c3e50",
            linewidth=1.5,
            height=0.6
        )
        
        # Add text annotations
        ax.text(
            row[y_col] + 2,
            0,
            f"{row[y_col]:.1f}",
            va="center",
            fontsize=9,
            fontweight="bold"
        )
        
        # Region as title
        ax.set_title(row["region"], fontsize=10, fontweight="bold", pad=8)
        ax.set_xlim(0, data[y_col].max() * 1.1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("PM2.5 (µg/m³)", fontsize=8)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linewidth=0.3, axis="x")
        
        # Zone label
        zone_label = f"Zone: {zone}"
        ax.text(0.98, 0.98, zone_label, transform=ax.transAxes,
               fontsize=8, verticalalignment="top", horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
    
    # Hide unused subplots
    for idx in range(len(data), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def plot_heatmap_grid(data: pd.DataFrame,
                     value_col: str = "pollution",
                     title: str = "Heatmap: Pollution Levels by Region & Zone",
                     cmap: str = "YlOrRd",
                     filepath: str = None) -> plt.Figure:
    """
    ALTERNATIVE: Heatmap grid encoding region × zone.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Reshape for heatmap: rows=regions, single column=pollution
    heatmap_data = data[[("region", "pollution")]].copy() if isinstance(data.columns, pd.MultiIndex) else data[["region", value_col]].copy()
    if isinstance(heatmap_data, pd.DataFrame):
        heatmap_data = heatmap_data.set_index("region") if "region" in heatmap_data.columns else heatmap_data
    
    # Create simple matrix
    heatmap_matrix = data[[value_col]].T
    heatmap_matrix.columns = data["region"].values
    
    im = ax.imshow(
        heatmap_matrix.values,
        cmap=cmap,
        aspect="auto",
        vmin=data[value_col].min(),
        vmax=data[value_col].max()
    )
    
    ax.set_yticks([0])
    ax.set_yticklabels(["Pollution"])
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["region"].values, rotation=45, ha="right")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PM2.5 (µg/m³)", fontsize=10, fontweight="bold")
    
    # Add text values
    for i, region in enumerate(data["region"].values):
        ax.text(i, 0, f"{data.iloc[i][value_col]:.0f}",
               ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig


def create_color_scale_comparison(title: str = "Color Scale Psychology: Sequential vs Rainbow",
                                  filepath: str = None) -> plt.Figure:
    """
    Compare perceptual properties of sequential vs rainbow colormaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 3))
    
    # Sequential colormap (YlOrRd)
    sequential_cmap = plt.cm.YlOrRd
    sequential_gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    ax = axes[0]
    im = ax.imshow(sequential_gradient, cmap=sequential_cmap, aspect="auto")
    ax.set_title("Sequential: YlOrRd\n(Recommended for Pollution)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Luminance", fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_xticklabels(["Low", "", "Medium", "", "High"])
    
    # Add analysis text
    analysis_text = """
    LUMINANCE PROGRESSION:
    • Light (yellow) → Dark (red)
    • Monotonic increase
    • Brain: brighter = more severe
    
    ADVANTAGES:
    ✓ Intuitive mapping to pollution
    ✓ Colorblind-friendly (with care)
    ✓ Prints well in black/white
    ✓ Uniform perceptual spacing
    
    PERCEPTION:
    Viewers immediately understand:
    Yellow = low risk
    Red = HIGH RISK
    """
    
    axes[0].text(0.98, -0.35, analysis_text, transform=axes[0].transAxes,
                fontsize=8, verticalalignment="top", horizontalalignment="right",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7, edgecolor="none"))
    
    # Rainbow colormap (Jet) - PROBLEMATIC
    rainbow_cmap = plt.cm.jet
    rainbow_gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    ax = axes[1]
    im = ax.imshow(rainbow_gradient, cmap=rainbow_cmap, aspect="auto")
    ax.set_title("Rainbow: Jet\n(NOT Recommended)", fontsize=11, fontweight="bold", color="red")
    ax.set_ylabel("Perceptual Value", fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_xticklabels(["Low", "", "Medium", "", "High"])
    
    # Add analysis text
    analysis_text2 = """
    LUMINANCE PROBLEMS:
    • Blue (dark) → Bright → Red
    • Non-monotonic luminance
    • Brain confused: is blue low or high?
    
    DISADVANTAGES:
    ✗ Counterintuitive for pollution
    ✗ Middle (green) looks neutral/safe
    ✗ Poor for colorblind viewers
    ✗ Non-uniform perceptual spacing
    ✗ Creates false features (bright spots)
    
    WHY AVOIDED:
    Rainbow has 3 brightness peaks
    (cyan, green, yellow) = artifacts!
    Viewers see patterns that don't exist
    """
    
    axes[1].text(0.98, -0.35, analysis_text2, transform=axes[1].transAxes,
                fontsize=8, verticalalignment="top", horizontalalignment="right",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5, edgecolor="none"))
    
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    
    return fig
