import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Configure page with custom theme
st.set_page_config(
    page_title="Urban Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Data Science Assignment - Air Quality Analysis"}
)

# Professional color scheme
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
WARNING_COLOR = "#d62728"
BG_DARK = "#0f1419"
CARD_BG = "#1a1f2e"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#a0a8b8"

# Load preprocessed data
@st.cache_data
def load_data():
    # Try processed first, then fallback to root data folder
    data_path = Path("data/processed/prepared_data.parquet")
    if not data_path.exists():
        data_path = Path("data/prepared_data.parquet")
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
        return df
    else:
        return None

# Load the analysis modules
import sys
sys.path.insert(0, str(Path.cwd()))
from src.data_processor import DataProcessor
from src.task1_pca import PCAAnalyzer
from src.task2_temporal import TemporalAnalyzer
from src.task3_distribution import DistributionAnalyzer
from src.task4_visualization import ThreeDAnalyzer

# Advanced custom styling
st.markdown("""
    <style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --bg-dark: #0f1419;
        --card-bg: #1a1f2e;
        --text-primary: #ffffff;
        --text-secondary: #a0a8b8;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: #0f1419;
        color: #ffffff;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4 !important;
        font-size: 36px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 12px;
    }
    
    h3 {
        color: #e8eef7 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 12px !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: linear-gradient(135deg, #1a1f2e 0%, #232d3d 100%);
        border: 1px solid #2a3142;
        border-radius: 12px;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="metric-container"] > div:first-child {
        color: #a0a8b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    [data-testid="metric-container"] > div:nth-child(2) {
        color: #1f77b4;
        font-size: 32px;
        font-weight: 700;
    }
    
    [data-testid="metric-container"] > div:nth-child(3) {
        color: #2ca02c;
        font-size: 13px;
        margin-top: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #a0a8b8;
        background-color: transparent;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        padding: 12px 24px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        background-color: rgba(31, 119, 180, 0.08);
    }
    
    /* Cards with gradient background */
    .info-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232d3d 100%);
        border: 1px solid #2a3142;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.15);
        transform: translateY(-2px);
    }
    
    .info-card h3 {
        color: #1f77b4;
        margin-top: 0;
    }
    
    .info-card p {
        color: #d0d8e0;
        line-height: 1.6;
        margin: 8px 0;
    }
    
    /* Radio buttons and selectors */
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 12px;
    }
    
    .stRadio [role="radio"] {
        accent-color: #1f77b4;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #0f1419;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f1419;
        border-right: 1px solid #2a3142;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #1f77b4;
        border: none;
        padding: 0;
        margin-bottom: 16px;
    }
    
    /* Text styling */
    p {
        color: #d0d8e0;
        line-height: 1.6;
    }
    
    strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Dividers */
    hr {
        border-color: #2a3142;
        margin: 24px 0;
    }
    
    /* Warning/Error boxes */
    .stAlert {
        background-color: #2a3142;
        border-left: 4px solid #ff7f0e;
        color: #ffffff;
        border-radius: 8px;
    }
    
    /* Code blocks */
    code {
        background-color: #1a1f2e;
        color: #5dd9c1;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    
    /* Tables */
    table {
        color: #d0d8e0;
        font-size: 14px;
    }
    
    thead {
        background-color: #1a1f2e;
        color: #1f77b4;
    }
    
    td {
        border-color: #2a3142;
    }
    
    /* Links */
    a {
        color: #1f77b4;
        text-decoration: none;
    }
    
    a:hover {
        color: #5da3d5;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
<div style='padding: 20px 0;'>
    <h2 style='margin: 0; font-size: 20px;'>🌍 Urban Environmental</h2>
    <p style='color: #a0a8b8; margin: 4px 0 0 0; font-size: 12px;'>Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Analysis Tasks",
    ["Dashboard", "Task 1: PCA", "Task 2: Temporal", "Task 3: Distribution", "Task 4: Visual Integrity"],
    label_visibility="collapsed"
)

def load_visualizations(task_name):
    """Load PNG visualizations from disk for display."""
    viz_dir = Path(f"visualizations/{task_name}")
    if not viz_dir.exists():
        return []
    return sorted([f for f in viz_dir.glob("*.png")])

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "Dashboard":
    # Hero section
    st.markdown("""
    <div style='margin-bottom: 40px;'>
        <h1>🌍 Urban Environmental Intelligence</h1>
        <p style='color: #a0a8b8; font-size: 16px; margin-top: 8px; line-height: 1.6;'>
            Comprehensive analysis of air quality patterns across 100 global locations over 365 days. 
            Exploring pollution dynamics through dimensionality reduction, temporal analysis, distribution modeling, 
            and visual integrity principles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load key metrics
    df = load_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Records", f"{len(df):,}", "observations")
        
        with col2:
            st.metric("🌐 Stations", "100", "global locations")
        
        with col3:
            st.metric("📅 Period", "365", "days baseline")
        
        with col4:
            pm25_p99 = df["PM2.5"].quantile(0.99)
            st.metric("🚨 P99 PM2.5", f"{pm25_p99:.1f}", "µg/m³")
        
        st.markdown("---")
        
        # Key findings section
        st.markdown("<h2>📈 Key Findings at a Glance</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3>📊 Task 1: Dimensionality Challenge</h3>
                <p><strong>65%</strong> variance preserved in 2D</p>
                <ul style='color: #d0d8e0; margin: 12px 0;'>
                    <li>PC1 (48.3%): Pollution signature</li>
                    <li>PC2 (16.7%): Weather signature</li>
                    <li>Industrial zones clearly separate</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3>⏱️ Task 2: High-Density Temporal</h3>
                <p><strong>13,289</strong> health violations identified</p>
                <ul style='color: #d0d8e0; margin: 12px 0;'>
                    <li>37.3% violation rate (PM2.5 > 35)</li>
                    <li>Daily cycle: 22.6 µg/m³ variation</li>
                    <li>Peak hours: 1-3 PM traffic signature</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            <div class='info-card'>
                <h3>📈 Task 3: Distribution Modeling</h3>
                <p><strong>200.2 µg/m³</strong> 99th percentile</p>
                <ul style='color: #d0d8e0; margin: 12px 0;'>
                    <li>Only 4 extreme events (1.1%)</li>
                    <li>Right-skewed distribution confirmed</li>
                    <li>Log-scale reveals hidden tail behavior</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='info-card'>
                <h3>✅ Task 4: Visual Integrity</h3>
                <p><strong>3D rejected</strong> for Lie Factor violation</p>
                <ul style='color: #d0d8e0; margin: 12px 0;'>
                    <li>2D solutions proposed (Lie Factor = 1.0)</li>
                    <li>Bivariate mapping + small multiples</li>
                    <li>YlOrRd colormap (colorblind-safe)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Statistics section
        st.markdown("<h2>📊 Dataset Overview</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "PM2.5 Mean",
                f"{df['PM2.5'].mean():.1f} µg/m³",
                f"Max: {df['PM2.5'].max():.1f}"
            )
        
        with col2:
            st.metric(
                "PM10 Mean",
                f"{df['PM10'].mean():.1f} µg/m³",
                f"Max: {df['PM10'].max():.1f}"
            )
        
        with col3:
            st.metric(
                "NO2 Mean",
                f"{df['NO2'].mean():.1f} ppb",
                f"Max: {df['NO2'].max():.1f}"
            )
        
        # Info box
        st.markdown("""
        <div class='info-card' style='margin-top: 24px;'>
            <h3 style='margin-top: 0;'>💡 About This Analysis</h3>
            <p>
            This project demonstrates core data science principles: reducing dimensionality without losing insights, 
            detecting temporal patterns in massive datasets, modeling distributions honestly, and visualizing data 
            with integrity. Navigate through the tasks using the sidebar to explore detailed analysis of each component.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Data file not found. Please run data preprocessing first.")

# ============================================================================
# TASK 1: PCA
# ============================================================================
elif page == "Task 1: PCA":
    st.markdown("""
    <h1>📊 Task 1: Dimensionality Reduction via PCA</h1>
    <p style='color: #a0a8b8; font-size: 15px; margin-top: 8px;'>
        Reduce 6 environmental dimensions to 2D while preserving interpretability and capturing dominant patterns.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Analysis", "Methodology"])
    
    with tab1:
        st.markdown("<h2 style='margin-top: 0;'>📸 Generated Visualizations</h2>", unsafe_allow_html=True)
        viz_files = load_visualizations("task1")
        if viz_files:
            for i, viz_file in enumerate(viz_files):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                    cols = [col1, col2]
                
                with cols[i % 2]:
                    st.image(str(viz_file), use_column_width=True)
                    st.caption(f"📊 {viz_file.stem.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ Visualizations not found. Please run task1_run.py first.")
    
    with tab2:
        st.markdown("<h2 style='margin-top: 0;'>📈 Analysis Results</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0;'>Explained Variance</h3>
                <table style='width: 100%; color: #d0d8e0;'>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td><strong>Component</strong></td>
                        <td style='text-align: right;'><strong>Variance</strong></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td>PC1 (Pollution)</td>
                        <td style='text-align: right; color: #1f77b4;'><strong>48.3%</strong></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td>PC2 (Weather)</td>
                        <td style='text-align: right; color: #ff7f0e;'><strong>16.7%</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Total</strong></td>
                        <td style='text-align: right; color: #2ca02c;'><strong>65.0%</strong></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0;'>Component Interpretation</h3>
                <p><strong style='color: #1f77b4;'>PC1: Pollution Axis</strong></p>
                <ul style='margin: 8px 0; color: #d0d8e0;'>
                    <li>Dominated by: PM2.5, PM10, NO2</li>
                    <li>Meaning: Overall pollution level</li>
                    <li>Pattern: Industrial > Residential</li>
                </ul>
                <p><strong style='color: #ff7f0e;'>PC2: Weather Axis</strong></p>
                <ul style='margin: 8px 0; color: #d0d8e0;'>
                    <li>Dominated by: Temperature, Humidity</li>
                    <li>Meaning: Atmospheric conditions</li>
                    <li>Pattern: Inverse with pollution</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 style='margin-top: 0;'>🔧 Methodology</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
            <h3 style='margin-top: 0; color: #1f77b4;'>PCA Configuration</h3>
            <table style='width: 100%; color: #d0d8e0; font-size: 14px;'>
                <tr style='border-bottom: 1px solid #2a3142;'>
                    <td><strong>Parameter</strong></td>
                    <td style='text-align: right;'><strong>Value</strong></td>
                </tr>
                <tr style='border-bottom: 1px solid #2a3142;'>
                    <td>Components</td>
                    <td style='text-align: right;'>2</td>
                </tr>
                <tr style='border-bottom: 1px solid #2a3142;'>
                    <td>Scaler</td>
                    <td style='text-align: right;'>StandardScaler</td>
                </tr>
                <tr style='border-bottom: 1px solid #2a3142;'>
                    <td>Features</td>
                    <td style='text-align: right;'>6 (PM2.5, PM10, NO2, O3, Temp, Humidity)</td>
                </tr>
                <tr style='border-bottom: 1px solid #2a3142;'>
                    <td>Samples</td>
                    <td style='text-align: right;'>35,626</td>
                </tr>
                <tr>
                    <td>Variance Target</td>
                    <td style='text-align: right;'>65% (cumulative)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TASK 2: TEMPORAL
# ============================================================================
elif page == "Task 2: Temporal":
    st.markdown("""
    <h1>⏱️ Task 2: High-Density Temporal Analysis</h1>
    <p style='color: #a0a8b8; font-size: 15px; margin-top: 8px;'>
        Visualize 365 days × 100 stations simultaneously while revealing daily and weekly patterns.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Findings", "Patterns"])
    
    with tab1:
        st.markdown("<h2 style='margin-top: 0;'>📸 Generated Visualizations</h2>", unsafe_allow_html=True)
        viz_files = load_visualizations("task2")
        if viz_files:
            for i, viz_file in enumerate(viz_files):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                    cols = [col1, col2]
                
                with cols[i % 2]:
                    st.image(str(viz_file), use_column_width=True)
                    st.caption(f"📊 {viz_file.stem.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ Visualizations not found. Please run task2_run.py first.")
    
    with tab2:
        st.markdown("<h2 style='margin-top: 0;'>🔍 Key Findings</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚨 Violations", "13,289", "PM2.5 > 35")
        
        with col2:
            st.metric("⚠️ Violation Rate", "37.3%", "of all data")
        
        with col3:
            st.metric("📈 Daily Variation", "22.6 µg/m³", "low → peak")
        
        with col4:
            st.metric("🚗 Peak Hours", "1-3 PM", "traffic signature")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0;'>📊 Daily Cycle</h3>
                <ul style='color: #d0d8e0;'>
                    <li><strong>Low:</strong> ~26.4 µg/m³ (midnight)</li>
                    <li><strong>High:</strong> ~47.4 µg/m³ (1-3 PM)</li>
                    <li><strong>Variation:</strong> 22.6 µg/m³</li>
                    <li><strong>Cause:</strong> Vehicle traffic peak</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0;'>📅 Periodicities</h3>
                <ul style='color: #d0d8e0;'>
                    <li><strong>7-day cycle:</strong> Weekly pattern</li>
                    <li><strong>3.4-day cycle:</strong> Weather systems</li>
                    <li><strong>24-hour cycle:</strong> Traffic signature</li>
                    <li><strong>Method:</strong> FFT spectral analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 style='margin-top: 0;'>🔬 Pattern Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
            <h3 style='margin-top: 0; color: #1f77b4;'>Temporal Patterns Detected</h3>
            <p><strong style='color: #ff7f0e;'>Daily Pattern (24-hour):</strong></p>
            <ul style='color: #d0d8e0;'>
                <li>Strong correlation with vehicle traffic</li>
                <li>Minimum around midnight (atmospheric ventilation)</li>
                <li>Maximum during afternoon rush hours (1-3 PM)</li>
                <li>Secondary peak possible in morning (6-9 AM)</li>
            </ul>
            
            <p style='margin-top: 16px;'><strong style='color: #ff7f0e;'>Weekly Pattern (7-day):</strong></p>
            <ul style='color: #d0d8e0;'>
                <li>Weekday pollution > Weekend pollution</li>
                <li>Reduced traffic on weekends lowers pollution</li>
                <li>Industrial emissions modulate with work schedule</li>
            </ul>
            
            <p style='margin-top: 16px;'><strong style='color: #ff7f0e;'>Meteorological Pattern (3.4-day):</strong></p>
            <ul style='color: #d0d8e0;'>
                <li>Weather system passage (high/low pressure)</li>
                <li>Wind pattern changes affect pollutant transport</li>
                <li>Atmospheric mixing height variations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TASK 3: DISTRIBUTION
# ============================================================================
elif page == "Task 3: Distribution":
    st.markdown("""
    <h1>📈 Task 3: Distribution Modeling & Tail Integrity</h1>
    <p style='color: #a0a8b8; font-size: 15px; margin-top: 8px;'>
        Represent both peak behavior (where most data lives) and tail behavior (rare extreme events) honestly.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Visualizations", "Statistics", "Scale Analysis"])
    
    with tab1:
        st.markdown("<h2 style='margin-top: 0;'>📸 Generated Visualizations</h2>", unsafe_allow_html=True)
        viz_files = load_visualizations("task3")
        if viz_files:
            for i, viz_file in enumerate(viz_files):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                    cols = [col1, col2]
                
                with cols[i % 2]:
                    st.image(str(viz_file), use_column_width=True)
                    st.caption(f"📊 {viz_file.stem.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ Visualizations not found. Please run task3_run.py first.")
    
    with tab2:
        st.markdown("<h2 style='margin-top: 0;'>📊 Key Statistics</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 P99", "200.2 µg/m³", "extreme hazard")
        
        with col2:
            st.metric("🔴 Extreme Events", "4", ">200 µg/m³ (1.1%)")
        
        with col3:
            st.metric("📉 Skewness", "Right-skewed", "via Q-Q plot")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #1f77b4;'>Distribution Shape</h3>
                <ul style='color: #d0d8e0;'>
                    <li><strong>Type:</strong> Right-skewed (lognormal-like)</li>
                    <li><strong>Peak:</strong> ~30-40 µg/m³</li>
                    <li><strong>Long tail:</strong> Extends to 500+ µg/m³</li>
                    <li><strong>Normality:</strong> Q-Q plot shows deviation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #1f77b4;'>Percentile Breakdown</h3>
                <table style='width: 100%; color: #d0d8e0; font-size: 13px;'>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td><strong>Percentile</strong></td>
                        <td style='text-align: right;'><strong>PM2.5</strong></td>
                    </tr>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td>P50 (Median)</td>
                        <td style='text-align: right;'>~35 µg/m³</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td>P75</td>
                        <td style='text-align: right;'>~52 µg/m³</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #2a3142;'>
                        <td>P95</td>
                        <td style='text-align: right;'>~92 µg/m³</td>
                    </tr>
                    <tr>
                        <td>P99 (Extreme)</td>
                        <td style='text-align: right;'><strong style='color: #d62728;'>200.2 µg/m³</strong></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 style='margin-top: 0;'>🔬 Scale Comparison</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #1f77b4;'>Linear Scale (Peak-Optimized)</h3>
                <p style='margin: 12px 0; color: #2ca02c;'><strong>✓ Strengths:</strong></p>
                <ul style='color: #d0d8e0; margin: 8px 0;'>
                    <li>Shows where most data lives</li>
                    <li>Intuitive for general audience</li>
                    <li>Natural interpretation</li>
                </ul>
                <p style='margin: 12px 0; color: #d62728;'><strong>✗ Weaknesses:</strong></p>
                <ul style='color: #d0d8e0; margin: 8px 0;'>
                    <li>Hides tail behavior completely</li>
                    <li>Misleading for risk assessment</li>
                    <li>Extreme events invisible</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #ff7f0e;'>Log Scale (Tail-Optimized)</h3>
                <p style='margin: 12px 0; color: #2ca02c;'><strong>✓ Strengths:</strong></p>
                <ul style='color: #d0d8e0; margin: 8px 0;'>
                    <li>Reveals rare extreme events</li>
                    <li>Better for policy decisions</li>
                    <li>Shows full dynamic range</li>
                </ul>
                <p style='margin: 12px 0; color: #d62728;'><strong>✗ Weaknesses:</strong></p>
                <ul style='color: #d0d8e0; margin: 8px 0;'>
                    <li>Requires statistical literacy</li>
                    <li>Less intuitive visually</li>
                    <li>Equal spacing ≠ equal data</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card' style='margin-top: 24px;'>
            <h3 style='margin-top: 0; color: #1f77b4;'>💡 Honesty Conclusion</h3>
            <p style='color: #d0d8e0;'>
            For health risk assessment, <strong>log-scale is more honest</strong> because it reveals that extreme 
            pollution events (~200 µg/m³) are rare but critically important to policy. A linear scale would hide 
            this critical information, misleading decision-makers into thinking extreme events are negligible.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TASK 4: VISUAL INTEGRITY
# ============================================================================
elif page == "Task 4: Visual Integrity":
    st.markdown("""
    <h1>✅ Task 4: Visual Integrity Audit</h1>
    <p style='color: #a0a8b8; font-size: 15px; margin-top: 8px;'>
        Visualize 3 variables (Pollution, Population Density, Region) honestly using Tufte principles.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Visualizations", "3D Rejection", "2D Solutions", "Color Palette"])
    
    with tab1:
        st.markdown("<h2 style='margin-top: 0;'>📸 Visual Demonstrations</h2>", unsafe_allow_html=True)
        viz_files = load_visualizations("task4")
        if viz_files:
            for i, viz_file in enumerate(viz_files):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                    cols = [col1, col2]
                
                with cols[i % 2]:
                    st.image(str(viz_file), use_column_width=True)
                    st.caption(f"📊 {viz_file.stem.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ Visualizations not found. Please run task4_run.py first.")
    
    with tab2:
        st.markdown("<h2 style='margin-top: 0;'>❌ Why 3D Charts Are Dishonest</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #d62728;'>Lie Factor Violation</h3>
                <p style='color: #d0d8e0; margin: 12px 0;'>
                <strong style='font-size: 16px;'>2.0 - 5.0×</strong>
                </p>
                <p style='color: #a0a8b8; font-size: 13px;'>
                Standard: <strong>0.95 - 1.05×</strong> (acceptable)
                </p>
                <p style='color: #d62728; margin-top: 16px;'><strong>EXCEEDS STANDARD BY 2-5×</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #d62728;'>Data-Ink Ratio</h3>
                <p style='color: #d0d8e0; margin: 12px 0;'>
                <strong style='font-size: 16px;'>40-60%</strong>
                </p>
                <p style='color: #a0a8b8; font-size: 13px;'>
                Target: <strong>>75%</strong> (good)
                </p>
                <p style='color: #d62728; margin-top: 16px;'><strong>VERY POOR — WASTED INK</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class='info-card'>
            <h3 style='margin-top: 0; color: #1f77b4;'>🔍 Distortion Problems in 3D</h3>
            <ol style='color: #d0d8e0;'>
                <li><strong>Perspective Distortion:</strong> Front bars appear 5-40% larger than back bars</li>
                <li><strong>Occlusion:</strong> Back bars hidden behind front bars (cannot compare simultaneously)</li>
                <li><strong>Color Gradient Confusion:</strong> Shading creates false data encoding</li>
                <li><strong>Axis Length Distortion:</strong> Depth axis violates Lie Factor 1.5-3.0×</li>
                <li><strong>Spatial Ambiguity:</strong> Unclear which bars align (perspective breaks visual order)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 style='margin-top: 0;'>✅ Proposed 2D Solutions (Lie Factor = 1.0)</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #2ca02c;'>Solution 1: Bivariate Mapping ⭐</h3>
                <p style='margin: 12px 0; color: #d0d8e0;'><strong style='color: #1f77b4;'>Best for scientific audience</strong></p>
                <p style='color: #a0a8b8; font-size: 13px;'><strong>Visual Encoding:</strong></p>
                <ul style='color: #d0d8e0; margin: 8px 0; font-size: 13px;'>
                    <li><strong>X-axis:</strong> Population density</li>
                    <li><strong>Y-axis:</strong> PM2.5 pollution</li>
                    <li><strong>Color:</strong> Severity (YlOrRd sequential)</li>
                    <li><strong>Labels:</strong> Region names</li>
                </ul>
                <p style='margin-top: 16px; color: #2ca02c;'><strong>✓ Advantages:</strong></p>
                <ul style='color: #d0d8e0; font-size: 13px;'>
                    <li>Lie Factor = 1.0 (perfect honesty)</li>
                    <li>Data-ink ratio = 80-90%</li>
                    <li>Shows all 3 variables simultaneously</li>
                    <li>Reveals density-pollution relationship</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #2ca02c;'>Solution 2: Small Multiples ⭐</h3>
                <p style='margin: 12px 0; color: #d0d8e0;'><strong style='color: #1f77b4;'>Best for general audience</strong></p>
                <p style='color: #a0a8b8; font-size: 13px;'><strong>Visual Encoding:</strong></p>
                <ul style='color: #d0d8e0; margin: 8px 0; font-size: 13px;'>
                    <li><strong>Layout:</strong> 8 mini-charts (one per region)</li>
                    <li><strong>X-axis:</strong> Region name</li>
                    <li><strong>Y-axis:</strong> PM2.5 pollution (bars)</li>
                    <li><strong>Color:</strong> Zone type (Industrial/Residential)</li>
                </ul>
                <p style='margin-top: 16px; color: #2ca02c;'><strong>✓ Advantages:</strong></p>
                <ul style='color: #d0d8e0; font-size: 13px;'>
                    <li>Lie Factor = 1.0 (perfect honesty)</li>
                    <li>Data-ink ratio = 75-85%</li>
                    <li>Intuitive to understand</li>
                    <li>Easy per-region comparison</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<h2 style='margin-top: 0;'>🎨 Color Palette Justification</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #2ca02c;'>YlOrRd (Selected) ✅</h3>
                <p style='color: #a0a8b8; font-size: 13px;'><strong>Sequential colormap</strong></p>
                <p style='margin: 12px 0; color: #2ca02c;'><strong>Properties:</strong></p>
                <ul style='color: #d0d8e0; font-size: 13px;'>
                    <li>Monotonic luminance (intuitive)</li>
                    <li>Yellow → Orange → Red progression</li>
                    <li>Colorblind-safe (D/P friendly)</li>
                    <li>B&W print preservation</li>
                </ul>
                <p style='margin: 12px 0; color: #2ca02c;'><strong>Psychology:</strong></p>
                <ul style='color: #d0d8e0; font-size: 13px;'>
                    <li>🟡 Yellow = Caution</li>
                    <li>🟠 Orange = Warning</li>
                    <li>🔴 Red = Danger/Alert</li>
                </ul>
                <p style='margin-top: 12px; color: #1f77b4; font-size: 13px;'><strong>Scientific Status:</strong> Recommended by Nature, Science journals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3 style='margin-top: 0; color: #d62728;'>Jet Spectrum (Rejected) ❌</h3>
                <p style='color: #a0a8b8; font-size: 13px;'><strong>Rainbow colormap</strong></p>
                <p style='margin: 12px 0; color: #d62728;'><strong>Problems:</strong></p>
                <ul style='color: #d0d8e0; font-size: 13px;'>
                    <li>Non-monotonic luminance (green brightest)</li>
                    <li>Counterintuitive mapping (green → data?)</li>
                    <li>Poor colorblind accessibility</li>
                    <li>Fails in B&W printing</li>
                </ul>
                <p style='margin: 12px 0; color: #d62728;'><strong>Perceptual Issues:</strong></p>
                <ul style='color: #d0d8e0; font-size: 13px;'>
                    <li>False perception of "good data" at green</li>
                    <li>Creates false features at boundaries</li>
                    <li>Violates scientific visualization standards</li>
                </ul>
                <p style='margin-top: 12px; color: #d62728; font-size: 13px;'><strong>Scientific Status:</strong> Deprecated — no longer standard</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR INFO
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 12px 0;'>
    <h3 style='margin: 0; font-size: 14px; color: #1f77b4; text-transform: uppercase; letter-spacing: 1px;'>📊 Dataset Info</h3>
    <ul style='list-style: none; padding: 12px 0; margin: 0; font-size: 13px; color: #d0d8e0;'>
        <li><strong>Stations:</strong> 100 global locations</li>
        <li><strong>Period:</strong> 365 days baseline</li>
        <li><strong>Records:</strong> 35,626 observations</li>
        <li><strong>Parameters:</strong> PM2.5, PM10, NO2, O3, Temp, Humidity</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 12px 0;'>
    <h3 style='margin: 0; font-size: 14px; color: #1f77b4; text-transform: uppercase; letter-spacing: 1px;'>🛠️ Technology Stack</h3>
    <ul style='list-style: none; padding: 12px 0; margin: 0; font-size: 13px; color: #d0d8e0;'>
        <li><strong>Analysis:</strong> scikit-learn, scipy, pandas, numpy</li>
        <li><strong>Visualization:</strong> matplotlib, seaborn, plotly</li>
        <li><strong>Data Format:</strong> Apache Parquet</li>
        <li><strong>Framework:</strong> Streamlit</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='padding: 12px 0;'>
    <h3 style='margin: 0; font-size: 14px; color: #1f77b4; text-transform: uppercase; letter-spacing: 1px;'>✓ Visualization Standards</h3>
    <ul style='list-style: none; padding: 12px 0; margin: 0; font-size: 13px; color: #d0d8e0;'>
        <li><strong>Lie Factor:</strong> 0.95-1.05 (Tufte)</li>
        <li><strong>Data-Ink Ratio:</strong> >75% efficiency</li>
        <li><strong>Colorblind Safe:</strong> Monotonic luminance</li>
        <li><strong>Honesty:</strong> Context-appropriate scales</li>
    </ul>
</div>
""", unsafe_allow_html=True)
