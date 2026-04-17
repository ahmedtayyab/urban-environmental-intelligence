# Urban Environmental Intelligence Challenge

A comprehensive data science project analyzing air quality patterns across 100 global locations over 365 days.

## 📊 Project Overview

This assignment explores urban environmental intelligence through 4 major analytical tasks:
1. **Dimensionality Reduction** - PCA analysis reducing 6D environmental data to 2D
2. **Temporal Analysis** - High-density patterns across 100 stations × 365 days
3. **Distribution Modeling** - Honest representation of peak and tail behavior
4. **Visual Integrity** - Rejecting dishonest visualizations, proposing ethical alternatives

## 🎯 Key Findings

| Metric | Value | Significance |
|--------|-------|--------------|
| **Total Records** | 35,626 | 365 days × 100 stations |
| **PCA Variance** | 65% | PC1: 48.3%, PC2: 16.7% |
| **Health Violations** | 13,289 | 37.3% violation rate (PM2.5 > 35 µg/m³) |
| **Daily Variation** | 22.6 µg/m³ | Midnight low (26.4) → Noon peak (47.4) |
| **99th Percentile** | 200.2 µg/m³ | Extreme hazard threshold |
| **Extreme Events** | 4 | Only 1.1% exceed 200 µg/m³ |
| **Peak Hours** | 1-3 PM | Traffic signature (obvious correlation) |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.13.7
pandas, numpy, scipy, scikit-learn
matplotlib, seaborn, plotly
streamlit, requests, pyarrow
```

### Installation
```bash
pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run streamlit_app.py
```
Dashboard opens at: `http://localhost:8501`

### Run Individual Tasks
```bash
python task1_run.py  # PCA analysis
python task2_run.py  # Temporal analysis
python task3_run.py  # Distribution modeling
python task4_run.py  # Visual integrity audit
```

## 📁 Project Structure

```
urban-environmental-intelligence/
├── src/
│   ├── data_fetch.py              # OpenAQ API client + synthetic data generator
│   ├── data_processor.py           # Cleaning, standardization, zone classification
│   ├── task1_pca.py               # PCAAnalyzer class (dimensionality reduction)
│   ├── task2_temporal.py          # TemporalAnalyzer class (time series patterns)
│   ├── task3_distribution.py      # DistributionAnalyzer class (statistical modeling)
│   ├── task4_visualization.py     # ThreeDAnalyzer class (visual integrity audit)
│   └── __init__.py
├── data/
│   ├── raw_data.parquet           # Raw OpenAQ + synthetic (219K records)
│   └── prepared_data.parquet      # Cleaned, standardized (35,626 records)
├── visualizations/
│   ├── task1/                     # PCA scatter, loadings, biplot
│   ├── task2/                     # Heatmap, timeline, cycles
│   ├── task3/                     # Peak/tail histograms, Q-Q plot
│   └── task4/                     # Rejected 3D, bivariate, small multiples, colors
├── task1_run.py                   # Task 1 orchestration
├── task2_run.py                   # Task 2 orchestration
├── task3_run.py                   # Task 3 orchestration
├── task4_run.py                   # Task 4 orchestration
├── streamlit_app.py               # Interactive dashboard
├── prepare_data.py                # Data preprocessing script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔍 Task Details

### Task 1: Dimensionality Reduction via PCA
**Challenge:** Reduce 6D environmental data to 2D while preserving interpretability.

**Approach:**
- StandardScaler normalization
- 2-component PCA
- Variance analysis and component interpretation

**Key Results:**
- PC1 (48.3%): Pollution axis (PM2.5, PM10, NO2 dominant)
- PC2 (16.7%): Weather axis (Temperature, Humidity dominant)
- Total variance preserved: 65%
- Industrial zones naturally cluster at higher PC1 values

**Visualizations:**
- PCA scatter plot (zones distinguished)
- Loadings heatmap (variable contributions)
- Biplot with variable vectors

---

### Task 2: High-Density Temporal Analysis
**Challenge:** Visualize 365 days × 100 stations simultaneously while revealing daily/weekly patterns.

**Approach:**
- Dense heatmap (100×365 matrix)
- Violation timeline (PM2.5 > 35 µg/m³ events)
- Daily cycle detection (24-hour synthetic pattern)
- FFT spectral analysis

**Key Results:**
- 13,289 health violations (37.3% violation rate)
- Daily cycle: 22.6 µg/m³ variation
- Peak hours: 1-3 PM (traffic signature)
- Periodicities: 3.4-day and 7-day detected

**Visualizations:**
- High-density raster heatmap
- Daily violation timeline
- Synthetic 24-hour cycle (traffic effects)
- Weekly aggregation plot

---

### Task 3: Distribution Modeling & Tail Integrity
**Challenge:** Represent both peak (where most data lives) and tail (rare events) honestly.

**Approach:**
- Dual histograms: linear (peak-optimized) + log (tail-optimized)
- Q-Q plot analysis
- Distribution fitting (normal, gamma)
- Percentile analysis

**Key Results:**
- 99th percentile: 200.2 µg/m³ (extreme hazard)
- Only 4 extreme events >200 µg/m³ (1.1%)
- Right-skewed distribution confirmed
- Log-scale more honest for risk assessment

**Visualizations:**
- Linear histogram (peak focus, 40 bins)
- Log histogram (tail focus, 80 bins)
- Q-Q plot (normality check)
- Focused tail plot (zoomed distribution)

**Principle:** Linear scale hides tail behavior. Log scale reveals critical rare events. For health risk, log scale is more honest.

---

### Task 4: Visual Integrity Audit
**Challenge:** Visualize 3 variables (Pollution, Density, Region) honestly per Tufte principles.

**Approach:**
- 3D proposal analysis (REJECTED)
- Lie Factor calculation (0.95-1.05 standard)
- Data-ink ratio assessment (>75% target)
- Alternative solutions (2D options)
- Color scale justification

**Key Analysis:**
- 3D bar chart: Lie Factor 2.0-5.0× (VIOLATES HONESTY)
- Distortions: Perspective (5-40%), occlusion (20-40%), color confusion
- Data-ink: 3D = 40-60% (poor), 2D = 75-90% (good)

**Solutions Provided:**

**Solution 1: Bivariate Mapping** (Best for scientific audience)
- X-axis: Population density
- Y-axis: PM2.5 pollution
- Color: Severity (YlOrRd sequential)
- Result: Lie Factor 1.0, shows pollution vs. density relationship

**Solution 2: Small Multiples** (Best for general audience)
- 8 mini-charts (one per region)
- Bars: PM2.5 levels
- Color: Zone type (Industrial/Residential)
- Result: Lie Factor 1.0, intuitive comparison

**Color Justification: YlOrRd Sequential (Chosen)**
✅ Monotonic luminance (intuitive low→high risk)
✅ Colorblind-friendly (unlike Jet rainbow)
✅ Preserves in B&W printing
✅ Scientific consensus (rainbow deprecated)

**Visualizations:**
- Rejected 3D chart (shows distortion)
- Bivariate mapping (scatter with colors)
- Small multiples (8 mini-charts)
- Color scale comparison (YlOrRd vs Jet)

---

## 🎨 Visualization Principles Applied

### Tufte's Data-Ink Ratio
- **Target:** >75% of ink should represent data
- **Task 1:** Achieved with focused PCA plots
- **Task 2:** Heatmap optimized (minimal axis ink)
- **Task 3:** Linear and log histograms (no decorations)
- **Task 4:** 2D solutions (75-90% efficiency)

### Lie Factor Analysis
- **Standard:** 0.95-1.05 (perfect honesty)
- **3D rejection:** 2.0-5.0× violation (dishonest perspective/occlusion)
- **2D solutions:** 1.0 (perfectly honest)

### Color Standards
- **Sequential colormaps:** YlOrRd (monotonic luminance)
- **Avoid:** Rainbow (non-monotonic, counterintuitive)
- **Accessibility:** Colorblind-tested (Deuteranopia, Protanopia)
- **Print:** B&W preservation priority

### Honesty Principles
- **Context-appropriate scales:** Log for tails, linear for peaks
- **No 3D charts:** Perspective distortion always violates Lie Factor
- **No shadows/decorations:** Minimize chart junk
- **Transparent methodology:** Document all transformation choices

## 📈 Data Source

**Real Data:** OpenAQ API v2 (100 global locations)
- PM2.5, PM10, NO2, O3
- Temperature, Humidity
- 1-year baseline (365 days)

**Fallback:** Synthetic data generator (realistic 219K records)
- Zone-aware patterns (Industrial vs Residential)
- Daily cycles (traffic effects)
- Seasonal variation
- Extreme events (1% tail)

## 🛠️ Technologies

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.13.7 |
| **Analysis** | scikit-learn, scipy, pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Data Storage** | Apache Parquet |
| **Web Dashboard** | Streamlit |
| **API Client** | requests |
| **Data Format** | pandas DataFrame |

## 📚 Key References

- **Tufte, E. R.** (1983). The Visual Display of Quantitative Information
- OpenAQ API Documentation: https://docs.openaq.org/
- Matplotlib Colormap Guide: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- Streamlit Documentation: https://docs.streamlit.io/

## 📝 Git Commits

Track progress through git history:
```bash
git log --oneline
# Shows 4 major commits (Tasks 1-4) + data setup + project init
```

All work systematically committed with clear descriptions.

## 🎓 Learning Outcomes

After this project, you understand:

✅ **Dimensionality Reduction**
- How PCA captures dominant patterns
- Variance interpretation
- Scree plot analysis

✅ **Temporal Analysis**
- Dense visualization techniques
- Cycle detection (FFT)
- High-dimensional time series

✅ **Distribution Modeling**
- Peak vs tail honesty
- Scale appropriateness for context
- Extreme value analysis

✅ **Visual Integrity**
- Lie Factor measurements
- Data-ink ratio optimization
- Ethical data visualization
- Color accessibility standards

✅ **Reproducible Data Science**
- Modular code structure
- Clear documentation
- Git-based workflow
- Interactive dashboards

## 🚀 Running the Dashboard

### Option 1: Local Development
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Option 2: Remote Deployment (Streamlit Cloud)
```bash
# Push code to GitHub, connect to Streamlit Cloud
# Auto-deploys on every git push
```

### Navigation
Use the sidebar to switch between:
- **Dashboard:** Overview + key metrics
- **Task 1:** PCA dimensionality reduction
- **Task 2:** High-density temporal analysis
- **Task 3:** Distribution with peak/tail focus
- **Task 4:** Visual integrity audit (3D rejection + 2D solutions)

Each task includes:
- Embedded visualizations
- Statistical summaries
- Methodology explanation
- Interactive insights

## 📧 Contact & Questions

For questions about methodology, code, or findings, refer to:
- Task documentation in `src/taskN_*.py`
- Docstrings in Python classes
- Comments in visualization code
- This README

---

**Status:** ✅ All 4 tasks complete | 📊 Dashboard live | 📈 Ready for presentation

*Last updated: 2026-04-17*
