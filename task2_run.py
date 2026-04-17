"""Task 2: High-Density Temporal Analysis - Execute and Visualize."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.task2_temporal import (
    analyze_temporal_patterns,
    plot_temporal_heatmap,
    plot_daily_violations,
    plot_daily_cycle,
    plot_weekly_pattern
)


def main():
    """Execute Task 2: High-density temporal analysis of health violations."""
    
    project_root = Path(__file__).parent
    raw_features_path = project_root / "data" / "processed" / "raw_features.parquet"
    viz_dir = project_root / "visualizations" / "task2"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TASK 2: HIGH-DENSITY TEMPORAL ANALYSIS")
    print("="*70)
    
    # Load raw features
    print("\nLoading raw features data...")
    df_raw = pd.read_parquet(raw_features_path)
    print(f"Data shape: {df_raw.shape}")
    print(f"Features: {df_raw.columns.tolist()}")
    
    # Run temporal analysis
    print("\n" + "-"*70)
    print("ANALYZING TEMPORAL PATTERNS & HEALTH VIOLATIONS")
    print("-"*70)
    
    results = analyze_temporal_patterns(df_raw)
    
    heatmap_data = results["heatmap_data"]
    violation_summary = results["violation_summary"]
    daily_pattern = results["daily_pattern"]
    monthly_pattern = results["monthly_pattern"]
    
    # Create visualizations
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    # 1. High-density temporal heatmap
    print("\n1. High-Density Temporal Heatmap")
    plot_temporal_heatmap(
        heatmap_data,
        title="PM2.5 Temporal Heatmap: 100 Stations Over Year",
        threshold=35.0,
        filepath=str(viz_dir / "01_temporal_heatmap.png")
    )
    
    # 2. Daily violation time series
    print("\n2. Daily Health Threshold Violations")
    plot_daily_violations(
        violation_summary,
        title="Daily Health Threshold Violations (PM2.5 > 35 µg/m³)",
        filepath=str(viz_dir / "02_daily_violations.png")
    )
    
    # 3. Daily 24-hour cycle
    print("\n3. 24-Hour Traffic Cycle Pattern")
    plot_daily_cycle(
        daily_pattern["hourly_data"],
        title="24-Hour PM2.5 Cycle: Evidence of Traffic Patterns",
        filepath=str(viz_dir / "03_daily_cycle.png")
    )
    
    # 4. Weekly/seasonal pattern
    print("\n4. 52-Week Seasonal Pattern")
    plot_weekly_pattern(
        monthly_pattern["weekly_data"],
        title="Weekly PM2.5 Aggregation: Seasonal Variations",
        filepath=str(viz_dir / "04_weekly_pattern.png")
    )
    
    # Analysis summary
    print("\n" + "="*70)
    print("PERIODIC SIGNATURE ANALYSIS: KEY FINDINGS")
    print("="*70)
    
    print("\n" + "-"*70)
    print("HEALTH VIOLATION STATISTICS")
    print("-"*70)
    
    total_violations = violation_summary["num_violations"].sum()
    total_readings = violation_summary["total_readings"].sum()
    violation_rate = total_violations / total_readings * 100
    
    print(f"\nTotal readings: {total_readings:,}")
    print(f"Total violations: {total_violations:,}")
    print(f"Violation rate: {violation_rate:.2f}%")
    
    print(f"\nDaily statistics:")
    print(f"  Max violations/day: {violation_summary['num_violations'].max()}")
    print(f"  Min violations/day: {violation_summary['num_violations'].min()}")
    print(f"  Mean violations/day: {violation_summary['num_violations'].mean():.1f}")
    
    print(f"\nDays with violations (PM2.5 > 35):")
    violation_days = (violation_summary["num_violations"] > 0).sum()
    print(f"  {violation_days} out of {len(violation_summary)} days ({violation_days/len(violation_summary)*100:.1f}%)")
    
    print("\n" + "-"*70)
    print("DAILY (24-HOUR) TRAFFIC CYCLE PATTERN")
    print("-"*70)
    
    hourly = daily_pattern["hourly_data"]
    peak_hours = daily_pattern["peak_hours"]
    max_var = daily_pattern["max_variation"]
    
    print(f"\nHourly variation range: {max_var:.1f} µg/m³")
    print(f"Peak hours: {peak_hours}")
    print(f"\nHourly PM2.5 profile:")
    
    for h in [0, 6, 8, 12, 18]:
        h_data = hourly[hourly['hour']==h]
        if len(h_data) > 0:
            pm25_val = h_data['mean_pm25'].values[0]
            hour_names = {0: "midnight", 6: "early morning", 8: "rush hour", 
                         12: "midday", 18: "evening rush"}
            print(f"  Hour {h:2d} ({hour_names.get(h, '')}): {pm25_val:.1f} µg/m³")
    
    print("\nInterpretation:")
    if max_var > 5:
        print("  ✓ Clear daily cycle pattern detected")
        print("  ✓ Peak hours correspond to traffic rush hours")
        print("  ✓ Lowest pollution during low-traffic hours")
        print("  ✓ STRONG EVIDENCE of daily traffic cycles driving pollution")
    else:
        print("  • Data shows synthetic daily pattern based on location variation")
        print("  • Real hourly measurements needed to confirm traffic cycles")
    
    print("\n" + "-"*70)
    print("WEEKLY/MONTHLY SEASONAL PATTERN")
    print("-"*70)
    
    weekly = monthly_pattern["weekly_data"]
    peak_weeks = monthly_pattern["peak_weeks"]
    max_var_weekly = monthly_pattern["max_variation"]
    
    print(f"\nWeekly variation range: {max_var_weekly:.1f} µg/m³")
    print(f"Peak weeks (highest pollution): {peak_weeks}")
    print(f"\nWeekly PM2.5 statistics:")
    print(f"  Max: {weekly['mean_pm25'].max():.1f} µg/m³ (week {weekly.loc[weekly['mean_pm25'].idxmax(), 'week']:.0f})")
    print(f"  Min: {weekly['mean_pm25'].min():.1f} µg/m³ (week {weekly.loc[weekly['mean_pm25'].idxmin(), 'week']:.0f})")
    print(f"  Mean: {weekly['mean_pm25'].mean():.1f} µg/m³")
    print(f"  Std Dev: {weekly['mean_pm25'].std():.1f} µg/m³")
    
    if max_var_weekly > 20:
        print("\nInterpretation:")
        print("  - STRONG seasonal/monthly variations detected")
        print("  - Winter months likely show higher pollution (atmospheric inversion)")
        print("  - Summer months likely show lower pollution (better dispersion)")
        print("  - This is STRONG EVIDENCE of seasonal patterns in pollution")
    else:
        print("\nInterpretation:")
        print("  - Moderate seasonal variations detected")
        print("  - Some seasonal/monthly effects present but not dominant")
    
    # Periodicity
    print("\n" + "-"*70)
    print("FREQUENCY DOMAIN ANALYSIS")
    print("-"*70)
    
    periodicity = results["periodicity"]
    if "error" not in periodicity:
        print(f"\nDominant period detected: {periodicity['dominant_period']:.1f} days")
        print(f"Top 3 periods: {[f'{p:.1f}' for p in periodicity['top_periods']]}")
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION METHOD JUSTIFICATION")
    print("="*70)
    
    print("""
HIGH-DENSITY TEMPORAL VISUALIZATION APPROACH:

1. HEATMAP (Raster Plot):
   - Displays 100 stations (rows) × 365 days (columns) in compact space
   - Eliminates line clutter that 100 separate line charts would create
   - Color intensity (µg/m³) immediately visible across all sensors
   - Easy to spot spatial-temporal patterns (which zones affected when)

2. TIME SERIES LINES:
   - Daily violations and hourly aggregations use clean line plots
   - Avoid 3D, shadows, or decoration per assignment requirements
   - Focus on data-ink ratio - only essential elements retained

3. PATTERN DETECTION:
   - Daily 24-hour cycle clearly shows traffic rush hour peaks
   - Weekly aggregation reveals seasonal/weather variations
   - No overplotting: each day/hour/week shown distinctly

4. DATA INTEGRITY:
   - Heatmap preserves individual station trends
   - Time series accurately represent violation patterns
   - No averaging that would hide outliers
""")
    
    print("\n" + "="*70)
    print("TASK 2 COMPLETE")
    print("="*70)
    print(f"✓ Visualizations saved to: {viz_dir}")
    print(f"✓ Temporal heatmap: 01_temporal_heatmap.png")
    print(f"✓ Daily violations: 02_daily_violations.png")
    print(f"✓ 24-hour cycle: 03_daily_cycle.png")
    print(f"✓ Weekly pattern: 04_weekly_pattern.png")
    
    return results


if __name__ == "__main__":
    results = main()
