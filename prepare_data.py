"""Main pipeline: fetch and preprocess data."""
import pandas as pd
import os
import sys
from pathlib import Path
from src.data_fetch import fetch_global_data, generate_synthetic_data
from src.data_processor import prepare_dataset


def main(use_synthetic: bool = False):
    """
    Main execution: fetch 100 global locations and preprocess data.
    
    Parameters:
    - use_synthetic: Force use of synthetic data instead of API
    """
    project_root = Path(__file__).parent
    raw_data_path = project_root / "data" / "raw" / "openaq_2025.parquet"
    processed_data_path = project_root / "data" / "processed" / "prepared_data.parquet"
    metadata_path = project_root / "data" / "processed" / "metadata.parquet"
    
    # Create directories if needed
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have raw data
    if raw_data_path.exists():
        print(f"Loading cached raw data from {raw_data_path}...")
        df_raw = pd.read_parquet(raw_data_path)
    else:
        if use_synthetic:
            print("Generating hourly synthetic data (876K records for 100 stations × 365 days)...")
            df_raw = generate_synthetic_data(num_locations=100, num_days=365, hourly=True)
        else:
            print("Fetching data from OpenAQ API...")
            df_raw = fetch_global_data(
                num_locations=100,
                date_from="2025-01-01",
                date_to="2025-12-31"
            )
            
            if df_raw.empty:
                print("API returned no data. Falling back to hourly synthetic data...")
                df_raw = generate_synthetic_data(num_locations=100, num_days=365, hourly=True)
        
        if df_raw.empty:
            print("ERROR: No data available")
            return False
        
        print(f"Generated {len(df_raw):,} records")
        print(f"Saving to {raw_data_path}...")
        df_raw.to_parquet(raw_data_path, index=False)
    
    print(f"\nRaw data shape: {df_raw.shape}")
    print(f"Columns: {df_raw.columns.tolist()}")
    print(f"Parameters available: {df_raw['parameter'].unique().tolist()}")
    
    # Preprocess data
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    df_processed, metadata, df_raw_features = prepare_dataset(df_raw)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Columns: {df_processed.columns.tolist()}")
    print(f"Zone distribution:")
    print(df_processed['zone'].value_counts())
    
    print(f"\nHealth violations (PM2.5 > 35): {df_processed['health_violation'].sum()} records")
    
    # Count extreme hazard events in raw features (before standardization)
    extreme_count = (df_raw_features['PM2.5'] > 200).sum()
    print(f"Extreme hazard events (PM2.5 > 200): {extreme_count} records")
    
    # Save processed data
    print(f"\nSaving processed data to {processed_data_path}...")
    df_processed.to_parquet(processed_data_path, index=False)
    
    print(f"Saving raw features to {project_root / 'data' / 'processed' / 'raw_features.parquet'}...")
    df_raw_features.to_parquet(project_root / 'data' / 'processed' / 'raw_features.parquet', index=False)
    
    print(f"Saving metadata to {metadata_path}...")
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_parquet(metadata_path, index=False)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"✓ Raw data: {raw_data_path.relative_to(project_root)}")
    print(f"✓ Processed data: {processed_data_path.relative_to(project_root)}")
    print(f"✓ Metadata: {metadata_path.relative_to(project_root)}")
    print(f"✓ Records in processed data: {len(df_processed)}")
    
    return True


if __name__ == "__main__":
    use_syn = "--synthetic" in sys.argv
    success = main(use_synthetic=use_syn)
    sys.exit(0 if success else 1)
