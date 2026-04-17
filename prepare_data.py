"""Main pipeline: fetch and preprocess data."""
import pandas as pd
import os
from pathlib import Path
from src.data_fetch import fetch_global_data
from src.data_processor import prepare_dataset


def main():
    """
    Main execution: fetch 100 global locations and preprocess data.
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
        print("Fetching data from OpenAQ API...")
        df_raw = fetch_global_data(
            num_locations=100,
            date_from="2025-01-01",
            date_to="2025-12-31"
        )
        
        if df_raw.empty:
            print("ERROR: No data fetched from API")
            return
        
        print(f"Fetched {len(df_raw)} records from OpenAQ")
        print(f"Saving to {raw_data_path}...")
        df_raw.to_parquet(raw_data_path, index=False)
    
    print(f"\nRaw data shape: {df_raw.shape}")
    print(f"Columns: {df_raw.columns.tolist()}")
    print(f"Parameters available: {df_raw['parameter'].unique().tolist()}")
    
    # Preprocess data
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    df_processed, metadata = prepare_dataset(df_raw)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Columns: {df_processed.columns.tolist()}")
    print(f"Zone distribution:\n{df_processed['zone'].value_counts()}")
    
    # Save processed data
    print(f"\nSaving processed data to {processed_data_path}...")
    df_processed.to_parquet(processed_data_path, index=False)
    
    print(f"Saving metadata to {metadata_path}...")
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_parquet(metadata_path, index=False)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"✓ Raw data: {raw_data_path.relative_to(project_root)}")
    print(f"✓ Processed data: {processed_data_path.relative_to(project_root)}")
    print(f"✓ Metadata: {metadata_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
