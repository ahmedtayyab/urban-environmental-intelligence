"""Data processing and preprocessing utilities."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")


class DataProcessor:
    """Handle data cleaning, preprocessing, and feature engineering."""
    
    # Required environmental parameters for analysis
    CORE_PARAMETERS = ["PM2.5", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data from API.
        
        Steps:
        - Convert date to datetime
        - Remove nulls
        - Convert values to numeric
        """
        df = df.copy()
        
        # Parse dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Remove rows with null dates or values
        df = df.dropna(subset=["date", "value"])
        
        # Convert value to numeric, coercing errors to NaN
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        
        # Remove negative values (physical impossibility for air quality)
        df = df[df["value"] >= 0]
        
        return df
    
    @staticmethod
    def pivot_to_features(df: pd.DataFrame, 
                         location_id_col: str = "location_id",
                         date_col: str = "date") -> pd.DataFrame:
        """
        Pivot measurements to wide format with parameters as columns.
        
        One row = one location + one timestamp
        Columns = PM2.5, PM10, NO2, Ozone, Temperature, Humidity
        """
        df = df.copy()
        
        # Keep only core parameters
        df = df[df["parameter"].isin(DataProcessor.CORE_PARAMETERS)].copy()
        
        # Pivot: each parameter becomes a column
        df_pivot = df.pivot_table(
            index=[location_id_col, date_col],
            columns="parameter",
            values="value",
            aggfunc="mean"  # If multiple readings, take mean
        )
        
        df_pivot = df_pivot.reset_index()
        df_pivot = df_pivot.dropna(subset=DataProcessor.CORE_PARAMETERS)
        
        return df_pivot
    
    @staticmethod
    def standardize_features(df: pd.DataFrame, 
                            feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize features (zero mean, unit variance).
        
        Returns:
        - Standardized DataFrame
        - Dictionary with mean and std for each feature
        """
        df = df.copy()
        scaling_params = {}
        
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            scaling_params[col] = {"mean": mean, "std": std}
            df[col] = (df[col] - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
        
        return df, scaling_params
    
    @staticmethod
    def aggregate_by_zone(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify locations into Industrial vs Residential zones.
        
        Simple heuristic: zones with lower humidity tend to be industrial.
        (This would normally use actual zone data)
        """
        df = df.copy()
        
        # Group by location and get median humidity
        location_humidity = df.groupby("location_id")["Humidity"].median()
        
        # Define threshold (50th percentile)
        threshold = location_humidity.median()
        
        df["zone"] = df["location_id"].map(
            lambda loc_id: "Residential" if location_humidity[loc_id] > threshold else "Industrial"
        )
        
        return df
    
    @staticmethod
    def identify_health_violations(df: pd.DataFrame, 
                                   threshold: float = 35) -> pd.DataFrame:
        """
        Mark health threshold violations (PM2.5 > threshold µg/m³).
        """
        df = df.copy()
        df["health_violation"] = df["PM2.5"] > threshold
        return df


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline from raw API data.
    
    Returns:
    - Processed DataFrame
    - Metadata (scaling parameters, zone assignments)
    """
    processor = DataProcessor()
    
    print("Step 1: Cleaning data...")
    df = processor.clean_data(df)
    
    print("Step 2: Pivoting to features...")
    df = processor.pivot_to_features(df)
    
    print("Step 3: Adding zone classification...")
    df = processor.aggregate_by_zone(df)
    
    print("Step 4: Identifying health violations...")
    df = processor.identify_health_violations(df)
    
    print("Step 5: Standardizing features...")
    df_standardized, scaling_params = processor.standardize_features(
        df,
        feature_cols=DataProcessor.CORE_PARAMETERS
    )
    
    metadata = {
        "scaling_params": scaling_params,
        "health_threshold": 35,
        "extreme_hazard_threshold": 200
    }
    
    return df_standardized, metadata
