"""Data fetching from OpenAQ API."""
import requests
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class OpenAQFetcher:
    """Fetch air quality data from OpenAQ API v2."""
    
    BASE_URL = "https://api.openaq.org/v2"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.5):
        """Initialize fetcher with optional API key and rate limit."""
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
    
    def get_locations(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get list of monitoring locations."""
        endpoint = f"{self.BASE_URL}/locations"
        params = {
            "limit": min(limit, 100),  # API max is 100
            "offset": offset
        }
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            response = self.session.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"Error fetching locations: {e}")
            return []
    
    def get_measurements(self, location_id: int, 
                        date_from: str, date_to: str,
                        parameter: str = "") -> List[Dict]:
        """Get measurements for a location in a date range."""
        endpoint = f"{self.BASE_URL}/measurements"
        params = {
            "location_id": location_id,
            "date_from": date_from,
            "date_to": date_to,
            "limit": 100,
            "offset": 0
        }
        
        if parameter:
            params["parameter"] = parameter
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            response = self.session.get(endpoint, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"Error fetching measurements: {e}")
            return []


def fetch_global_data(num_locations: int = 100, 
                     date_from: str = "2025-01-01",
                     date_to: str = "2025-12-31",
                     api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch air quality data from global locations.
    
    Parameters:
    - num_locations: Number of locations to fetch
    - date_from, date_to: Date range (YYYY-MM-DD format)
    - api_key: Optional OpenAQ API key for higher rate limits
    
    Returns DataFrame with columns: location_id, location, city, country, 
                                    parameter, value, unit, date
    """
    fetcher = OpenAQFetcher(api_key=api_key)
    
    print(f"Fetching {num_locations} global locations...")
    
    all_locations = []
    offset = 0
    while len(all_locations) < num_locations:
        batch = fetcher.get_locations(limit=100, offset=offset)
        if not batch:
            break
        all_locations.extend(batch)
        offset += 100
    
    all_locations = all_locations[:num_locations]
    
    if not all_locations:
        print("ERROR: No locations available from API")
        return pd.DataFrame()
    
    print(f"Found {len(all_locations)} locations. Fetching measurements...")
    
    all_data = []
    parameters = ["pm25", "pm10", "no2", "o3", "temperature", "humidity"]
    
    for idx, location in enumerate(all_locations):
        location_id = location.get("id")
        city = location.get("city", "Unknown")
        country = location.get("country", "Unknown")
        location_name = location.get("name", city)
        
        print(f"  [{idx+1}/{len(all_locations)}] {city}, {country}...", end=" ")
        
        location_measurements = []
        for param in parameters:
            measurements = fetcher.get_measurements(
                location_id, date_from, date_to, parameter=param
            )
            location_measurements.extend(measurements)
        
        for measurement in location_measurements:
            all_data.append({
                "location_id": location_id,
                "location": location_name,
                "city": city,
                "country": country,
                "parameter": measurement.get("parameter"),
                "value": measurement.get("value"),
                "unit": measurement.get("unit"),
                "date": measurement.get("date"),
            })
        
        print(f"({len(location_measurements)} records)")
    
    df = pd.DataFrame(all_data)
    return df


def generate_synthetic_data(num_locations: int = 100,
                           num_days: int = 365) -> pd.DataFrame:
    """
    Generate realistic synthetic air quality data for testing and fallback.
    
    Simulates industrial vs residential zones with realistic pollution patterns.
    """
    np.random.seed(42)
    
    # City templates
    cities = [
        ("Beijing", "China"), ("Delhi", "India"), ("Lahore", "Pakistan"),
        ("Shanghai", "China"), ("São Paulo", "Brazil"), ("Mexico City", "Mexico"),
        ("Cairo", "Egypt"), ("Mumbai", "India"), ("Tokyo", "Japan"), ("New York", "USA"),
        ("Los Angeles", "USA"), ("London", "UK"), ("Paris", "France"), ("Berlin", "Germany"),
        ("Madrid", "Spain"), ("Rome", "Italy"), ("Istanbul", "Turkey"), ("Moscow", "Russia"),
        ("Bangkok", "Thailand"), ("Singapore", "Singapore"), ("Hong Kong", "China"),
    ]
    
    all_data = []
    base_date = pd.to_datetime("2025-01-01")
    
    for loc_idx in range(num_locations):
        location_id = loc_idx + 1
        city, country = cities[loc_idx % len(cities)]
        location_name = f"{city}-{loc_idx}"
        
        # Assign zone type stochastically
        zone = np.random.choice(["Industrial", "Residential"], p=[0.4, 0.6])
        
        # Base pollution levels by zone
        base_pm25 = np.random.uniform(20, 80) if zone == "Industrial" else np.random.uniform(10, 40)
        base_pm10 = base_pm25 * 1.5
        base_no2 = np.random.uniform(30, 100) if zone == "Industrial" else np.random.uniform(10, 50)
        base_o3 = np.random.uniform(30, 80)
        
        for day in range(num_days):
            date = base_date + timedelta(days=day)
            
            # Daily cycle (traffic patterns)
            hour = (day * 7) % 24  # Vary hour pattern
            daily_factor = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Peak at hour 7, 18
            
            # Weekly pattern
            day_of_week = date.dayofweek
            weekly_factor = 1 - 0.2 * (day_of_week >= 5)  # Lower on weekends
            
            # Random noise
            noise = np.random.normal(1, 0.1)
            
            # Occasional extreme events (more common in industrial zones)
            extreme_prob = 0.05 if zone == "Industrial" else 0.02  # More extreme events in industrial
            extreme_event = np.random.random() < extreme_prob
            extreme_factor = np.random.uniform(3, 6) if extreme_event else 1
            
            # Calculate values
            combined_factor = daily_factor * weekly_factor * noise * extreme_factor
            
            record_date = date.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            all_data.extend([
                {
                    "location_id": location_id,
                    "location": location_name,
                    "city": city,
                    "country": country,
                    "parameter": "pm25",
                    "value": max(0, np.random.normal(base_pm25 * combined_factor, base_pm25 * 0.2)),
                    "unit": "µg/m³",
                    "date": record_date,
                    "zone": zone,
                },
                {
                    "location_id": location_id,
                    "location": location_name,
                    "city": city,
                    "country": country,
                    "parameter": "pm10",
                    "value": max(0, np.random.normal(base_pm10 * combined_factor, base_pm10 * 0.2)),
                    "unit": "µg/m³",
                    "date": record_date,
                    "zone": zone,
                },
                {
                    "location_id": location_id,
                    "location": location_name,
                    "city": city,
                    "country": country,
                    "parameter": "no2",
                    "value": max(0, np.random.normal(base_no2 * combined_factor, base_no2 * 0.2)),
                    "unit": "µg/m³",
                    "date": record_date,
                    "zone": zone,
                },
                {
                    "location_id": location_id,
                    "location": location_name,
                    "city": city,
                    "country": country,
                    "parameter": "o3",
                    "value": max(0, np.random.normal(base_o3 * np.sqrt(combined_factor), base_o3 * 0.15)),
                    "unit": "µg/m³",
                    "date": record_date,
                    "zone": zone,
                },
                {
                    "location_id": location_id,
                    "location": location_name,
                    "city": city,
                    "country": country,
                    "parameter": "temperature",
                    "value": np.random.normal(20, 10),  # Celsius
                    "unit": "°C",
                    "date": record_date,
                    "zone": zone,
                },
                {
                    "location_id": location_id,
                    "location": location_name,
                    "city": city,
                    "country": country,
                    "parameter": "humidity",
                    "value": np.random.uniform(20, 90),  # Percent
                    "unit": "%",
                    "date": record_date,
                    "zone": zone,
                },
            ])
    
    df = pd.DataFrame(all_data)
    return df
