"""Data fetching from OpenAQ API."""
import requests
import pandas as pd
import time
from typing import List, Dict, Optional


class OpenAQFetcher:
    """Fetch air quality data from OpenAQ API."""
    
    BASE_URL = "https://api.openaq.org/v1"
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize fetcher with rate limit delay."""
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
    
    def get_locations(self, limit: int = 100) -> List[Dict]:
        """Get list of monitoring locations."""
        endpoint = f"{self.BASE_URL}/locations"
        params = {"limit": limit}
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"Error fetching locations: {e}")
            return []
    
    def get_latest_measurements(self, location_id: int) -> Optional[Dict]:
        """Get latest measurements for a location."""
        endpoint = f"{self.BASE_URL}/latest"
        params = {"location": location_id}
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)
            return data.get("results", [{}])[0] if data.get("results") else None
        except requests.RequestException as e:
            print(f"Error fetching measurements: {e}")
            return None
    
    def get_measurements_by_location(self, location_id: int, 
                                     date_from: str, date_to: str) -> List[Dict]:
        """Get measurements for a location in a date range."""
        endpoint = f"{self.BASE_URL}/measurements"
        params = {
            "location": location_id,
            "date_from": date_from,
            "date_to": date_to,
            "limit": 10000
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"Error fetching measurements: {e}")
            return []


def fetch_global_data(num_locations: int = 100, 
                     date_from: str = "2025-01-01",
                     date_to: str = "2025-12-31") -> pd.DataFrame:
    """
    Fetch air quality data from 100 locations globally.
    
    Returns DataFrame with columns: location, parameter, value, date, city, country
    """
    fetcher = OpenAQFetcher()
    
    # Get available locations
    print(f"Fetching {num_locations} global locations...")
    locations = fetcher.get_locations(limit=num_locations)
    
    if not locations:
        print("No locations available")
        return pd.DataFrame()
    
    print(f"Found {len(locations)} locations. Fetching measurements...")
    
    all_data = []
    for idx, location in enumerate(locations):
        location_id = location.get("id")
        city = location.get("city", "Unknown")
        country = location.get("country", "Unknown")
        
        print(f"  [{idx+1}/{len(locations)}] {city}, {country}...", end=" ")
        
        measurements = fetcher.get_measurements_by_location(
            location_id, date_from, date_to
        )
        
        for measurement in measurements:
            all_data.append({
                "location_id": location_id,
                "location": location.get("location"),
                "city": city,
                "country": country,
                "parameter": measurement.get("parameter"),
                "value": measurement.get("value"),
                "unit": measurement.get("unit"),
                "date": measurement.get("date", {}).get("utc"),
            })
        
        print(f"({len(measurements)} records)")
    
    df = pd.DataFrame(all_data)
    return df
