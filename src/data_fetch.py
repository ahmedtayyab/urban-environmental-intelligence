"""Synthetic data generation for air quality analysis."""
import pandas as pd
import numpy as np
from datetime import timedelta


def generate_synthetic_data(num_locations: int = 100,
                           num_days: int = 365,
                           hourly: bool = True) -> pd.DataFrame:
    """
    Generate realistic synthetic air quality data for testing and fallback.
    
    Simulates industrial vs residential zones with realistic pollution patterns.
    
    If hourly=True: generates 24 hourly readings per day (876K records for 100 stations, 365 days)
    If hourly=False: generates 1 daily reading per day (legacy mode)
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
        
        # Monthly temperature pattern (Northern hemisphere)
        month_temps = [5, 7, 12, 18, 24, 28, 30, 29, 24, 18, 10, 6]
        
        for day in range(num_days):
            date = base_date + timedelta(days=day)
            month_idx = date.month - 1
            base_temp = month_temps[month_idx]
            
            # Weekly pattern
            day_of_week = date.dayofweek
            weekly_factor = 1 - 0.15 * (day_of_week >= 5)  # Lower on weekends
            
            if hourly:
                # Generate 24 hourly readings per day
                for hour in range(24):
                    # Realistic traffic-based hourly pattern
                    # Peaks at rush hours: 8-9 AM, 5-6 PM
                    if 8 <= hour <= 9 or 17 <= hour <= 18:
                        hour_factor = 1.4  # Rush hour peak
                    elif 6 <= hour <= 22:
                        hour_factor = 1.0  # Daytime elevated
                    else:
                        hour_factor = 0.6  # Nighttime low
                    
                    # Random noise
                    noise = np.random.normal(1, 0.08)
                    
                    # Occasional extreme events
                    extreme_prob = 0.003 if zone == "Industrial" else 0.001
                    extreme_event = np.random.random() < extreme_prob
                    extreme_factor = np.random.uniform(2.5, 5) if extreme_event else 1
                    
                    combined_factor = hour_factor * weekly_factor * noise * extreme_factor
                    
                    # Temperature with diurnal cycle
                    temp = base_temp + 8 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 1)
                    humidity = 50 + 30 * np.sin(2 * np.pi * (hour + 6) / 24) + np.random.normal(0, 5)
                    
                    record_date = (date + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    all_data.extend([
                        {
                            "location_id": location_id,
                            "location": location_name,
                            "city": city,
                            "country": country,
                            "parameter": "pm25",
                            "value": max(0, np.random.normal(base_pm25 * combined_factor, base_pm25 * 0.15)),
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
                            "value": max(0, np.random.normal(base_pm10 * combined_factor, base_pm10 * 0.15)),
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
                            "value": max(0, np.random.normal(base_no2 * combined_factor, base_no2 * 0.15)),
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
                            "value": max(0, np.random.normal(base_o3 * np.sqrt(combined_factor), base_o3 * 0.12)),
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
                            "value": temp,
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
                            "value": max(0, min(100, humidity)),
                            "unit": "%",
                            "date": record_date,
                            "zone": zone,
                        },
                    ])
            else:
                # Legacy: Daily readings (one per day)
                # Daily cycle (traffic patterns)
                hour = (day * 7) % 24  # Vary hour pattern
                daily_factor = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Peak at hour 7, 18
                
                # Random noise
                noise = np.random.normal(1, 0.1)
                
                # Occasional extreme events
                extreme_prob = 0.05 if zone == "Industrial" else 0.02
                extreme_event = np.random.random() < extreme_prob
                extreme_factor = np.random.uniform(3, 6) if extreme_event else 1
                
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
                        "value": base_temp + np.random.normal(0, 5),
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
                        "value": np.random.uniform(20, 90),
                        "unit": "%",
                        "date": record_date,
                        "zone": zone,
                    },
                ])
    
    df = pd.DataFrame(all_data)
    return df
