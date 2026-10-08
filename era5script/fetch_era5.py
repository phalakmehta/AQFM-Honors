import cdsapi
import os
import time

# Delhi NCR roughly bounding box [North, West, South, East]
AREA = [29.0, 76.5, 28.2, 77.5]

VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards',
    'surface_pressure',
    'total_precipitation',
    'boundary_layer_height',
]

def fetch_month(year, month, output_dir):
    c = cdsapi.Client()
    
    filename = os.path.join(output_dir, f"era5_delhi_{year}_{month:02d}.nc")
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping.", flush=True)
        return
        
    print(f"Fetching data for {year}-{month:02d}...", flush=True)
    
    # Days in month logic
    if month in [1, 3, 5, 7, 8, 10, 12]:
        days = [f"{d:02d}" for d in range(1, 32)]
    elif month in [4, 6, 9, 11]:
        days = [f"{d:02d}" for d in range(1, 31)]
    else:
        # Leap year check for Feb
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days = [f"{d:02d}" for d in range(1, 30)]
        else:
            days = [f"{d:02d}" for d in range(1, 29)]
            
    max_retries = 1000
    for attempt in range(max_retries):
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': VARIABLES,
                    'year': str(year),
                    'month': f"{month:02d}",
                    'day': days,
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': AREA,
                },
                filename)
            print(f"Saved to {filename}", flush=True)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {year}-{month:02d}: {e}", flush=True)
            print("Wi-Fi might be down or request failed. Waiting 60 seconds before retrying...", flush=True)
            time.sleep(60)

if __name__ == '__main__':
    output_dir = "era5_data_raw"
    os.makedirs(output_dir, exist_ok=True)
    
    for year in range(2021, 2026):
        for month in range(1, 13):
            fetch_month(year, month, output_dir)
