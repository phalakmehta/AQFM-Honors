"""ERA5 single-level download (CDS API), one NetCDF per month.

    python data_pipeline/fetch_era5.py                      # v1 defaults: Delhi box, 2021-2025
    python data_pipeline/fetch_era5.py --city <city> --area <N> <W> <S> <E> \
        --start 2017-12 --end 2025-12 --out-dir data/era5_<city>_raw
"""
import argparse
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

def fetch_month(year, month, output_dir, area=AREA, city="delhi"):
    """area: [North, West, South, East] degrees; files are era5_<city>_<year>_<month>.nc."""
    import cdsapi
    c = cdsapi.Client()
    
    filename = os.path.join(output_dir, f"era5_{city}_{year}_{month:02d}.nc")
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
                    'area': list(area),
                },
                filename)
            print(f"Saved to {filename}", flush=True)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {year}-{month:02d}: {e}", flush=True)
            print("Wi-Fi might be down or request failed. Waiting 60 seconds before retrying...", flush=True)
            time.sleep(60)

def months(start, end):
    """'YYYY-MM' .. 'YYYY-MM' inclusive -> [(year, month), ...]"""
    (y0, m0), (y1, m1) = (map(int, s.split("-")) for s in (start, end))
    return [(k // 12, k % 12 + 1) for k in range(y0 * 12 + m0 - 1, y1 * 12 + m1)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="delhi", help="used in the file names")
    ap.add_argument("--area", nargs=4, type=float, default=AREA, metavar=("N", "W", "S", "E"))
    ap.add_argument("--start", default="2021-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2025-12", help="last month, YYYY-MM")
    ap.add_argument("--out-dir", default="data/era5_data_raw")
    args = ap.parse_args(argv)
    n, w, s_, e = args.area
    if not (n > s_ and e > w):
        ap.error("--area is North West South East")
    os.makedirs(args.out_dir, exist_ok=True)
    for year, month in months(args.start, args.end):
        fetch_month(year, month, args.out_dir, args.area, args.city)


if __name__ == '__main__':
    main()
