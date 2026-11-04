"""Geocode station names (Nominatim) -> a stations CSV. One-off helper.

v2 uses official CPCB coordinates instead (data_pipeline/stations/<city>.csv); geocoding by
name is not reliable. If a name is not found the script stops with an error: it never falls
back to a default point or to a search outside the region.

    python data_pipeline/generate_stations_csv.py --raw-dir data/yearly_data \
        --region "Delhi, India" --out data/stations_geocoded.csv
"""
import argparse
import os
import glob
import pandas as pd
import time


def station_ids(raw_dir):
    """Unique station ids from <raw_dir>/<year>/<station_id>.csv"""
    return sorted({os.path.basename(f).replace(".csv", "")
                   for f in glob.glob(os.path.join(raw_dir, "*", "*.csv"))})


def geocode(ids, region, geolocator, pause=1.0):
    """[(station_id, lat, lon)]; raises if any station is not found within `region`."""
    results, missing = [], []
    for st_id in ids:
        # e.g. "ashok_vihar,_delhi_-_dpcc_cleaned" -> "Ashok Vihar"
        clean_name = st_id.split(",")[0].replace("_", " ").title()
        search_query = f"{clean_name}, {region}"
        print(f"Geocoding: {search_query}...")
        location = geolocator.geocode(search_query, timeout=10)
        if location:
            results.append({"station_id": st_id, "lat": location.latitude, "lon": location.longitude})
            print(f"  -> {location.latitude}, {location.longitude}")
        else:
            missing.append(st_id)
            print("  -> NOT FOUND")
        time.sleep(pause)  # Be nice to the Nominatim API
    if missing:
        raise SystemExit(f"not found in '{region}': {missing}. Add their coordinates by hand "
                         f"from an official source; there is no default location.")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/yearly_data")
    ap.add_argument("--region", default="Delhi, India", help="appended to every station name")
    ap.add_argument("--out", required=True, help="never data/stations.csv: that v1 file stays unchanged")
    args = ap.parse_args(argv)
    from geopy.geocoders import Nominatim
    ids = station_ids(args.raw_dir)
    print(f"Found {len(ids)} unique stations.")
    rows = geocode(ids, args.region, Nominatim(user_agent="aqfm_station_geocoder"))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved coordinates to {args.out}")


if __name__ == '__main__':
    main()
