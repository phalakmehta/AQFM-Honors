import os
import glob
import pandas as pd
from geopy.geocoders import Nominatim
import time

def main():
    print("Extracting station names from CPCB data...")
    # Find all CSV files in yearly_data
    csv_files = glob.glob(r"C:\Users\phala\Foundation Model\data\yearly_data\*\*.csv")
    
    # Extract unique station names
    station_ids = set()
    for f in csv_files:
        basename = os.path.basename(f)
        st_id = basename.replace(".csv", "")
        station_ids.add(st_id)
        
    print(f"Found {len(station_ids)} unique stations.")
    
    geolocator = Nominatim(user_agent="cpcb_delhi_stations")
    
    results = []
    
    for st_id in sorted(list(station_ids)):
        # Try to parse the real location name from the file name
        # e.g., "ashok_vihar,_delhi_-_dpcc_cleaned" -> "ashok_vihar"
        clean_name = st_id.split(",")[0].replace("_", " ").title()
        
        search_query = f"{clean_name}, Delhi, India"
        
        print(f"Geocoding: {search_query}...")
        try:
            location = geolocator.geocode(search_query, timeout=10)
            if location:
                results.append({
                    "station_id": st_id,
                    "lat": location.latitude,
                    "lon": location.longitude
                })
                print(f"  -> {location.latitude}, {location.longitude}")
            else:
                print(f"  -> NOT FOUND. Trying without 'Delhi, India'")
                location = geolocator.geocode(clean_name, timeout=10)
                if location:
                    results.append({
                        "station_id": st_id,
                        "lat": location.latitude,
                        "lon": location.longitude
                    })
                    print(f"  -> {location.latitude}, {location.longitude}")
                else:
                    print(f"  -> STILL NOT FOUND. Putting default Delhi coords.")
                    # default delhi coordinates as fallback
                    results.append({
                        "station_id": st_id,
                        "lat": 28.6139,
                        "lon": 77.2090
                    })
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({
                "station_id": st_id,
                "lat": 28.6139,
                "lon": 77.2090
            })
            
        time.sleep(1) # Be nice to the Nominatim API
        
    df = pd.DataFrame(results)
    output_path = r"C:\Users\phala\Foundation Model\data\stations.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved coordinates to {output_path}")

if __name__ == '__main__':
    main()
