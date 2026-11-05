"""ERA5 monthly NetCDFs -> bilinear values at each station, hourly -> 15 min -> one parquet.

    python data_pipeline/interpolate_era5_to_stations.py            # v1 defaults (Delhi)
    python data_pipeline/interpolate_era5_to_stations.py --stations data_pipeline/stations/delhi.csv \
        --era5-dir data/era5_data_raw --out data/delhi_era5_interpolated.parquet
"""
import argparse
import os
import glob
import pandas as pd
import xarray as xr

class OutsideERA5Box(ValueError):
    pass


def load_stations(csv_path):
    # Expected columns: station_id, lat, lon
    return pd.read_csv(csv_path)

def process_month(nc_file, stations_df):
    ds = xr.open_dataset(nc_file)
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    # a station outside the downloaded box would get NaN weather silently: stop instead
    la, lo = ds['latitude'].values, ds['longitude'].values
    out = ((stations_df['lat'] < la.min()) | (stations_df['lat'] > la.max())
           | (stations_df['lon'] < lo.min()) | (stations_df['lon'] > lo.max()))
    if out.any():
        raise OutsideERA5Box(f"{os.path.basename(nc_file)}: stations outside the ERA5 box "
                         f"lat {la.min()}..{la.max()}, lon {lo.min()}..{lo.max()}: "
                         + ", ".join(stations_df.loc[out, 'station_id']))
    st_lats = xr.DataArray(stations_df['lat'].values, dims='station')
    st_lons = xr.DataArray(stations_df['lon'].values, dims='station')
    st_ids = stations_df['station_id'].values
    
    # Interpolate using xarray's built-in linear interpolation
    interpolated = ds.interp(latitude=st_lats, longitude=st_lons, method='linear')
    
    # Convert to dataframe
    df = interpolated.to_dataframe().reset_index()
    
    # Map station dimension index back to the actual station_id
    df['station_id'] = df['station'].map(lambda idx: st_ids[idx])
    
    # Keep time, station_id, and all ERA5 variables
    keep_cols = ['time', 'station_id'] + list(ds.data_vars.keys())
    df = df[keep_cols]
    
    return df

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="data/stations.csv", help="CSV with station_id, lat, lon")
    ap.add_argument("--era5-dir", default="data/era5_data_raw", help="monthly ERA5 NetCDFs (fetch_era5.py)")
    ap.add_argument("--out", default="data/era5_interpolated.parquet")
    args = ap.parse_args(argv)
    stations_csv = args.stations
    if not os.path.exists(stations_csv):
        print(f"Error: {stations_csv} not found. Please provide it in data/.")
        return
        
    stations_df = load_stations(stations_csv)
    if stations_df[['lat', 'lon']].isna().any(axis=None):
        raise ValueError("stations without coordinates in " + stations_csv)
    
    raw_dir = args.era5_dir
    nc_files = sorted(glob.glob(os.path.join(raw_dir, "*.nc")))
    
    if not nc_files:
        print(f"No .nc files found in {raw_dir}.")
        return
        
    all_data = []
    for f in nc_files:
        print(f"Processing {f}...")
        try:
            df_month = process_month(f, stations_df)
            all_data.append(df_month)
        except OutsideERA5Box:
            raise
        except Exception as e:
            print(f"Failed processing {f}: {e}")
        
    if not all_data:
        print("No data was processed.")
        return
        
    combined = pd.concat(all_data, ignore_index=True)
    
    print("Resampling hourly data to 15-minute intervals...")
    combined.set_index('time', inplace=True)
    
    resampled_dfs = []
    for st_id, group in combined.groupby('station_id'):
        group_numeric = group.drop(columns=['station_id', 'station'], errors='ignore')
        # Resample to 15min and interpolate linearly between hours. limit=3 fills
        # only the 3 slots inside an hour, so missing months stay NaN instead of
        # becoming a straight line across the gap.
        group_15m = group_numeric.resample('15min').interpolate(method='linear', limit=3,
                                                                limit_area='inside')
        group_15m['station_id'] = st_id
        resampled_dfs.append(group_15m.reset_index())
        
    final_df = pd.concat(resampled_dfs, ignore_index=True)
    
    output_parquet = args.out
    final_df.to_parquet(output_parquet, index=False)
    print(f"Successfully saved {len(final_df)} rows to {output_parquet}")

if __name__ == '__main__':
    main()
