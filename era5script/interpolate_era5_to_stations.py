import os
import glob
import pandas as pd
import xarray as xr

def load_stations(csv_path):
    # Expected columns: station_id, lat, lon
    return pd.read_csv(csv_path)

def process_month(nc_file, stations_df):
    ds = xr.open_dataset(nc_file)
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
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

def main():
    stations_csv = "stations.csv"
    if not os.path.exists(stations_csv):
        print(f"Error: {stations_csv} not found. Please provide it in the root directory.")
        return
        
    stations_df = load_stations(stations_csv)
    
    raw_dir = "era5_data_raw"
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
        # Resample to 15min and interpolate linearly between hours
        group_15m = group_numeric.resample('15min').interpolate(method='linear')
        group_15m['station_id'] = st_id
        resampled_dfs.append(group_15m.reset_index())
        
    final_df = pd.concat(resampled_dfs, ignore_index=True)
    
    output_parquet = "era5_interpolated.parquet"
    final_df.to_parquet(output_parquet, index=False)
    print(f"Successfully saved {len(final_df)} rows to {output_parquet}")

if __name__ == '__main__':
    main()
