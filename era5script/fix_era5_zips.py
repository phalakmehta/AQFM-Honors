import os
import glob
import zipfile
import xarray as xr
import tempfile

def fix_zips():
    raw_dir = "era5_data_raw"
    files = glob.glob(os.path.join(raw_dir, "*.nc"))
    
    for filepath in files:
        if not zipfile.is_zipfile(filepath):
            continue
            
        print(f"Fixing {filepath}...")
        
        # Rename to .zip temporarily to avoid confusion
        zip_path = filepath + ".zip"
        os.rename(filepath, zip_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                
            nc_files = glob.glob(os.path.join(tmpdir, "*.nc"))
            if len(nc_files) > 0:
                # Load all extracted nc files
                datasets = [xr.open_dataset(f) for f in nc_files]
                # Merge them
                merged = xr.merge(datasets)
                # Save as the original filename
                merged.to_netcdf(filepath)
                # Close datasets
                for ds in datasets:
                    ds.close()
                merged.close()
                
        # Delete the zip
        os.remove(zip_path)
        print(f"Successfully merged and saved {filepath}")

if __name__ == "__main__":
    fix_zips()
