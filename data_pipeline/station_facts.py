"""Station facts for the model's `static` input (v2, item B3).

Facts that exist for any place in India and mean the same in every city. They
come from free global datasets around the station's coordinates; nothing is
taken from a station's own readings. Each fact has one fixed scaling rule for
all cities (no per-city z-score), so the same number means the same thing in
Delhi and in Mumbai.

The facts are a registry (FACTS); DEFAULT_FACTS is the set in use, in the order
of the static columns. To swap a fact, add it to FACTS and pass --facts.
AQFMConfig.n_static must equal the number of facts.

Computing the facts needs rasterio, which is not in the project environment, so
this runs once per city in a separate tools venv and writes a small CSV of raw
values; preprocess_aqfm.py (project venv) reads that CSV and applies
`scaled_static`, which needs only numpy/pandas:

    /home/phalak/aqfm/.venv_geo/bin/python data_pipeline/station_facts.py \
        --stations data_pipeline/stations/delhi.csv --out data_pipeline/stations/delhi_facts.csv

Datasets (downloaded into --global-dir; sources, licences and notices in
data_pipeline/stations/SOURCES.md):
  copernicus_dem_glo30/        Copernicus DEM GLO-30 tiles (Copernicus_DSM_COG_10_<N..>_<E..>_DEM.tif)
  esa_worldcover_2021_v200/    ESA WorldCover 10 m 2021 v200 tiles (..._<N..E..>_Map.tif)
  natural_earth_10m_coastline/ ne_10m_coastline.geojson (Natural Earth 5.1.2)
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

RADIUS_KM = 1.0
KM_PER_DEG = 111.32
# ESA WorldCover classes (legend embedded in the tiles)
BUILT = (50,)                       # Built-up
GREEN = (10, 20, 30, 90, 95)        # Tree cover, Shrubland, Grassland, Herbaceous wetland, Mangroves
                                    # (not 40 Cropland: seasonal, often bare; not 100 Moss/lichen: absent in India)


def _disc(path_glob, lat, lon, radius_km, fill=None):
    """Pixel values whose centres lie within radius_km of (lat, lon), from the
    tiles matching path_glob. fill: value for parts of the disc outside every
    tile (None = error)."""
    import rasterio
    from rasterio.merge import merge
    dlat = radius_km / KM_PER_DEG
    dlon = radius_km / (KM_PER_DEG * np.cos(np.radians(lat)))
    bounds = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    tiles = []
    for f in sorted(glob.glob(path_glob)):
        with rasterio.open(f) as r:
            b = r.bounds
            if b.left < bounds[2] and b.right > bounds[0] and b.bottom < bounds[3] and b.top > bounds[1]:
                tiles.append(f)
    if not any(_contains(f, lat, lon) for f in tiles):
        raise FileNotFoundError(f"no tile in {path_glob} contains ({lat}, {lon}): download it first")
    srcs = [rasterio.open(f) for f in tiles]
    try:
        nodata = srcs[0].nodata
        arr, tr = merge(srcs, bounds=bounds, nodata=nodata if nodata is not None else np.nan)
    finally:
        for s in srcs:
            s.close()
    a = arr[0]
    rows, cols = np.indices(a.shape)
    x, y = tr * (cols + 0.5, rows + 0.5)                         # pixel centres (lon, lat)
    d = np.hypot((x - lon) * KM_PER_DEG * np.cos(np.radians(lat)), (y - lat) * KM_PER_DEG)
    inside = d <= radius_km
    v = a[inside]
    missing = np.isnan(v) if nodata is None else (v == nodata)
    if missing.any():
        if fill is None:
            v = v[~missing]
        else:
            v = np.where(missing, fill, v)
    return v


def _contains(f, lat, lon):
    import rasterio
    with rasterio.open(f) as r:
        b = r.bounds
        return b.left <= lon < b.right and b.bottom <= lat < b.top


def elevation_m(lat, lon, gdir):
    """Median Copernicus DEM GLO-30 height (m) within 1 km. GLO-30 is a surface
    model (includes buildings and trees); the median over the disc damps them.
    Sea outside any tile counts as 0 m (no ocean tiles exist)."""
    v = _disc(os.path.join(gdir, "copernicus_dem_glo30", "*.tif"), lat, lon, RADIUS_KM, fill=0.0)
    return float(np.median(v))


def _landcover_share(lat, lon, gdir, classes):
    v = _disc(os.path.join(gdir, "esa_worldcover_2021_v200", "*_Map.tif"), lat, lon, RADIUS_KM)
    return float(np.isin(v, classes).mean())


def built_frac_1km(lat, lon, gdir):
    """Share of ESA WorldCover 2021 pixels within 1 km that are Built-up."""
    return _landcover_share(lat, lon, gdir, BUILT)


def green_frac_1km(lat, lon, gdir):
    """Share of ESA WorldCover 2021 pixels within 1 km that are tree cover, shrubland,
    grassland, herbaceous wetland or mangroves."""
    return _landcover_share(lat, lon, gdir, GREEN)


_COAST = {}


def coast_km(lat, lon, gdir):
    """Great-circle distance (km) to the nearest Natural Earth 1:10m coastline
    segment (accuracy ~1% of the distance: local flat projection of the
    nearest segments)."""
    path = os.path.join(gdir, "natural_earth_10m_coastline", "ne_10m_coastline.geojson")
    if path not in _COAST:
        g = json.load(open(path))
        a, b = [], []
        for ft in g["features"]:
            c = np.asarray(ft["geometry"]["coordinates"], np.float64)   # (n, 2) lon, lat
            a.append(c[:-1])
            b.append(c[1:])
        _COAST[path] = np.concatenate(a), np.concatenate(b)
    a, b = _COAST[path]
    hav = lambda p: _haversine_km(lat, lon, p[:, 1], p[:, 0])
    da, db = hav(a), hav(b)
    seg_len = _haversine_km(a[:, 1], a[:, 0], b[:, 1], b[:, 0])
    cand = np.minimum(da, db) <= min(da.min(), db.min()) + seg_len.max()
    k = np.cos(np.radians(lat))
    to_xy = lambda p: np.stack([(p[:, 0] - lon) * KM_PER_DEG * k, (p[:, 1] - lat) * KM_PER_DEG], -1)
    pa, pb = to_xy(a[cand]), to_xy(b[cand])
    ab = pb - pa
    t = np.clip(-(pa * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    seg = float(np.linalg.norm(pa + t[:, None] * ab, axis=-1).min())
    return min(seg, float(min(da.min(), db.min())))          # never above the exact nearest-vertex distance


def _haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, np.float64)) for x in (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# name -> (compute(lat, lon, global_dir) -> raw value, fixed scaling rule for every city)
FACTS = {
    "elevation_m": (elevation_m, lambda v: v / 1000.0),                     # km
    "built_frac_1km": (built_frac_1km, lambda v: v),                         # share 0..1
    "green_frac_1km": (green_frac_1km, lambda v: v),                         # share 0..1
    "coast_km": (coast_km, lambda v: np.log10(1.0 + v) / 3.0),               # 0 at the coast, 1 at 999 km
}
DEFAULT_FACTS = ("elevation_m", "built_frac_1km", "green_frac_1km", "coast_km")


def scaled_static(facts_df, station_order, names=DEFAULT_FACTS):
    """(S, len(names)) float32 static array in station_order, each fact scaled by
    its fixed rule. Every station must have every fact (no defaults)."""
    df = facts_df.set_index("station_id")
    missing = [s for s in station_order if s not in df.index]
    if missing:
        raise KeyError(f"no station facts for {missing}")
    cols = []
    for n in names:
        raw = df.loc[list(station_order), n].to_numpy(np.float64)
        if np.isnan(raw).any():
            raise ValueError(f"fact {n} is missing for some stations")
        cols.append(FACTS[n][1](raw))
    return np.stack(cols, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stations", required=True, help="CSV with station_id, lat, lon")
    ap.add_argument("--out", required=True, help="output CSV of raw facts")
    ap.add_argument("--global-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "..", "data", "global"))
    ap.add_argument("--facts", nargs="+", default=list(DEFAULT_FACTS), choices=sorted(FACTS))
    args = ap.parse_args()
    st = pd.read_csv(args.stations)
    if st[["lat", "lon"]].isna().any().any():
        raise SystemExit("stations without coordinates: " + ", ".join(st.station_id[st.lat.isna() | st.lon.isna()]))
    rows = []
    for _, r in st.iterrows():
        row = {"station_id": r.station_id, "lat": r.lat, "lon": r.lon}
        for n in args.facts:
            row[n] = FACTS[n][0](float(r.lat), float(r.lon), args.global_dir)
        rows.append(row)
        print("  " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} stations x {len(args.facts)} facts -> {args.out}")


if __name__ == "__main__":
    main()
