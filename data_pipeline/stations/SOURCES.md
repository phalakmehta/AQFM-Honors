# Station lists and station facts: sources

## `<city>.csv`: station coordinates

`delhi.csv`: CPCB, *List of CAAQM Stations (All India)*,
https://app.cpcbccr.com/ccr_docs/caaqms_list_All_India.pdf (dated 2026-07-07; a snapshot of CPCB's CCR station
directory as published on data.gov.in; downloaded 2026-11-04, sha256 `72bd87631dffac40b29ebf111bae8fe5f773943734073b93007d18810f055513`).
Columns `cpcb_name` and `cpcb_code` give the CPCB row each station was taken from. CPCB lists Pusa DPCC and Pusa IMD at
the same point (see `docs/V2_PROGRESS.md`, step 1).

## `<city>_facts.csv`: station facts (model `static` input)

Written by `data_pipeline/station_facts.py`. Raw values; the fixed scaling rule (one rule for every city) is applied by
`station_facts.scaled_static` when the city's arrays are built. No fact uses a station's own readings.

| Fact | Definition | Scaling | Source |
|---|---|---|---|
| `elevation_m` | median height (m) of the DEM pixels within 1 km | m / 1000 (km) | Copernicus DEM GLO-30 |
| `built_frac_1km` | share of land-cover pixels within 1 km that are Built-up (class 50) | as is (0–1) | ESA WorldCover 2021 v200 |
| `green_frac_1km` | share within 1 km that are Tree cover, Shrubland, Grassland, Herbaceous wetland or Mangroves (10, 20, 30, 90, 95); not Cropland (40) | as is (0–1) | ESA WorldCover 2021 v200 |
| `coast_km` | great-circle distance (km) to the nearest coastline segment | log10(1 + km) / 3 (0 at the coast, 1 at 999 km) | Natural Earth 1:10m coastline 5.1.2 |

Notes: GLO-30 is a surface model (buildings and trees included); the 1 km median damps them. Sea outside every DEM
tile counts as 0 m (there are no ocean tiles). Land-cover shares count every valid pixel in the disc, water included.
`coast_km` uses a local flat projection of the nearest segments (about 1 % of the distance).

### Datasets (in `data/global/`, git-ignored; downloaded 2026-11-04)

| File | URL | sha256 |
|---|---|---|
| `copernicus_dem_glo30/Copernicus_DSM_COG_10_N28_00_E076_00_DEM.tif` | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N28_00_E076_00_DEM/Copernicus_DSM_COG_10_N28_00_E076_00_DEM.tif | `827adf975ce68fea8ac5f53844284e7441dac0d74b8b780237ed530bb1e9fde5` |
| `copernicus_dem_glo30/Copernicus_DSM_COG_10_N28_00_E077_00_DEM.tif` | same pattern, `N28_00_E077_00` | `a8e69d432869bf85a63b7cfcaa61b5a770c7705125b3cfde9b892d65dfed98b4` |
| `esa_worldcover_2021_v200/ESA_WorldCover_10m_2021_v200_N27E075_Map.tif` | https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E075_Map.tif | `447f74f2168bb14947a77c99a5a05bdf861300f1e0bdb66dd26e7262620f563f` |
| `natural_earth_10m_coastline/ne_10m_coastline.geojson` | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_10m_coastline.geojson | `6f75ae0e0de157b14946e2255eb1f5486d9a13819032e26d4610852d296788f6` |

Another city needs its own DEM tiles (1° × 1°, named by the south-west corner) and WorldCover tiles (3° × 3°, south-west
corner a multiple of 3); `station_facts.py` stops with an error if a station's tile is missing. The coastline file is global.

### Licences and required notices

- **Copernicus DEM GLO-30** — *Licence for the use of the Copernicus WorldDEM-30* (free, worldwide, without time limit;
  reproduction, distribution, adaptation allowed). Registry: https://registry.opendata.aws/copernicus-dem/ ;
  licence text: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf .
  Required notice for derived data (Art. 6(b)): "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus
  Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved".
  Required sentence (Art. 6(c)): "The organisations in charge of the Copernicus programme by law or by delegation do not
  incur any liability for any use of the Copernicus WorldDEM-30".
- **ESA WorldCover 2021 v200** — CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), DOI 10.5281/zenodo.7254221;
  https://github.com/ESA-WorldCover/esa-worldcover-datasets . Attribution: "© ESA WorldCover project 2021 / Contains
  modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium". Changes made: pixel shares within 1 km.
- **Natural Earth** — public domain (https://www.naturalearthdata.com/about/terms-of-use/). "Made with Natural Earth."
