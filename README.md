# AQFM: Spatio-Temporal Multi-Pollutant Air Quality Forecasting for Delhi

Honours project (Phalak Mehta). Forecasts PM2.5, PM10, NOx, Ozone and other pollutants at 37 Delhi CPCB stations, from 15 minutes to 72 hours ahead, with calibrated uncertainty. It uses a wind-aware axial transformer with a query-based decoder.

**Status (Oct 2026):** data pipeline and baselines done; model implemented and tested (31 tests pass); **training pending GPU access.**

## Start here

| Read | For |
|---|---|
| [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) | The whole project in plain language, including the architecture (basis for the slides) |
| [docs/DATA_REPORT.md](docs/DATA_REPORT.md) | Data sources, coverage, statistics, quality issues |
| [docs/BOTTLENECKS_AND_IMPROVEMENTS.md](docs/BOTTLENECKS_AND_IMPROVEMENTS.md) | What limits the project and how to fix it |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Detailed technical log: bugs fixed, baseline numbers, full roadmap |

## Layout

| Folder | Contents |
|---|---|
| `aqfm/` | The model package (config, layers, model, losses, masking, physical, data, train, evaluate) |
| `tests/` | Tests for `aqfm/` |
| `data_pipeline/` | ERA5 download/unzip/interpolation, station geocoding, preprocessing |
| `baselines/` | Persistence, seasonal naive, climatology, XGBoost, IBM TTM; `results/` holds the JSON results |
| `data/` | Raw and processed data (gitignored except `stations.csv`) |
| `docs/` | Documentation, plans, `figures/` (with the script that regenerates the correlation figure) |
| `presentation/` | Slide-deck builder (`npm install` here, then `node build_pptx_js.js`) and the presented PDF |
| `files_AqFM/` | Original reference implementation and design spec (frozen; do not edit) |
| `_archive_old/` | Superseded material (gitignored) |

## Quick commands (from the repository root)

```bash
python -m pytest tests -q                                   # verify the model code
python data_pipeline/preprocess_aqfm.py                     # rebuild data/preprocessed/
cd baselines && python run_baselines_xgboost.py && cd ..    # baselines run from inside baselines/
python benchmark_gpu.py                                     # speed and memory on this machine
python -m aqfm.train --phase main --out runs/main --grad-checkpoint --device cuda
```

See §13 of the project documentation for the full pipeline.
