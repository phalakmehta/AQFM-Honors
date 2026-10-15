# AQFM: Spatio-Temporal Multi-Pollutant Air Quality Forecasting for Delhi

Honours project (Phalak Mehta). Forecasts PM2.5, PM10, NOx, Ozone and other pollutants at 37 Delhi CPCB stations, from 15 minutes to 72 hours ahead, with calibrated uncertainty. It uses a wind-aware axial transformer with a query-based decoder.

**Status (Oct 2026):** data pipeline, baselines and the model are done; AQFM is trained and evaluated on the 2025 test year (45 tests pass). Ablations, deep competitors and the write-up are next. All result tables: [results/TABLES.md](results/TABLES.md); submission checklist: [SUBMISSION.md](SUBMISSION.md).

## Start here

| Read | For |
|---|---|
| [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) | The whole project in plain language, including the architecture (basis for the slides) |
| [docs/DATA_REPORT.md](docs/DATA_REPORT.md) | Data sources, coverage, statistics, quality issues |
| [docs/BOTTLENECKS_AND_IMPROVEMENTS.md](docs/BOTTLENECKS_AND_IMPROVEMENTS.md) | What limits the project and how to fix it |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Detailed technical log: bugs fixed, baseline numbers, full roadmap |
| [results/TABLES.md](results/TABLES.md) | Every result table (generated from the results files) |
| [SUBMISSION.md](SUBMISSION.md) | What the paper needs, where each item lives, and its status |

## Layout

| Folder | Contents |
|---|---|
| `aqfm/` | The model package (config, layers, model, losses, masking, physical, data, train, evaluate) |
| `tests/` | Tests for `aqfm/` |
| `data_pipeline/` | ERA5 download/unzip/interpolation, station geocoding, preprocessing |
| `baselines/` | Persistence, seasonal naive, climatology, XGBoost, IBM TTM; `results/` holds the JSON results |
| `results/` | AQFM results (JSON), `TABLES.md` (generated) and summaries |
| `runs/` | Training and evaluation logs and settings (model files are on Google Drive) |
| `scripts/` | `manifest.json` (every model and result with its command and checksum), `check_submission.py`, `make_tables.py`, `seed_stats.py` |
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
python -m aqfm.train --phase main --out runs/main_s0_do3 --device cuda --batch-size 8 --steps 12000 \
    --weight-decay 0.05 --patience 3 --dropout 0.3 --compile --seed 0    # final recipe
python scripts/check_submission.py                          # check data, models, results, tables, tests, git
```

See §13 of the project documentation for the full pipeline.
