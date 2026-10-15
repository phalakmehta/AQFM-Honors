"""Mean and sample std (ddof=1) of every scalar metric across single-seed result files.

    python scripts/seed_stats.py OUT.json seed0.json seed1.json seed2.json
"""
import json
import sys

import numpy as np


def leaves(d, prefix=""):
    for k, v in d.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            yield from leaves(v, p)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield p, float(v)


out, files = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f)) for f in files]
keep = ("AQFM", "AQFM_extras", "imputation_24h_outage")
flat = [dict(leaves({k: r[k] for k in keep if k in r})) for r in runs]
common = [k for k in flat[0] if all(k in f for f in flat[1:])]
stats = {k: {"mean": float(np.mean([f[k] for f in flat])),
             "std": float(np.std([f[k] for f in flat], ddof=1)),
             "values": [f[k] for f in flat]} for k in common}
json.dump({"files": files, "metrics": stats}, open(out, "w"), indent=2)
for c in ("PM2.5", "PM10"):
    print(c, " ".join(f"{h} {stats[f'AQFM/{c}/{h}/MAE']['mean']:.2f}±{stats[f'AQFM/{c}/{h}/MAE']['std']:.2f}"
                      for h in ("1h", "6h", "24h", "72h")))
print("saved", out)
