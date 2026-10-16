"""Mean and sample std (ddof=1) of every metric over single-seed deep-baseline results files, in the
format of scripts/seed_stats.py, so scripts/make_tables.py reads it the same way.

    python deep_baselines/seed_stats.py OUT.json seed0.json seed1.json seed2.json
"""
import json
import sys

import numpy as np

MODELS = ("PatchTST", "E-STGCN")          # the results section of each deep baseline


def leaves(d, prefix=""):
    for k, v in d.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            yield from leaves(v, p)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield p, float(v)


def main():
    out, files = sys.argv[1], sys.argv[2:]
    runs = [json.load(open(f)) for f in files]
    flat = [dict(leaves({k: r[k] for k in MODELS if k in r})) for r in runs]
    assert flat[0] and all(f.keys() == flat[0].keys() for f in flat), "files hold different models or metrics"
    stats = {k: {"mean": float(np.mean([f[k] for f in flat])),
                 "std": float(np.std([f[k] for f in flat], ddof=1)),
                 "values": [f[k] for f in flat]} for k in flat[0]}
    json.dump({"files": files, "metrics": stats}, open(out, "w"), indent=2)
    name = next(k.split("/")[0] for k in stats)
    for c in ("PM2.5", "PM10"):
        print(name, c, " ".join(f"{h} {stats[f'{name}/{c}/{h}/MAE']['mean']:.2f}±{stats[f'{name}/{c}/{h}/MAE']['std']:.2f}"
                                for h in ("1h", "6h", "24h", "72h")))
    print("saved", out)


if __name__ == "__main__":
    main()
