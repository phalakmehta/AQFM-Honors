"""Regenerate docs/figures/correlation_heatmap.png from data/preprocessed/.

Pearson correlation between channels over all stations, hourly sub-sample
(every 4th 15-min step), using for each pair only the steps where both are
observed. AT and RH are CPCB readings with gaps filled from ERA5 (IST-aligned).

    python docs/figures/make_correlation_heatmap.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "preprocessed")
CHANNELS = ["PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO", "AT", "RH"]

order = json.load(open(os.path.join(DATA, "preprocessing_report.json")))["channel_order"]
v = np.load(os.path.join(DATA, "values.npy"), mmap_mode="r")
o = np.load(os.path.join(DATA, "observed.npy"), mmap_mode="r")
x = {c: np.asarray(v[:, order.index(c), ::4], np.float64).ravel() for c in CHANNELS}
m = {c: np.asarray(o[:, order.index(c), ::4]).ravel() > 0 for c in CHANNELS}

n = len(CHANNELS)
corr = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        ok = m[CHANNELS[i]] & m[CHANNELS[j]]
        corr[i, j] = corr[j, i] = np.corrcoef(x[CHANNELS[i]][ok], x[CHANNELS[j]][ok])[0, 1]

BG, PANEL, FG = "#0b1426", "#13213a", "#c8d6ea"
fig, ax = plt.subplots(figsize=(11, 9.5), facecolor=BG)
ax.set_facecolor(PANEL)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n), CHANNELS, rotation=45, ha="right", color=FG, fontsize=11)
ax.set_yticks(range(n), CHANNELS, color=FG, fontsize=11)
for i in range(n):
    for j in range(n):
        strong = abs(corr[i, j]) >= 0.5
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if strong else "#1d2b44", fontweight="bold" if strong else "normal")
ax.set_title("Cross-Pollutant Correlation Matrix (Delhi CPCB, 2018-2025)\n"
             "Pearson, hourly sub-sample; AT/RH = CPCB with ERA5 gap-fill", color="white",
             fontsize=14, fontweight="bold", pad=14)
for s in ax.spines.values():
    s.set_color("#2c3e5c")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors=FG)
cb.outline.set_edgecolor("#2c3e5c")
fig.tight_layout()
out = os.path.join(HERE, "correlation_heatmap.png")
fig.savefig(out, dpi=150, facecolor=BG)
print("saved", out)
print(np.round(corr, 2))
