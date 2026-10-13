"""Configuration and channel layout for AQFM.

Channel bookkeeping lives here and nowhere else. Every other module refers to
channels by name through `CH`, so adding or reordering a channel is a one-file
change.
"""

from dataclasses import dataclass
from typing import Tuple

# values.npy / observed.npy axis 1, exactly as written by preprocess_aqfm.py
RAW_CHANNELS = (
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF", "Benzene", "BLH",
)

# Model input channels. WS/WD become U/V because a direction in degrees cannot
# be averaged or normalised (350 deg and 10 deg would average to 180 deg).
# The two fractions are logits of PM2.5/PM10 and NO/NOx (both in ppb).
INPUT_CHANNELS = (
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "AT", "RH", "U", "V", "SR", "BP", "TOT_RF", "Benzene", "BLH",
    "PM_FRAC", "NOX_FRAC",
)
CH = {name: i for i, name in enumerate(INPUT_CHANNELS)}

# era5_met.npy axis 1 (pure ERA5, IST-aligned): the known-future covariates
ERA5_MET_CHANNELS = ("AT", "RH", "U", "V", "SR", "BP", "TOT_RF", "BLH")

# Right-skewed non-negative channels, modelled in log1p space
LOG_CHANNELS = tuple(CH[c] for c in (
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "TOT_RF", "Benzene", "BLH"))

# Channels measured by the CPCB analysers. A station outage removes these.
# Met channels are ERA5-backed (never missing at inference), so outages leave
# them visible. The fractions are not listed: their visibility always follows
# their constituents (see masking.enforce_derived_visibility).
OUTAGE_CHANNELS = tuple(CH[c] for c in (
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO", "Benzene"))

# Physical reparameterisation: (base slot name, fraction slot name)
PM_PAIR = ("PM10", "PM_FRAC")       # PM2.5 = PM10 * sigmoid(frac)
NOX_PAIR = ("NOx", "NOX_FRAC")      # NO = NOx * f, NO2 = NOx * (1 - f), in ppb


@dataclass
class AQFMConfig:
    # ---- data geometry -------------------------------------------------
    n_channels: int = len(INPUT_CHANNELS)
    target_channels: Tuple[str, ...] = (
        "PM10", "PM_FRAC", "NOx", "NOX_FRAC", "Ozone", "SO2", "CO", "NH3")
    n_static: int = 4
    n_met_future: int = len(ERA5_MET_CHANNELS)
    patch_len: int = 16           # 16 x 15 min = 4 h
    n_context_patches: int = 32   # 5.33 days
    n_horizon_patches: int = 18   # 72 h
    step_hours: float = 0.25
    n_neighbours: int = 12

    # ---- architecture --------------------------------------------------
    d_model: int = 384
    n_heads: int = 8
    d_ff: int = 1536
    n_axial_blocks: int = 4
    n_latents_per_station: int = 16
    n_decoder_layers: int = 4
    dropout: float = 0.1
    n_fourier_bands: int = 6      # log-spaced periods, 3 h .. 1 year
    wind_length_scale_km: float = 25.0

    # ---- switches used by the planned ablations -------------------------
    use_future_met: bool = True        # ERA5 as perfect-prognosis future met
    use_last_obs: bool = True          # decoder anchors on each series' latest visible value
    use_spatial_attention: bool = True
    spatial_bias: str = "wind"         # "wind" | "distance" | "none"

    # ---- output head ---------------------------------------------------
    n_mixture: int = 3
    min_log_sigma: float = -7.0
    max_log_sigma: float = 3.0

    # ---- normalisation -------------------------------------------------
    revin_min_obs: int = 16       # fewer visible steps -> global train stats

    # ---- pre-training mask schedule ------------------------------------
    p_tail_mask: float = 0.50
    p_block_mask: float = 0.35
    p_channel_dropout: float = 0.15
    # Defaults give a median outage of ~5 h. Overwrite with values fitted on
    # the training split (masking.fit_block_lognormal) before pre-training.
    block_len_lognormal_mu: float = 3.0
    block_len_lognormal_sigma: float = 1.1
    n_blocks_per_sample: Tuple[int, int] = (1, 4)
    channel_dropout_frac: Tuple[float, float] = (0.2, 0.6)
    forecast_targets_always: bool = True   # False only for the random-masking ablation
    max_impute_positions: int = 48         # (station, patch) slots scored per sample

    # ---- loss ----------------------------------------------------------
    n_crps_samples: int = 32      # only for the reconstructed (physical) CRPS
    w_nll: float = 1.0
    w_crps: float = 0.5
    w_physical: float = 0.25
    episode_alpha: float = 0.0    # >0 up-weights PM2.5 above episode_threshold
    episode_threshold: float = 250.0

    grad_checkpoint: bool = False

    @classmethod
    def small(cls, **kw):
        """Reduced config for tests and CPU debugging."""
        base = dict(d_model=64, n_heads=4, d_ff=128, n_axial_blocks=2,
                    n_decoder_layers=2, n_context_patches=8,
                    n_horizon_patches=4, n_latents_per_station=4,
                    n_neighbours=4, max_impute_positions=8,
                    n_crps_samples=16, dropout=0.0)
        base.update(kw)
        return cls(**base)

    # ---- derived -------------------------------------------------------
    @property
    def target_index(self) -> Tuple[int, ...]:
        return tuple(CH[c] for c in self.target_channels)

    @property
    def n_target_channels(self) -> int:
        return len(self.target_channels)

    @property
    def context_steps(self) -> int:
        return self.patch_len * self.n_context_patches

    @property
    def horizon_steps(self) -> int:
        return self.patch_len * self.n_horizon_patches

    @property
    def window_steps(self) -> int:
        return self.context_steps + self.horizon_steps

    @property
    def patch_hours(self) -> float:
        return self.patch_len * self.step_hours

    def physical_pairs(self):
        """[(name, base_slot, frac_slot)] for each reparameterised pair present
        among the targets."""
        out = []
        for name, (base, frac) in (("PM", PM_PAIR), ("NOX", NOX_PAIR)):
            if base in self.target_channels and frac in self.target_channels:
                out.append((name, self.target_channels.index(base),
                            self.target_channels.index(frac)))
        return out
