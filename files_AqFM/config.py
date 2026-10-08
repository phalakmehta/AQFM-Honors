"""Configuration for the air-quality spatio-temporal foundation model (AQFM)."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AQFMConfig:
    # ---- data geometry -------------------------------------------------
    # 17 measured channels + 2 derived ratio channels (see physical.py).
    # Ratios live in the input tensor so that their RevIN statistics exist and
    # the decoder can predict them like any other channel.
    n_channels: int = 19
    n_target_channels: int = 8    # channels the decoder predicts
    # which input channels the 8 decoder targets correspond to
    target_index: Tuple[int, ...] = (1, 17, 3, 18, 5, 6, 8, 7)
    n_static: int = 4             # static per-station covariates
    patch_len: int = 16           # 16 x 15min = 4 hours
    n_context_patches: int = 32   # 32 x 4h = 5.33 days of context
    n_horizon_patches: int = 18   # 18 x 4h = 72 hours of forecast
    max_stations: int = 64        # upper bound for the spatial axis

    # ---- indices of met channels used for the wind-aware spatial bias ---
    idx_wind_speed: int = 11
    idx_wind_dir: int = 12

    # ---- architecture --------------------------------------------------
    d_model: int = 384
    n_heads: int = 8
    d_ff: int = 1536
    n_axial_blocks: int = 4       # each block = temporal + variate + spatial
    n_latents_per_station: int = 16
    n_decoder_layers: int = 4
    dropout: float = 0.1

    # ---- output head ---------------------------------------------------
    n_mixture: int = 3            # mixture-of-log-normals components
    min_log_sigma: float = -7.0
    max_log_sigma: float = 3.0

    # ---- time features -------------------------------------------------
    n_fourier_bands: int = 6
    step_minutes: int = 15

    # ---- normalisation -------------------------------------------------
    revin_min_obs: int = 16       # below this many observed steps -> passthrough

    # ---- pretraining mask schedule -------------------------------------
    p_tail_mask: float = 0.50
    p_block_mask: float = 0.35
    p_channel_dropout: float = 0.15
    block_len_lognormal_mu: float = 3.0    # ln(steps); exp(3.0)~20 steps ~5h
    block_len_lognormal_sigma: float = 1.1
    n_blocks_per_sample: Tuple[int, int] = (1, 4)
    channel_dropout_frac: Tuple[float, float] = (0.2, 0.6)

    # ---- loss ----------------------------------------------------------
    n_crps_samples: int = 32
    w_nll: float = 1.0
    w_crps: float = 0.5
    w_physical: float = 0.25

    # Axial attention is cheap in FLOPs but stores a lot of activations
    # (~1.5 GB per sample at the default size). Checkpointing the encoder
    # blocks trades ~30% step time for roughly a 4x memory reduction and is
    # what makes a useful batch size fit on a single 40 GB card.
    grad_checkpoint: bool = False

    @classmethod
    def small(cls, **kw):
        """Reduced config for tests and for single-GPU debugging."""
        base = dict(d_model=128, n_heads=4, d_ff=512, n_axial_blocks=2,
                    n_decoder_layers=2, n_context_patches=16,
                    n_horizon_patches=6, n_latents_per_station=8,
                    n_crps_samples=16)
        base.update(kw)
        return cls(**base)

    @property
    def n_time_feats(self) -> int:
        # 2 * bands absolute-time Fourier + tod(2) + dow(2) + doy(2) + log-gap(1)
        return 2 * self.n_fourier_bands + 7

    @property
    def context_steps(self) -> int:
        return self.patch_len * self.n_context_patches

    @property
    def horizon_steps(self) -> int:
        return self.patch_len * self.n_horizon_patches


DELHI_CHANNELS = [
    "PM2.5", "PM10", "NO2", "NOx", "NO", "Ozone", "SO2", "NH3", "CO",
    "AT", "RH", "WS", "WD", "SR", "BP", "TOT_RF", "Benzene",
]
