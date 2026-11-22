"""Tests for the seasonal-aware model variant (use_season=True)."""
import torch
from aqfm.config import AQFMConfig, MONTH_TO_SEASON
from aqfm.data import build_queries, synthetic_raw, AQData, make_sample


def test_month_to_season_mapping():
    """Every month maps to the correct India-specific season."""
    expected = {12: 0, 1: 0, 2: 0,       # winter
                3: 1, 4: 1, 5: 1,         # summer
                6: 2, 7: 2, 8: 2, 9: 2,   # monsoon
                10: 3, 11: 3}             # post-monsoon
    assert MONTH_TO_SEASON == expected


def test_season_embedding_changes_output():
    """With use_season=True, different season indices produce different outputs."""
    from aqfm.model import AQFM
    cfg = AQFMConfig.small(use_season=True)
    model = AQFM(cfg)
    model.eval()

    v, o, t, coords, static, era5 = synthetic_raw(8, 3000, seed=0)
    data = AQData(v, o, t, coords, static, train_end=2600, era5_met=era5)
    sel = data.neighbourhood(0, cfg.n_neighbours)
    sample = make_sample(data, cfg, sel, 100)

    batch0 = {k: v.unsqueeze(0) for k, v in sample.items()}
    batch0["season"] = torch.tensor([0], dtype=torch.long)
    batch1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch0.items()}
    batch1["season"] = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        q0, _, _ = build_queries(cfg, batch0, impute=False)
        q1, _, _ = build_queries(cfg, batch1, impute=False)
        out0 = model(batch0, q0)
        out1 = model(batch1, q1)
    assert not torch.allclose(out0["mu"], out1["mu"], atol=1e-6), \
        "Season embedding should produce different outputs for different seasons"


def test_season_default_off_ignores_season_key():
    """With use_season=False, the model ignores a 'season' key in the batch."""
    from aqfm.model import AQFM
    cfg = AQFMConfig.small(use_season=False)
    model = AQFM(cfg)
    model.eval()

    v, o, t, coords, static, era5 = synthetic_raw(8, 3000, seed=0)
    data = AQData(v, o, t, coords, static, train_end=2600, era5_met=era5)
    sel = data.neighbourhood(0, cfg.n_neighbours)
    sample = make_sample(data, cfg, sel, 100)

    batch_no = {k: v.unsqueeze(0) for k, v in sample.items()}
    # Delete season key if present
    batch_no.pop("season", None)
    batch_with = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_no.items()}
    batch_with["season"] = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        q_no, _, _ = build_queries(cfg, batch_no, impute=False)
        q_with, _, _ = build_queries(cfg, batch_with, impute=False)
        out_no = model(batch_no, q_no)
        out_with = model(batch_with, q_with)
    assert torch.allclose(out_no["mu"], out_with["mu"], atol=1e-7), \
        "With use_season=False, season key should be ignored"
