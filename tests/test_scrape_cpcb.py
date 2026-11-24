"""Tests for CPCB air quality scraper (data_pipeline/scrape_cpcb.py)."""

import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import MagicMock
from data_pipeline.scrape_cpcb import (
    DELHI_RAW_COLUMNS,
    format_to_cpcb_csv,
    check_for_captcha_or_auth,
    fetch_station_data,
    CaptchaRequiredError,
    AuthRequiredError,
)


def test_format_to_cpcb_csv_matches_delhi_schema():
    """Verify generated DataFrame schema matches Delhi CAAQM raw CSV format."""
    dummy_records = [
        {
            "Timestamp": "2024-01-01 00:00:00",
            "PM2.5 (µg/m³)": 45.2,
            "PM10 (µg/m³)": 85.0,
            "AT (°C)": 15.6,
        },
        {
            "Timestamp": "2024-01-01 01:00:00",
            "PM2.5 (µg/m³)": 50.1,
            "PM10 (µg/m³)": 92.4,
            "AT (°C)": 15.1,
        },
    ]
    df = format_to_cpcb_csv("Site Test - MPCB", dummy_records)

    # Check columns
    assert list(df.columns) == DELHI_RAW_COLUMNS
    assert len(df) == 2
    assert (df["Station"] == "Site Test - MPCB").all()
    assert df.loc[0, "PM2.5 (µg/m³)"] == 45.2
    assert pd.isna(df.loc[0, "Benzene (µg/m³)"])


def test_captcha_and_auth_detection_logic():
    """Verify that CAPTCHA or auth challenges trigger the corresponding exceptions."""
    # 1. Normal response passes
    check_for_captcha_or_auth('{"status": "success", "records": []}', 200)

    # 2. CAPTCHA challenges raise CaptchaRequiredError
    with pytest.raises(CaptchaRequiredError):
        check_for_captcha_or_auth("<html>Please solve the reCAPTCHA below to proceed.</html>", 200)

    with pytest.raises(CaptchaRequiredError):
        check_for_captcha_or_auth("Cloudflare Turnstile verification required", 403)

    # 3. Auth/Login challenges raise AuthRequiredError
    with pytest.raises(AuthRequiredError):
        check_for_captcha_or_auth("<html>Please login with your username and password</html>", 200)

    with pytest.raises(AuthRequiredError):
        check_for_captcha_or_auth("Access Denied", 401)

    with pytest.raises(AuthRequiredError):
        check_for_captcha_or_auth("Forbidden", 403)


def test_fetch_station_data_with_mock():
    """Verify fetch_station_data writes a valid CSV from mock API response."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"records": [{"Timestamp": "2024-01-01 00:00:00", "PM2.5 (µg/m³)": 32.5}]}'
    mock_response.json.return_value = {
        "records": [{"Timestamp": "2024-01-01 00:00:00", "PM2.5 (µg/m³)": 32.5}]
    }
    mock_session.post.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = fetch_station_data(
            station_code="site_5102",
            station_name="Vasai West, Mumbai - MPCB",
            start="2024-01-01",
            end="2024-01-02",
            params=["PM2.5", "PM10"],
            out_dir=tmpdir,
            rate_limit=0.01,
            session=mock_session,
        )
        assert os.path.exists(out_csv)
        df = pd.read_csv(out_csv)
        assert list(df.columns) == DELHI_RAW_COLUMNS
        assert len(df) == 1
        assert df.loc[0, "PM2.5 (µg/m³)"] == 32.5
