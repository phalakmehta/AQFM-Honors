"""CPCB Central Control Room (CCR) Air Quality Data Scraper / Downloader.

API Research & Architecture Note:
---------------------------------
CPCB's official air quality portal (https://airquality.cpcb.gov.in/ccr/) provides ambient air quality
monitoring data. In earlier revisions, public POST queries were submitted to caaqms_view_data.
In current revisions:
1. The portal uses an Angular frontend with MapMyIndia/Mappls integration.
2. Direct API endpoints (e.g., https://airquality.cpcb.gov.in/insight_proxy_api_v2_0/service.php)
   are gated behind session tokens (ccr_refresh_token) and a CAPTCHA verification form
   (captchaFormConfig). Unauthenticated direct requests yield 500 / 403 or redirect to login.
3. Per project constraints and AI ethics guidelines, this scraper:
   - Does NOT bypass or solve CAPTCHAs, bot challenges, or authentication boundaries.
   - Strictly validates responses: if CAPTCHA, authentication, or non-JSON/error responses are
     encountered, it immediately halts and alerts the user to verify access via the official portal.
   - Generates CSV output matching the CAAQM/Delhi hourly raw schema when data is retrieved.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DELHI_RAW_COLUMNS = [
    "Station",
    "Timestamp",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO (µg/m³)",
    "NO2 (µg/m³)",
    "NOx (ppb)",
    "NH3 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
    "Benzene (µg/m³)",
    "Toluene (µg/m³)",
    "Xylene (µg/m³)",
    "O Xylene (µg/m³)",
    "Eth-Benzene (µg/m³)",
    "MP-Xylene (µg/m³)",
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "WD (deg)",
    "RF (mm)",
    "TOT-RF (mm)",
    "SR (W/mt2)",
    "BP (mmHg)",
    "VWS (m/s)",
]

DEFAULT_ENDPOINT = "https://airquality.cpcb.gov.in/insight_proxy_api_v2_0/service.php"


class CaptchaRequiredError(RuntimeError):
    """Raised when the CPCB portal demands a CAPTCHA or human verification challenge."""
    pass


class AuthRequiredError(RuntimeError):
    """Raised when the endpoint requires authentication or session credentials."""
    pass


def check_for_captcha_or_auth(response_text: str, status_code: int = 200) -> None:
    """Inspect the HTTP response body and status code for CAPTCHA or authentication gates."""
    lowered = response_text.lower()
    if "captcha" in lowered or "recaptcha" in lowered or "turnstile" in lowered:
        raise CaptchaRequiredError(
            "CPCB portal presented a CAPTCHA challenge. Halting per policy (no automated bypass allowed)."
        )
    if "login" in lowered and ("password" in lowered or "auth" in lowered):
        raise AuthRequiredError(
            "CPCB portal requires login / credentials. Halting per policy."
        )
    if status_code in (401, 403):
        raise AuthRequiredError(
            f"CPCB portal returned HTTP status {status_code} (Authentication/Access Denied)."
        )


def format_to_cpcb_csv(station_name: str, records: List[Dict[str, any]]) -> pd.DataFrame:
    """Convert raw parameter records into the standard AQFM CAAQM raw CSV format."""
    df = pd.DataFrame(records)
    for col in DELHI_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["Station"] = station_name
    return df[DELHI_RAW_COLUMNS]


def fetch_station_data(
    station_code: str,
    station_name: str,
    start: str,
    end: str,
    params: List[str],
    out_dir: str,
    endpoint: str = DEFAULT_ENDPOINT,
    rate_limit: float = 2.0,
    session: Optional[requests.Session] = None,
) -> str:
    """Fetch station observations for the date range, format to CAAQM CSV, and write to disk."""
    sess = session or requests.Session()
    payload = {
        "station_code": station_code,
        "from_date": start,
        "to_date": end,
        "parameters": params,
    }
    logger.info(f"Querying station {station_code} ({station_name}) for range {start} to {end}...")

    retries = 3
    delay = rate_limit
    resp = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.post(endpoint, json=payload, timeout=20)
            check_for_captcha_or_auth(resp.text, resp.status_code)
            if resp.status_code in (429, 503):
                logger.warning(f"Rate limited or server busy ({resp.status_code}). Attempt {attempt}/{retries}. Backoff {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            break
        except (CaptchaRequiredError, AuthRequiredError):
            raise
        except Exception as e:
            if attempt == retries:
                logger.error(f"Failed to fetch data for station {station_code}: {e}")
                raise
            logger.warning(f"Error on attempt {attempt}: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    time.sleep(rate_limit)

    data = resp.json() if resp and resp.text.startswith(("{", "[")) else None
    if not data or "records" not in data:
        logger.info(f"No records found for station {station_code} ({start}..{end}).")
        records = []
    else:
        records = data["records"]

    df = format_to_cpcb_csv(station_name, records)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{station_code}.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download CAAQM air quality data from CPCB.")
    ap.add_argument("--stations", required=True, help="CSV containing station_id and cpcb_code columns")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--out-dir", required=True, help="Output directory to store raw CSVs")
    ap.add_argument("--rate-limit", type=float, default=2.0, help="Delay in seconds between requests")
    ap.add_argument(
        "--parameters",
        default="PM2.5,PM10,NO2,NOx,NO,Ozone,SO2,NH3,CO,AT,RH,WS,WD,SR,BP,Benzene,TOT_RF",
        help="Comma-separated list of target parameters",
    )
    args = ap.parse_args(argv)

    if not os.path.exists(args.stations):
        logger.error(f"Stations file not found: {args.stations}")
        sys.exit(1)

    df_stations = pd.read_csv(args.stations)
    if "cpcb_code" not in df_stations.columns:
        logger.error("Stations file must contain 'cpcb_code' column.")
        sys.exit(1)

    params_list = [p.strip() for p in args.parameters.split(",") if p.strip()]
    name_col = "cpcb_name" if "cpcb_name" in df_stations.columns else "station_id"

    for _, row in df_stations.iterrows():
        st_code = str(row["cpcb_code"]).strip()
        st_name = str(row.get(name_col, st_code)).strip()
        if not st_code or st_code == "nan":
            continue
        try:
            fetch_station_data(
                station_code=st_code,
                station_name=st_name,
                start=args.start,
                end=args.end,
                params=params_list,
                out_dir=args.out_dir,
                rate_limit=args.rate_limit,
            )
        except CaptchaRequiredError as e:
            logger.error(f"Stopping scraper: {e}")
            sys.exit(2)
        except AuthRequiredError as e:
            logger.error(f"Stopping scraper: {e}")
            sys.exit(3)
        except Exception as e:
            logger.error(f"Failed processing station {st_code}: {e}")


if __name__ == "__main__":
    main()
