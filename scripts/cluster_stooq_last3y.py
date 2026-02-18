import os
import json
import time
import hashlib
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ====== Universe (32) ======
TICKERS = [
    "NVDA", "TSLA", "PLTR", "IREN", "BE", "ASTS", "CRCL", "HOOD", "OKLO", "NBIS", "ABAT", "AMD", "RGTI",
    "AAPL", "LLY", "SMR", "FLNC", "IONQ", "RIVN", "QBTS", "MU", "TSM", "INTC", "NVO", "RKLB", "ADBE",
    "NFLX", "GOOGL", "MSFT", "META", "UNH", "AVGO"
]

OUT_DIR = "public/cluster"
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ Single stable source: CSV already generated in your repo
RAW_BASE = "https://raw.githubusercontent.com/Junwoo-Genius/market-snapshot/main/public/csv"


def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fetch_daily_from_github_raw(ticker: str):
    """
    Source (fixed):
      https://raw.githubusercontent.com/Junwoo-Genius/market-snapshot/main/public/csv/{TICKER}_daily.csv

    Expect columns:
      Date,Open,High,Low,Close,Volume
    """
    url = f"{RAW_BASE}/{ticker}_daily.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; market-snapshot-bot/1.0)",
        "Accept": "text/csv,*/*",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    r = requests.get(url, headers=headers, timeout=30)
    status = r.status_code
    head200 = (r.text[:200] if r.text else "")

    # 실패 시에도 status/head200은 failed에 남길 수 있도록 예외 메시지에 포함
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(json.dumps({
            "ticker": ticker,
            "url": url,
            "http_status": status,
            "head200": head200,
            "error": str(e),
        }, ensure_ascii=False))

    b = r.content
    sha = hashlib.sha256(b).hexdigest()

    df = pd.read_csv(io.BytesIO(b))
    required = {"Date", "Close"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"{ticker}: missing columns {required - set(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    meta = {
        "ticker": ticker,
        "source_url": url,
        "download_time_utc": now_utc_iso(),
        "http_status": status,
        "bytes": len(b),
        "sha256": sha,
        "row_count": int(len(df)),
        "date_range": [
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d"),
        ] if len(df) else None,
    }
    return df, meta


def last_3y(df: pd.DataFrame):
    if df.empty:
        return df
    end = df["Date"].max()
    start = end - relativedelta(years=3)
    return df[df["Date"] >= start].copy().reset_index(drop=True)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def slope_pct(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return np.nan
    a = series.iloc[-n - 1]
    b = series.iloc[-1]
    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return (b / a - 1.0) * 100.0


def disparity_pct(x: float, ma: float) -> float:
    if pd.isna(x) or pd.isna(ma) or ma == 0:
        return np.nan
    return (x / ma - 1.0) * 100.0


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.set_index("Date").resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()
    return w


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = df.set_index("Date").resample("M").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()
    return m


def build_features(df_d: pd.DataFrame):
    """
    End = last3y 구간의 max Close(종가 최고점)
    End 시점 기준으로 D/W/M의 slope/disp/rsi 산출
    """
    d = df_d.copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA60"] = ema(d["Close"], 60)
    d["RSI14"] = rsi_wilder(d["Close"], 14)

    end_idx = d["Close"].idxmax()
    end = d.loc[end_idx]
    end_date = end["Date"]
    end_close = float(end["Close"])

    w = to_weekly(d)
    w["EMA20"] = ema(w["Close"], 20)
    w["EMA60"] = ema(w["Close"], 60)
    w["RSI14"] = rsi_wilder(w["Close"], 14)

    m = to_monthly(d)
    m["EMA20"] = ema(m["Close"], 20)
    m["RSI14"] = rsi_wilder(m["Close"], 14)

    w_end = w[w["Date"] >= end_date].head(1)
    if w_end.empty:
        w_end = w.tail(1)
    w_end_row = w_end.iloc[0]

    m_end = m[m["Date"] >= end_date].head(1)
    if m_end.empty:
        m_end = m.tail(1)
    m_end_row = m_end.iloc[0]

    feat = {}

    # ===== Slopes (fixed) =====
    # D: N=5, W: N=4, M: N=3/6 (as you previously locked in doc discussions)
    feat["D20_slope"] = slope_pct(d.loc[:end_idx, "EMA20"].tail(260), 5)
    feat["D60_slope"] = slope_pct(d.loc[:end_idx, "EMA60"].tail(400), 5)
    feat["D_diff"] = feat["D20_slope"] - feat["D60_slope"]

    feat["W20_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA20"].tail(120), 4)
    feat["W60_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA60"].tail(160), 4)

    feat["M3_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(60), 3)
    feat["M6_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(80), 6)

    # ===== Disparity
