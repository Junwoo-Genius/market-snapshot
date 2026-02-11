#!/usr/bin/env python3
# bot.py
# market-snapshot 자동화: report.json(1년 OHLCV+지표) + csv(5년 OHLCV)
# - 트리거: workflow_dispatch (수동/외부 호출)
# - 산출물:
#   1) public/report.json
#   2) public/csv/{TICKER}_daily.csv  (최근 5년)

import os, json, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# =========================
# 설정
# =========================
ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
CSV_DIR = PUBLIC_DIR / "csv"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = {
    "rsi": 14,
    "ema": 20,
    "sma": 60,
    "history_days_report": 252,   # report.json 저장용 (약 1년)
    "history_days_csv": 1260,     # csv 저장용 (약 5년)
}

# tickers.json (없으면 기본)
# 예: ["IREN","PLTR","NVDA"]
TICKERS_FILE = ROOT / "tickers.json"

# Stooq 심볼 변환: 미국 주식 기준 ".us"
def to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if t.endswith(".us"):
        return t
    return f"{t}.us"

def stooq_daily_url(ticker: str) -> str:
    # Stooq CSV 다운로드 (일봉)
    # 예: https://stooq.com/q/d/l/?s=iren.us&i=d
    return f"https://stooq.com/q/d/l/?s={to_stooq_symbol(ticker)}&i=d"


# =========================
# 기술지표
# =========================
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (EMA with alpha=1/period) 유사
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =========================
# 데이터 로드
# =========================
def fetch_stooq_daily(ticker: str, retries: int = 3, timeout: int = 20) -> pd.DataFrame:
    url = stooq_daily_url(ticker)
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "market-snapshot-bot/1.0"})
            r.raise_for_status()
            # Stooq: Date,Open,High,Low,Close,Volume
            df = pd.read_csv(pd.compat.StringIO(r.text)) if hasattr(pd, "compat") else pd.read_csv(pd.io.common.StringIO(r.text))
            if "Date" not in df.columns:
                raise ValueError("Stooq CSV format unexpected (no Date column).")
            df["Date"] = pd.to_datetime(df["Date"], utc=False)
            df = df.sort_values("Date").reset_index(drop=True)
            # 일부 종목은 Volume이 '-' 혹은 NaN일 수 있음
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"Failed to fetch {ticker} from Stooq after {retries} retries: {last_err}")


# =========================
# 저장
# =========================
def save_csv_5y(ticker: str, df: pd.DataFrame, history_days: int):
    out = df.tail(history_days).copy()
    out_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    out = out[out_cols]
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    path = CSV_DIR / f"{ticker.upper()}_daily.csv"
    out.to_csv(path, index=False)

def build_report_entry(ticker: str, df: pd.DataFrame, params: dict) -> dict:
    df1y = df.tail(params["history_days_report"]).copy()

    df1y["EMA20"] = df1y["Close"].ewm(span=params["ema"], adjust=False).mean()
    df1y["SMA60"] = df1y["Close"].rolling(params["sma"]).mean()
    df1y["RSI14"] = compute_rsi(df1y["Close"], params["rsi"])

    last = df1y.iloc[-1]
    last_date = last["Date"].strftime("%Y-%m-%d")

    # history_1y: OHLCV 전부 저장 (차트/매물대/지지저항 계산용)
    history_1y = []
    for _, row in df1y.iterrows():
        history_1y.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        })

    # daily snapshot (마지막 1일 값 + 지표)
    def safe_float(x):
        try:
            return float(x) if pd.notna(x) else None
        except Exception:
            return None

    daily = {
        "last_date": last_date,
        "last_close": float(last["Close"]),
        "last_volume": int(last["Volume"]),
        "rsi14": safe_float(last["RSI14"]),
        "ema20": safe_float(last["EMA20"]),
        "sma60": safe_float(last["SMA60"]),
    }

    return {
        "daily": daily,
        "history_1y": history_1y
    }


def main():
    if TICKERS_FILE.exists():
        tickers = json.loads(TICKERS_FILE.read_text(encoding="utf-8"))
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    else:
        tickers = ["IREN"]

    params = DEFAULT_PARAMS.copy()

    asof_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    report = {
        "asof_utc": asof_utc,
        "source": "stooq",
        "params": {
            "rsi": params["rsi"],
            "ema": params["ema"],
            "sma": params["sma"],
            "history_days_report": params["history_days_report"],
            "history_days_csv": params["history_days_csv"],
        },
        "tickers": {}
    }

    errors = {}

    for t in tickers:
        try:
            df = fetch_stooq_daily(t)

            # CSV(5년) 저장
            save_csv_5y(t, df, params["history_days_csv"])

            # report entry(1년) 생성
            report["tickers"][t] = build_report_entry(t, df, params)

        except Exception as e:
            errors[t] = str(e)

    # 에러도 함께 저장(디버깅용)
    if errors:
        report["errors"] = errors

    (PUBLIC_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

