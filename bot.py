import os
import json
import csv
from datetime import datetime, timezone
import yfinance as yf

# ===== Parameters (fixed) =====
RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60

TICKERS_FILE = os.environ.get("TICKERS_FILE", "tickers.txt")
OUT_PATH = os.environ.get("OUT_PATH", "public/report.json")

# CSV 저장 위치 (차트/시계열용)
CSV_DIR = os.environ.get("CSV_DIR", "public/csv")

# yfinance에서 가져올 최대 행 수
KEEP_LAST = int(os.environ.get("KEEP_LAST", "5000"))

def read_tickers(path=TICKERS_FILE):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s.upper())
    return out

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(values, period=14):
    # Wilder's RSI (TradingView-style smoothing)
    if len(values) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_indicators(close_series):
    return {
        "rsi14": rsi(close_series, RSI_PERIOD),
        "ema20": ema(close_series, EMA_PERIOD),
        "sma60": sma(close_series, SMA_PERIOD),
    }

def fetch_yfinance_daily(symbol: str, keep_last: int = KEEP_LAST):
    """
    yfinance 라이브러리를 사용하여 야후 파이낸스에서 일봉 OHLCV 데이터를 수집합니다.
    """
    ticker = yf.Ticker(symbol)
    
    # max 기간으로 호출하여 분할/배당 보정된 전체 데이터를 가져온 후 필요한 만큼 자릅니다.
    df = ticker.history(period="max")
    
    if df.empty:
        raise RuntimeError(f"{symbol}: no data from yfinance")
        
    df = df.tail(keep_last)
    
    if len(df) < 10:
        raise RuntimeError(f"{symbol}: insufficient rows ({len(df)})")

    dates, o, h, l, c, v = [], [], [], [], [], []
    for index, row in df.iterrows():
        # yfinance index는 Datetime이므로 문자열로 변환 (stooq과 동일한 포맷)
        dates.append(index.strftime("%Y-%m-%d"))
        o.append(float(row["Open"]))
        h.append(float(row["High"]))
        l.append(float(row["Low"]))
        c.append(float(row["Close"]))
        v.append(int(row["Volume"]))

    return dates, o, h, l, c, v

def write_daily_csv(sym, dates, o, h, l, c, v, out_dir=CSV_DIR):
    """
    종목별 시계열(일봉 OHLCV)을 CSV로 저장
    저장 경로: public/csv/{TICKER}_daily.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sym.upper()}_daily.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for i in range(len(dates)):
            w.writerow([dates[i], o[i], h[i], l[i], c[i], v[i]])
    return path

def _iso_week_key(date_str: str):
    dt = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]))
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)

def resample_weekly(dates, o, h, l, c, v):
    buckets = {}
    for i, d in enumerate(dates):
        key = _iso_week_key(d)
        if key not in buckets:
            buckets[key] = {"fi": i, "li": i, "high": h[i], "low": l[i], "vol": v[i]}
        else:
            b = buckets[key]
            b["li"] = i
            b["high"] = max(b["high"], h[i])
            b["low"] = min(b["low"], l[i])
            b["vol"] += v[i]

    out_dates, out_o, out_h, out_l, out_c, out_v = [], [], [], [], [], []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        fi, li = b["fi"], b["li"]
        out_dates.append(dates[li])  # last trading day of the week
        out_o.append(o[fi])
        out_h.append(b["high"])
        out_l.append(b["low"])
        out_c.append(c[li])
        out_v.append(b["vol"])

    return {"dates": out_dates, "open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v}

def resample_monthly(dates, o, h, l, c, v):
    buckets = {}
    for i, d in enumerate(dates):
        key = (int(d[0:4]), int(d[5:7]))
        if key not in buckets:
            buckets[key] = {"fi": i, "li": i, "high": h[i], "low": l[i], "vol": v[i]}
        else:
            b = buckets[key]
            b["li"] = i
            b["high"] = max(b["high"], h[i])
            b["low"] = min(b["low"], l[i])
            b["vol"] += v[i]

    out_dates, out_o, out_h, out_l, out_c, out_v = [], [], [], [], [], []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        fi, li = b["fi"], b["li"]
        out_dates.append(dates[li])  # last trading day of the month
        out_o.append(o[fi])
        out_h.append(b["high"])
        out_l.append(b["low"])
        out_c.append(c[li])
        out_v.append(b["vol"])

    return {"dates": out_dates, "open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v}

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    tickers = read_tickers()

    report = {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "yfinance",  # 소스 이름 업데이트됨
        "params": {"rsi": RSI_PERIOD, "ema": EMA_PERIOD, "sma": SMA_PERIOD},
        "tickers": {},
    }

    for sym in tickers:
        try:
            dates, o, h, l, c, v = fetch_yfinance_daily(sym, keep_last=KEEP_LAST)
        except Exception as e:
            report["tickers"][sym] = {"error": str(e)}
            continue

        try:
            csv_path = write_daily_csv(sym, dates, o, h, l, c, v, out_dir=CSV_DIR)
        except Exception as e:
            report["tickers"].setdefault(sym, {})
            report["tickers"][sym]["csv_error"] = str(e)
            csv_path = None

        last5_vol = v[-5:] if len(v) >= 5 else v[:]
        last5_dates = dates[-5:] if len(dates) >= 5 else dates[:]

        prev_vol = v[-2] if len(v) >= 2 else None
        vol_change_pct = None
        if prev_vol and prev_vol != 0:
            vol_change_pct = (v[-1] - prev_vol) / prev_vol * 100.0

        daily_ind = compute_indicators(c)

        w = resample_weekly(dates, o, h, l, c, v)
        m = resample_monthly(dates, o, h, l, c, v)

        weekly_ind = compute_indicators(w["close"]) if w["close"] else {"rsi14": None, "ema20": None, "sma60": None}
        monthly_ind = compute_indicators(m["close"]) if m["close"] else {"rsi14": None, "ema20": None, "sma60": None}

        report["tickers"][sym] = {
            "daily": {
                "last_date": dates[-1],
                "last_close": c[-1],
                "last_volume": v[-1],
                "volumes_dates_last5": last5_dates,
                "volumes_last5": last5_vol,
                "prev_volume": prev_vol,
                "vol_change_pct": vol_change_pct,
                **daily_ind,
            },
            "weekly": {
                "last_date": w["dates"][-1] if w["dates"] else None,
                "last_close": w["close"][-1] if w["close"] else None,
                "last_volume": w["volume"][-1] if w["volume"] else None,
                **weekly_ind,
            },
            "monthly": {
                "last_date": m["dates"][-1] if m["dates"] else None,
                "last_close": m["close"][-1] if m["close"] else None,
                "last_volume": m["volume"][-1] if m["volume"] else None,
                **monthly_ind,
            },
            "csv_path": csv_path,
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
