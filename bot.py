import os
import json
import csv
import requests
from io import StringIO
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ============================================================
# FIXED PARAMETERS
# ============================================================
RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60

# Stooq CSV endpoint
STOOQ_BASE = "https://stooq.com/q/d/l/"

# Files
TICKERS_FILE = os.environ.get("TICKERS_FILE", "tickers.txt")
OUT_PATH = os.environ.get("OUT_PATH", "public/report.json")

# ============================================================
# SERIES EXPORT (for charting later in ChatGPT by paste-in)
# - report.json 안에 10년치(대략) OHLCV 시계열을 같이 저장
# ============================================================
EXPORT_SERIES = os.environ.get("EXPORT_SERIES", "1") == "1"
# 10년 영업일 대략 2520개 + 여유
SERIES_KEEP_LAST_DAILY = int(os.environ.get("SERIES_KEEP_LAST_DAILY", "2600"))
# 주봉/월봉은 대략치
SERIES_KEEP_LAST_WEEKLY = int(os.environ.get("SERIES_KEEP_LAST_WEEKLY", "600"))
SERIES_KEEP_LAST_MONTHLY = int(os.environ.get("SERIES_KEEP_LAST_MONTHLY", "150"))

# ============================================================
# HELPERS
# ============================================================

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


def ema_last(values, period):
    """EMA 마지막 값만 반환"""
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def rsi_wilder(values, period=14):
    """Wilder RSI (TradingView 스타일 smoothing) - 마지막 값만 반환"""
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
        "rsi14": rsi_wilder(close_series, RSI_PERIOD),
        "ema20": ema_last(close_series, EMA_PERIOD),
        "sma60": sma(close_series, SMA_PERIOD),
    }


def to_stooq_symbol(us_symbol: str) -> str:
    # 미국 주식은 .us
    return f"{us_symbol.lower()}.us"


def fetch_stooq_daily(us_symbol: str, keep_last: int = 6000):
    """
    Stooq CSV (daily):
      https://stooq.com/q/d/l/?s=aapl.us&i=d
    Columns: Date,Open,High,Low,Close,Volume
    """
    params = {"s": to_stooq_symbol(us_symbol), "i": "d"}
    r = requests.get(STOOQ_BASE, params=params, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text or "No data" in text:
        raise RuntimeError(f"{us_symbol}: no data from stooq")

    reader = csv.DictReader(StringIO(text))
    rows = [row for row in reader if row.get("Date")]

    if len(rows) < 10:
        raise RuntimeError(f"{us_symbol}: insufficient rows ({len(rows)})")

    rows.sort(key=lambda x: x["Date"])
    rows = rows[-keep_last:]

    dates, o, h, l, c, v = [], [], [], [], [], []
    for row in rows:
        dates.append(row["Date"])
        o.append(float(row["Open"]))
        h.append(float(row["High"]))
        l.append(float(row["Low"]))
        c.append(float(row["Close"]))
        # volume sometimes comes as float-like text
        v.append(int(float(row["Volume"])))

    return dates, o, h, l, c, v


def is_market_closed_plus_30m_et(now_utc: datetime) -> bool:
    """
    미국 정규장 마감(16:00 ET) + 30분 버퍼(16:30 ET) 이후면 True
    DST 자동 처리
    """
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    # 주말이면 닫힘 취급
    if et.weekday() >= 5:
        return True
    return (et.hour, et.minute) >= (16, 30)


def maybe_drop_today_if_not_closed(dates, o, h, l, c, v):
    """
    '오늘 데이터'가 끼어드는 상황(장중/애프터 혼입 가능성)을 최대한 방지:
    - 실행 시점이 16:30 ET 이전이면, 마지막 row의 Date가 '오늘(ET)'이면 제거
    """
    now_utc = datetime.now(timezone.utc)
    if is_market_closed_plus_30m_et(now_utc):
        return dates, o, h, l, c, v

    today_et = now_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    if dates and dates[-1] == today_et:
        return dates[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]
    return dates, o, h, l, c, v


def _iso_week_key(date_str: str):
    dt = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]))
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)


def resample_weekly(dates, o, h, l, c, v):
    """
    ISO 주차 기준으로 일봉 -> 주봉 리샘플링
    """
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
        out_dates.append(dates[li])  # 주 마지막 거래일
        out_o.append(o[fi])
        out_h.append(b["high"])
        out_l.append(b["low"])
        out_c.append(c[li])
        out_v.append(b["vol"])

    return {"dates": out_dates, "open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v}


def resample_monthly(dates, o, h, l, c, v):
    """
    연/월 기준으로 일봉 -> 월봉 리샘플링
    """
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
        out_dates.append(dates[li])  # 월 마지막 거래일
        out_o.append(o[fi])
        out_h.append(b["high"])
        out_l.append(b["low"])
        out_c.append(c[li])
        out_v.append(b["vol"])

    return {"dates": out_dates, "open": out_o, "high": out_h, "low": out_l, "close": out_c, "volume": out_v}


def pack_ohlcv(dates, o, h, l, c, v, keep_last: int):
    """
    JSON에 넣기 위한 OHLCV 시계열 패키징
    """
    if len(dates) > keep_last:
        dates = dates[-keep_last:]
        o = o[-keep_last:]
        h = h[-keep_last:]
        l = l[-keep_last:]
        c = c[-keep_last:]
        v = v[-keep_last:]

    return {
        "dates": dates,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    tickers = read_tickers()

    report = {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "stooq",
        "params": {"rsi": RSI_PERIOD, "ema": EMA_PERIOD, "sma": SMA_PERIOD},
        "tickers": {},
    }

    for sym in tickers:
        try:
            dates, o, h, l, c, v = fetch_stooq_daily(sym, keep_last=6000)
            # 장중/애프터 혼입 방지(가능한 범위)
            dates, o, h, l, c, v = maybe_drop_today_if_not_closed(dates, o, h, l, c, v)

            if len(dates) < 10:
                raise RuntimeError(f"{sym}: insufficient rows after filtering ({len(dates)})")

        except Exception as e:
            report["tickers"][sym] = {"error": str(e)}
            continue

        # 최근 5거래일 거래량
        last5_vol = v[-5:] if len(v) >= 5 else v[:]
        last5_dates = dates[-5:] if len(dates) >= 5 else dates[:]

        prev_vol = v[-2] if len(v) >= 2 else None
        vol_change_pct = None
        if prev_vol is not None and prev_vol != 0:
            vol_change_pct = (v[-1] - prev_vol) / prev_vol * 100.0

        # Indicators
        daily_ind = compute_indicators(c)

        w = resample_weekly(dates, o, h, l, c, v)
        m = resample_monthly(dates, o, h, l, c, v)

        weekly_ind = compute_indicators(w["close"]) if w["close"] else {"rsi14": None, "ema20": None, "sma60": None}
        monthly_ind = compute_indicators(m["close"]) if m["close"] else {"rsi14": None, "ema20": None, "sma60": None}

        payload = {
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
        }

        # ✅ 10년치(대략) OHLCV를 JSON에 같이 저장 (차트는 ChatGPT에서 paste-in으로 그릴 용도)
        if EXPORT_SERIES:
            payload["series"] = {
                "daily_10y": pack_ohlcv(dates, o, h, l, c, v, keep_last=SERIES_KEEP_LAST_DAILY),
                "weekly_10y": pack_ohlcv(
                    w["dates"], w["open"], w["high"], w["low"], w["close"], w["volume"],
                    keep_last=SERIES_KEEP_LAST_WEEKLY
                ),
                "monthly_10y": pack_ohlcv(
                    m["dates"], m["open"], m["high"], m["low"], m["close"], m["volume"],
                    keep_last=SERIES_KEEP_LAST_MONTHLY
                ),
            }

        report["tickers"][sym] = payload

    out_dir = os.path.dirname(OUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
