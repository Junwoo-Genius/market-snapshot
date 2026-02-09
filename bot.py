import os
import json
import csv
import requests
from io import StringIO
from datetime import datetime, timezone

RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60

STOOQ_BASE = "https://stooq.com/q/d/l/"

TICKERS_FILE = "tickers.txt"
OUT_PATH = "public/report.json"


def read_tickers():
    out = []
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
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
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def to_stooq_symbol(symbol):
    return f"{symbol.lower()}.us"


def fetch_stooq_daily(symbol, keep_last=3000):
    params = {"s": to_stooq_symbol(symbol), "i": "d"}
    r = requests.get(STOOQ_BASE, params=params, timeout=30)
    r.raise_for_status()

    reader = csv.DictReader(StringIO(r.text))
    rows = [row for row in reader if row.get("Date")]

    rows.sort(key=lambda x: x["Date"])
    rows = rows[-keep_last:]

    dates, close, volume = [], [], []
    for row in rows:
        dates.append(row["Date"])
        close.append(float(row["Close"]))
        volume.append(int(float(row["Volume"])))

    return dates, close, volume


def main():
    tickers = read_tickers()

    report = {
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "stooq",
        "tickers": {}
    }

    for sym in tickers:
        try:
            dates, close, volume = fetch_stooq_daily(sym)
        except Exception as e:
            report["tickers"][sym] = {"error": str(e)}
            continue

        report["tickers"][sym] = {
            "daily": {
                "last_date": dates[-1],
                "last_close": close[-1],
                "last_volume": volume[-1],
                "rsi14": rsi(close),
                "ema20": ema(close, EMA_PERIOD),
                "sma60": sma(close, SMA_PERIOD)
            }
        }

    os.makedirs("public", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
