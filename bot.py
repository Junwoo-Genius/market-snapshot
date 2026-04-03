import os
import json
import csv
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

RSI_PERIOD = 14
EMA_PERIOD = 20
SMA_PERIOD = 60
TICKERS_FILE = os.environ.get("TICKERS_FILE", "tickers.txt")
OUT_PATH = os.environ.get("OUT_PATH", "public/report.json")
CSV_DIR = os.environ.get("CSV_DIR", "public/csv")
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
    if len(values) < period: return None
    return sum(values[-period:]) / period

def ema(values, period):
    if len(values) < period: return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def rsi(values, period=14):
    if len(values) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_indicators(close_series):
    return {"rsi14": rsi(close_series, RSI_PERIOD), "ema20": ema(close_series, EMA_PERIOD), "sma60": sma(close_series, SMA_PERIOD)}

def fetch_yfinance_daily(symbol: str, keep_last: int = KEEP_LAST):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max")
    
    if df.empty: raise RuntimeError(f"{symbol}: no data from yfinance")
    
    # 🚨 [핵심 해결책: 서머타임 버그 원천 차단]
    if df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York').tz_localize(None)
    
    df.index = df.index.normalize()
    df = df.sort_index()
    
    df = df[~df.index.duplicated(keep='last')]
    df = df.ffill().bfill()
    df['Volume'] = df['Volume'].fillna(0)

    # 🚨 [신규 추가] 20일 평균 거래량 및 이격도(%) 연산
    df['AvgVol20'] = df['Volume'].rolling(window=20).mean()
    # 분모가 0이 되는 오류(ZeroDivisionError) 방지 처리
    df['VolDisp20'] = np.where(df['AvgVol20'] > 0, (df['Volume'] - df['AvgVol20']) / df['AvgVol20'] * 100, 0.0)

    # 지표 계산
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    df['SMA120'] = df['Close'].rolling(window=120).mean()
    df['SMA480'] = df['Close'].rolling(window=480).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    target_df = df.tail(keep_last).copy()

    def calc_poc_series(days, target_bins):
        poc_list = []
        for current_date in target_df.index:
            start_date = current_date - pd.Timedelta(days=days)
            window_df = df.loc[start_date:current_date].copy()
            if window_df.empty or len(window_df) < 3:
                poc_list.append("기간 부족")
                continue
            window_df = window_df[window_df['Volume'] > 0]
            if window_df.empty:
                poc_list.append("거래량 0")
                continue
            min_price = window_df['Close'].min()
            max_price = window_df['Close'].max()
            auto_bin = 0.1 if max_price <= min_price else max(0.1, round((max_price - min_price) / target_bins, 1))
            window_df['Price_Bin'] = np.floor(window_df['Close'] / auto_bin) * auto_bin
            poc_data = window_df.groupby('Price_Bin')['Volume'].sum()
            max_bin = poc_data.idxmax()
            max_vol = poc_data.max()
            total_vol = window_df['Volume'].sum()
            pct = max_vol / total_vol
            if pct >= 0.99: poc_list.append(f"{max_bin:.1f} - {max_bin + auto_bin:.1f} (오류)")
            else: poc_list.append(f"{max_bin:.1f} - {max_bin + auto_bin:.1f} ({pct:.1%})")
        return poc_list

    target_df['POC_20D'] = calc_poc_series(20, 10)
    target_df['POC_90D'] = calc_poc_series(90, 12)
    target_df['POC_365D'] = calc_poc_series(365, 13)
    target_df['POC_1095D'] = calc_poc_series(1095, 15)

    if len(target_df) < 10: raise RuntimeError(f"{symbol}: insufficient rows ({len(target_df)})")

    dates, o, h, l, c, v = [], [], [], [], [], []
    avg_vol20, vol_disp20 = [], []
    sma20, sma60, sma120, sma480, rsi14 = [], [], [], [], []
    poc20, poc90, poc365, poc1095 = [], [], [], []

    for index, row in target_df.iterrows():
        dates.append(index.strftime("%Y-%m-%d"))
        o.append(float(row["Open"]) if pd.notna(row["Open"]) else 0.0)
        h.append(float(row["High"]) if pd.notna(row["High"]) else 0.0)
        l.append(float(row["Low"]) if pd.notna(row["Low"]) else 0.0)
        c.append(float(row["Close"]) if pd.notna(row["Close"]) else 0.0)
        v.append(int(row["Volume"]) if pd.notna(row["Volume"]) else 0)
        
        # 🚨 추가된 거래량 지표 데이터 리스트에 삽입
        avg_vol20.append(int(row["AvgVol20"]) if pd.notna(row["AvgVol20"]) else 0)
        vol_disp20.append(round(row["VolDisp20"], 2) if pd.notna(row["VolDisp20"]) else 0.0)

        sma20.append(round(row["SMA20"], 2) if pd.notna(row["SMA20"]) else "")
        sma60.append(round(row["SMA60"], 2) if pd.notna(row["SMA60"]) else "")
        sma120.append(round(row["SMA120"], 2) if pd.notna(row["SMA120"]) else "")
        sma480.append(round(row["SMA480"], 2) if pd.notna(row["SMA480"]) else "")
        rsi14.append(round(row["RSI14"], 2) if pd.notna(row["RSI14"]) else "")
        poc20.append(row["POC_20D"])
        poc90.append(row["POC_90D"])
        poc365.append(row["POC_365D"])
        poc1095.append(row["POC_1095D"])
        
    return dates, o, h, l, c, v, avg_vol20, vol_disp20, sma20, sma60, sma120, sma480, rsi14, poc20, poc90, poc365, poc1095

def write_daily_csv(sym, dates, o, h, l, c, v, avg_vol20, vol_disp20, sma20, sma60, sma120, sma480, rsi14, poc20, poc90, poc365, poc1095, out_dir=CSV_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sym.upper()}_daily.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 🚨 CSV 헤더에 AvgVol20, VolDisp20 순서 맞게 추가
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "AvgVol20", "VolDisp20", "SMA20", "SMA60", "SMA120", "SMA480", "RSI14", "POC_20D_UltraShort", "POC_90D_Short", "POC_365D_Mid", "POC_1095D_Long"])
        for i in range(len(dates)):
            w.writerow([dates[i], o[i], h[i], l[i], c[i], v[i], avg_vol20[i], vol_disp20[i], sma20[i], sma60[i], sma120[i], sma480[i], rsi14[i], poc20[i], poc90[i], poc365[i], poc1095[i]])
    return path

def _iso_week_key(date_str: str):
    dt = datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10]))
    iso = dt.isocalendar()
    return int(iso.year), int(iso.week)

def resample_weekly(dates, o, h, l, c, v):
    buckets = {}
    for i, d in enumerate(dates):
        key = _iso_week_key(d)
        if key not in buckets: buckets[key] = {"fi": i, "li": i, "high": h[i], "low": l[i], "vol": v[i]}
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
        out_dates.append(dates[li])
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
        if key not in buckets: buckets[key] = {"fi": i, "li": i, "high": h[i], "low": l[i], "vol": v[i]}
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
        out_dates.append(dates[li])
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
        "source": "yfinance",
        "params": {"rsi": RSI_PERIOD, "ema": EMA_PERIOD, "sma": SMA_PERIOD},
        "tickers": {},
    }

    for sym in tickers:
        print(f"\n▶ [{sym}] 데이터 수집 및 연산 시작...", flush=True) # 🚨 진행상황 출력되도록 flush 복구
        try:
            dates, o, h, l, c, v, avg_vol20, vol_disp20, sma20, sma60, sma120, sma480, rsi14, poc20, poc90, poc365, poc1095 = fetch_yfinance_daily(sym, keep_last=KEEP_LAST)
        except Exception as e:
            print(f"❌ [{sym}] 치명적 에러 발생 (fetch_yfinance_daily):", flush=True)
            traceback.print_exc(file=sys.stdout)
            report["tickers"][sym] = {"error": str(e)}
            continue

        try:
            csv_path = write_daily_csv(sym, dates, o, h, l, c, v, avg_vol20, vol_disp20, sma20, sma60, sma120, sma480, rsi14, poc20, poc90, poc365, poc1095, out_dir=CSV_DIR)
            print(f"✅ [{sym}] CSV 생성 완료: {csv_path}", flush=True)
        except Exception as e:
            print(f"❌ [{sym}] CSV 저장 에러 발생:", flush=True)
            traceback.print_exc(file=sys.stdout)
            report["tickers"].setdefault(sym, {})
            report["tickers"][sym]["csv_error"] = str(e)
            csv_path = None

        last5_vol = v[-5:] if len(v) >= 5 else v[:]
        last5_dates = dates[-5:] if len(dates) >= 5 else dates[:]
        prev_vol = v[-2] if len(v) >= 2 else None
        vol_change_pct = None
        if prev_vol and prev_vol != 0: vol_change_pct = (v[-1] - prev_vol) / prev_vol * 100.0

        daily_ind = compute_indicators(c)
        w = resample_weekly(dates, o, h, l, c, v)
        m = resample_monthly(dates, o, h, l, c, v)

        weekly_ind = compute_indicators(w["close"]) if w["close"] else {"rsi14": None, "ema20": None, "sma60": None}
        monthly_ind = compute_indicators(m["close"]) if m["close"] else {"rsi14": None, "ema20": None, "sma60": None}

        report["tickers"][sym] = {
            "daily": {"last_date": dates[-1], "last_close": c[-1], "last_volume": v[-1], "volumes_dates_last5": last5_dates, "volumes_last5": last5_vol, "prev_volume": prev_vol, "vol_change_pct": vol_change_pct, **daily_ind},
            "weekly": {"last_date": w["dates"][-1] if w["dates"] else None, "last_close": w["close"][-1] if w["close"] else None, "last_volume": w["volume"][-1] if w["volume"] else None, **weekly_ind},
            "monthly": {"last_date": m["dates"][-1] if m["dates"] else None, "last_close": m["close"][-1] if m["close"] else None, "last_volume": m["volume"][-1] if m["volume"] else None, **monthly_ind},
            "csv_path": csv_path,
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n✅ 모든 프로세스 완료.")

if __name__ == "__main__":
    main()
