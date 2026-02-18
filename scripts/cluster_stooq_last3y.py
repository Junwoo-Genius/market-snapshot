import os
import json
import time
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# UNIVERSE (FIXED)
# =========================
TICKERS = [
    "NVDA", "TSLA", "PLTR", "IREN", "BE", "ASTS", "CRCL", "HOOD", "OKLO", "NBIS", "ABAT", "AMD", "RGTI",
    "AAPL", "LLY", "SMR", "FLNC", "IONQ", "RIVN", "QBTS", "MU", "TSM", "INTC", "NVO", "RKLB", "ADBE",
    "NFLX", "GOOGL", "MSFT", "META", "UNH", "AVGO"
]

# =========================
# PATH RESOLUTION
# =========================
def find_repo_root() -> str:
    gw = os.environ.get("GITHUB_WORKSPACE")
    if gw and os.path.isdir(gw):
        return os.path.abspath(gw)

    cur = os.path.abspath(os.getcwd())
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "public", "csv")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return os.path.abspath(os.getcwd())

ROOT = find_repo_root()
OUT_DIR = os.path.join(ROOT, "public", "cluster")
CSV_DIR = os.path.join(ROOT, "public", "csv")

def csv_path(ticker: str) -> str:
    return os.path.join(CSV_DIR, f"{ticker}_daily.csv")

# =========================
# UTILS
# =========================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def hard_print_env_and_fs() -> None:
    print("=== CLUSTER DEBUG START ===")
    print("UTC:", now_utc_iso())
    print("CWD:", os.path.abspath(os.getcwd()))
    print("REPO_ROOT:", ROOT)
    print("CSV_DIR:", CSV_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("GITHUB_WORKSPACE:", os.environ.get("GITHUB_WORKSPACE"))
    print("GITHUB_REF:", os.environ.get("GITHUB_REF"))
    print("GITHUB_SHA:", os.environ.get("GITHUB_SHA"))

    print("\n--- DIR EXISTS CHECK ---")
    print("exists(public):", os.path.isdir(os.path.join(ROOT, "public")))
    print("exists(public/csv):", os.path.isdir(CSV_DIR))
    print("exists(public/cluster):", os.path.isdir(os.path.join(ROOT, "public", "cluster")))

    print("\n--- CSV LIST (top 30) ---")
    if os.path.isdir(CSV_DIR):
        files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith("_daily.csv")])
        print("csv_count:", len(files))
        print("sample:", files[:30])
    else:
        print("csv_count: 0 (CSV_DIR missing)")

    print("\n--- SANITY OPEN TEST (AAPL, BE) ---")
    for t in ["AAPL", "BE"]:
        p = csv_path(t)
        print(f"try_open {t} -> {p} exists={os.path.exists(p)}")
        if os.path.exists(p):
            try:
                dfh = pd.read_csv(p, nrows=5)
                print(f"{t} head columns:", list(dfh.columns))
            except Exception as e:
                print(f"{t} read_csv ERROR:", repr(e))

    print("=== CLUSTER DEBUG END ===\n")

# =========================
# DATA LOADER (LOCAL ONLY)
# =========================
def read_local_daily(ticker: str):
    path = csv_path(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{ticker}: missing local csv -> {path}")

    df = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{ticker}: local csv missing columns {sorted(list(missing))} -> {path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if len(df) < 60:
        raise RuntimeError(f"{ticker}: too few rows after parse ({len(df)}) -> {path}")

    meta = {
        "ticker": ticker,
        "source_path": path,
        "sha256": sha256_file(path),
        "row_count": int(len(df)),
        "date_range": [
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d"),
        ],
    }
    return df, meta

# =========================
# INDICATORS / FEATURES
# =========================
def last_3y(df: pd.DataFrame) -> pd.DataFrame:
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
    w = df.set_index("Date").resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna().reset_index()
    return w

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    # ✅ FIX: pandas 환경에서 'M'이 막혀서 'ME'(month-end) 사용
    m = df.set_index("Date").resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna().reset_index()
    return m

def build_features(df_d: pd.DataFrame) -> dict:
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

    # slopes (fixed): D N=5, W N=4, M N=3/6
    feat["D20_slope"] = slope_pct(d.loc[:end_idx, "EMA20"].tail(260), 5)
    feat["D60_slope"] = slope_pct(d.loc[:end_idx, "EMA60"].tail(400), 5)
    feat["D_diff"] = feat["D20_slope"] - feat["D60_slope"]

    feat["W20_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA20"].tail(120), 4)
    feat["W60_slope"] = slope_pct(w[w["Date"] <= w_end_row["Date"]]["EMA60"].tail(160), 4)

    feat["M3_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(60), 3)
    feat["M6_slope"] = slope_pct(m[m["Date"] <= m_end_row["Date"]]["EMA20"].tail(80), 6)

    # disparities/spreads (End-based)
    feat["d_DISP20"] = disparity_pct(end_close, float(end["EMA20"]))
    feat["d_DISP60"] = disparity_pct(end_close, float(end["EMA60"]))
    feat["D_SPREAD"] = disparity_pct(float(end["EMA20"]), float(end["EMA60"]))

    feat["w_DISP20"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA20"]))
    feat["w_DISP60"] = disparity_pct(float(w_end_row["Close"]), float(w_end_row["EMA60"]))
    feat["w_SPREAD"] = disparity_pct(float(w_end_row["EMA20"]), float(w_end_row["EMA60"]))

    feat["M_DISP20"] = disparity_pct(float(m_end_row["Close"]), float(m_end_row["EMA20"]))

    # RSI
    feat["D_RSI_END"] = float(end["RSI14"]) if not pd.isna(end["RSI14"]) else np.nan
    feat["W_RSI_END"] = float(w_end_row["RSI14"]) if not pd.isna(w_end_row["RSI14"]) else np.nan
    feat["M_RSI_END"] = float(m_end_row["RSI14"]) if not pd.isna(m_end_row["RSI14"]) else np.nan

    feat["_END_DATE"] = end_date.strftime("%Y-%m-%d")
    feat["_END_CLOSE"] = float(end_close)
    return feat

def percentile_dict(arr: np.ndarray) -> dict:
    return {
        "P10": float(np.nanpercentile(arr, 10)),
        "P25": float(np.nanpercentile(arr, 25)),
        "Med": float(np.nanpercentile(arr, 50)),
        "P75": float(np.nanpercentile(arr, 75)),
        "P90": float(np.nanpercentile(arr, 90)),
    }

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hard_print_env_and_fs()

    rows = []
    failed = []

    for t in TICKERS:
        try:
            df, _meta = read_local_daily(t)
            df3 = last_3y(df)
            if len(df3) < 60:
                raise RuntimeError(f"{t}: last_3y too short ({len(df3)})")

            f = build_features(df3)
            f["Ticker"] = t
            rows.append(f)

        except Exception as e:
            failed.append({"ticker": t, "path": csv_path(t), "error": repr(e)})

        time.sleep(0.005)

    print(f"\nSUMMARY: ok={len(rows)} failed={len(failed)} universe={len(TICKERS)}")

    if len(rows) == 0:
        print("\n=== FAILED SAMPLE (top 10) ===")
        for x in failed[:10]:
            print(x["ticker"], "->", x["error"])
        print("=== FAILED SAMPLE END ===\n")
        raise SystemExit("All tickers failed.")

    feat_df = pd.DataFrame(rows).set_index("Ticker")

    feature_cols = [c for c in feat_df.columns if not c.startswith("_")]
    X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) < 5:
        dropped = sorted(set(feat_df.index) - set(X.index))
        print("\n=== INVALID FEATURE ROWS (top 10) ===")
        print("valid:", len(X), "dropped:", len(dropped))
        print("dropped sample:", dropped[:10])
        print("=== INVALID FEATURE ROWS END ===\n")
        raise SystemExit(f"Not enough valid rows for K=5 clustering. valid={len(X)}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=50)
    labels = kmeans.fit_predict(Xs)

    out = X.copy()
    out["Cluster"] = labels

    bands = {}
    for cl in sorted(out["Cluster"].unique()):
        sub = out[out["Cluster"] == cl]
        b = {col: percentile_dict(sub[col].to_numpy()) for col in feature_cols}
        bands[str(int(cl))] = {"count": int(len(sub)), "bands": b}

    assign = out[["Cluster"]].copy()
    assign = assign.join(feat_df[["_END_DATE", "_END_CLOSE"]], how="left")

    assign.to_csv(os.path.join(OUT_DIR, "cluster_assign.csv"), encoding="utf-8")
    with open(os.path.join(OUT_DIR, "cluster_bands.json"), "w", encoding="utf-8") as f:
        json.dump(bands, f, ensure_ascii=False, indent=2)

    print("\nDONE: wrote public/cluster/cluster_assign.csv and cluster_bands.json")

if __name__ == "__main__":
    main()
