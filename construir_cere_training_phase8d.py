"""
CERE FASE 8D — CERE vs LOW-VOL FACTOR

Objetivo:
    Congelar la tesis descubierta en Fases 8A/8B y tratar de REFUTARLA,
    no volver a optimizarla.

Tesis congelada principal:
    - Señal: VOL252
    - Selección principal: threshold descubierto en Discovery H20
    - Sizing principal: VOL_ADJ
    - Máximo por posición: 20%
    - Universo: 335 tickers congelados

Pruebas:
    1) Costos: 0/20/50/100/150 bps.
    2) Tamaño de cartera: TOP5/TOP10/TOP15/TOP20 y threshold CERE.
    3) Entrada retrasada: Open(t+1), Open(t+2), Open(t+3).
    4) Holding: 15/20/25 sesiones.
    5) Perturbación de fechas OOS: -2/-1/0/+1/+2 sesiones.
    6) Benchmarks: equal-weight universo y raw VOL252 TOP-N.

IMPORTANTE:
    - Discovery se utiliza exclusivamente para congelar dirección y threshold.
    - No hay refit durante OOS.
    - Los forwards se construyen desde Open de entrada hasta Close de salida.
    - No se usa información futura para seleccionar ni ponderar.
    - Esta fase NO escribe Google Sheets ni toca producción.
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

TICKERS = [
    "AA",
    "AAPL",
    "ABNB",
    "ABT",
    "ADI",
    "AFRM",
    "ALAB",
    "ALGN",
    "AMAT",
    "AMD",
    "AMGN",
    "AMT",
    "AMTD",
    "AMZN",
    "ANET",
    "APA",
    "APD",
    "APO",
    "APP",
    "ARKF",
    "ARKG",
    "ARKK",
    "ARKQ",
    "ARKW",
    "ASB",
    "ASML",
    "ASTS",
    "ATI",
    "AVB",
    "AVGO",
    "AX",
    "AXON",
    "AXP",
    "AZN",
    "BA",
    "BABA",
    "BAC",
    "BE",
    "BEAM",
    "BEN",
    "BIDU",
    "BIIB",
    "BITO",
    "BKNG",
    "BKR",
    "BLK",
    "BLNK",
    "BMY",
    "BNTX",
    "BOKF",
    "BSX",
    "BX",
    "BXP",
    "C",
    "CACC",
    "CAT",
    "CATY",
    "CAVA",
    "CCI",
    "CDNS",
    "CEG",
    "CELH",
    "CF",
    "CFG",
    "CHPT",
    "CHTR",
    "CHWY",
    "CIFR",
    "CL",
    "CLF",
    "CLS",
    "CMCSA",
    "COIN",
    "COMP",
    "COP",
    "COST",
    "CRDO",
    "CRSP",
    "CRWD",
    "CSCO",
    "CSX",
    "CUBI",
    "CVNA",
    "CVX",
    "CX",
    "DD",
    "DDOG",
    "DE",
    "DELL",
    "DHI",
    "DIS",
    "DJT",
    "DKNG",
    "DLR",
    "DOCU",
    "DOW",
    "DUOL",
    "DVN",
    "EA",
    "EBAY",
    "ECL",
    "EDIT",
    "EOG",
    "EQIX",
    "EQR",
    "ETN",
    "ETSY",
    "EW",
    "EXPE",
    "EXR",
    "FCEL",
    "FCF",
    "FCX",
    "FDX",
    "FITB",
    "FLEX",
    "FLYW",
    "FMC",
    "FNB",
    "FOXA",
    "FSLR",
    "GD",
    "GE",
    "GEV",
    "GILD",
    "GIS",
    "GLD",
    "GLW",
    "GOOGL",
    "GPRO",
    "GS",
    "GSK",
    "HAL",
    "HAPN",
    "HBAN",
    "HD",
    "HOMB",
    "HON",
    "HOOD",
    "HUT",
    "IEFA",
    "ILMN",
    "INTC",
    "IREN",
    "ISRG",
    "IWM",
    "JD",
    "JETS",
    "JMIA",
    "JNJ",
    "JPM",
    "KEY",
    "KKR",
    "KLAC",
    "KMB",
    "KO",
    "KRE",
    "LCID",
    "LEN",
    "LI",
    "LIN",
    "LITE",
    "LLY",
    "LMT",
    "LOT",
    "LOW",
    "LPRO",
    "LRCX",
    "LYV",
    "MA",
    "MAA",
    "MARA",
    "MCHP",
    "MDB",
    "MDLZ",
    "MDT",
    "MELI",
    "META",
    "MLM",
    "MMM",
    "MNST",
    "MOD",
    "MOS",
    "MP",
    "MRNA",
    "MRVL",
    "MS",
    "MSFT",
    "MTB",
    "MTCH",
    "MU",
    "NEM",
    "NET",
    "NFLX",
    "NIO",
    "NOC",
    "NOW",
    "NSC",
    "NTLA",
    "NTR",
    "NTRS",
    "NU",
    "NUE",
    "NVDA",
    "NVO",
    "NWSA",
    "NXPI",
    "O",
    "OIH",
    "OKTA",
    "OMF",
    "ON",
    "OPEN",
    "OPFI",
    "OXY",
    "OZK",
    "PAGS",
    "PANW",
    "PARA",
    "PAYS",
    "PDD",
    "PEP",
    "PFE",
    "PG",
    "PHM",
    "PINS",
    "PLD",
    "PLTR",
    "PLUG",
    "PSA",
    "PSNY",
    "PWR",
    "PYPL",
    "QCOM",
    "QQQ",
    "QS",
    "RDDT",
    "REGN",
    "RF",
    "RIOT",
    "RIVN",
    "RKLB",
    "ROKU",
    "RSG",
    "RTX",
    "RUM",
    "RUN",
    "SBAC",
    "SBUX",
    "SCHW",
    "SE",
    "SEDG",
    "SHOP",
    "SHW",
    "SIFY",
    "SLB",
    "SMCI",
    "SMH",
    "SNAP",
    "SNOW",
    "SNPS",
    "SNY",
    "SOFI",
    "SOXX",
    "SPG",
    "SPWR",
    "STLD",
    "STNE",
    "STT",
    "SYK",
    "T",
    "TAK",
    "TEAM",
    "TELFY",
    "TEM",
    "TER",
    "TFC",
    "TGT",
    "TMDX",
    "TMUS",
    "TRIP",
    "TROW",
    "TSLA",
    "TSM",
    "TTWO",
    "TWLO",
    "TXN",
    "UMBF",
    "UNG",
    "UNP",
    "UPS",
    "UPST",
    "USO",
    "V",
    "VEA",
    "VICI",
    "VLY",
    "VMC",
    "VNQ",
    "VOD",
    "VRT",
    "VRTX",
    "VST",
    "VWO",
    "VZ",
    "W",
    "WAY",
    "WBD",
    "WBS",
    "WDAY",
    "WFC",
    "WM",
    "WMT",
    "WULF",
    "WY",
    "XBI",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "XOM",
    "XPEV",
    "XYZ",
    "Z",
    "ZBH",
    "ZG",
    "ZION",
    "ZM",
    "ZS",
]

PERIOD = "10y"
MIN_STATE = 260
DISCOVERY_DATES = 12
EMBARGO_DAYS = 5
BASE_HORIZON = 20
HOLDING_HORIZONS = (15, 20, 25)
ENTRY_LAGS = (1, 2, 3)
DATE_OFFSETS = (-2, -1, 0, 1, 2)
COST_BPS = (0, 20, 50, 100, 150)
TOP_NS = (5, 10, 15, 20)
MAX_POSITION = 0.20
THRESHOLD_Q = 0.90

OUTDIR = Path("cere_phase8d_lowvol")
OUTDIR.mkdir(exist_ok=True)

STATE_COLS = [
    "R1", "R5", "R20", "R60", "R120",
    "Vol5", "Vol20", "Vol60", "Vol252",
    "VolRatio20", "VolRatio60", "Skew20", "Kurt20", "AC1"
]

def canonical_state(close: np.ndarray, i: int):
    if i < 252 or i < MIN_STATE:
        return None

    c = close[: i + 1]
    lr = np.diff(np.log(c))
    if len(lr) < 252:
        return None

    def ret(n):
        return c[-1] / c[-1 - n] - 1.0

    def vol(n):
        x = lr[-n:]
        if len(x) < n or np.std(x, ddof=1) <= 0:
            return np.nan
        return np.std(x, ddof=1) * math.sqrt(252)

    v5, v20, v60, v252 = vol(5), vol(20), vol(60), vol(252)
    x20 = lr[-20:]
    x60 = lr[-60:]

    if any(not np.isfinite(x) for x in [v5, v20, v60, v252]):
        return None

    volratio20 = v20 / v252 if v252 > 0 else np.nan
    volratio60 = v60 / v252 if v252 > 0 else np.nan

    skew20 = pd.Series(x20).skew()
    kurt20 = pd.Series(x20).kurt()  # excess kurtosis

    ac1 = np.nan
    if len(x60) >= 3 and np.std(x60[:-1]) > 0 and np.std(x60[1:]) > 0:
        ac1 = np.corrcoef(x60[:-1], x60[1:])[0, 1]

    vals = [
        ret(1), ret(5), ret(20), ret(60), ret(120),
        v5, v20, v60, v252, volratio20, volratio60,
        skew20, kurt20, ac1
    ]
    if any(not np.isfinite(x) for x in vals):
        return None
    return dict(zip(STATE_COLS, vals))


def build_ticker(ticker: str):
    try:
        df = yf.download(
            ticker,
            period=PERIOD,
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Open" not in df.columns or "Close" not in df.columns:
            return None

        df = df[["Open", "Close"]].dropna().copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        close = df["Close"].to_numpy(dtype=float)
        opn = df["Open"].to_numpy(dtype=float)

        rows = []
        for i in range(MIN_STATE, len(df) - 40):
            st = canonical_state(close, i)
            if st is None:
                continue

            row = {
                "Ticker": ticker,
                "Date": df.index[i],
                "Price": float(close[i]),
                **st,
            }

            for h in HORIZONS:
                if i + h >= len(df) or i + 1 >= len(df):
                    row[f"Forward{h}"] = np.nan
                else:
                    row[f"Forward{h}"] = float(close[i + h] / opn[i + 1] - 1.0)

            rows.append(row)

        if not rows:
            return None

        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[WARN] {ticker}: {e}")
        return None


def signal_value(df: pd.DataFrame, signal: str):
    if signal == "VOL60":
        return df["Vol60"].to_numpy(float)
    if signal == "VOL252":
        return df["Vol252"].to_numpy(float)
    if signal == "VOL60_VOL252":
        # Equal-component standardized composite. The standardization is
        # performed within the historical Discovery sample only when the
        # frozen parameter is learned.
        return None
    raise ValueError(signal)


def choose_dates(all_dates, horizon):
    dates = sorted(pd.Timestamp(x) for x in all_dates)
    min_gap = horizon + EMBARGO_DAYS
    selected=[]
    for d in dates:
        if not selected:
            selected.append(d); continue
        gap=sum(x > selected[-1] and x <= d for x in dates)
        if gap >= min_gap:
            selected.append(d)
    return selected


def freeze_signal_params(discovery: pd.DataFrame, signal: str, horizon: int):
    if signal in ("VOL60", "VOL252"):
        x = discovery[signal.replace("VOL", "Vol")].astype(float)
        # Discovery direction is frozen from correlation with future return.
        rho = spearmanr(x, discovery[f"Forward{horizon}"], nan_policy="omit").statistic
        direction = 1 if (np.isfinite(rho) and rho >= 0) else -1
        threshold = np.nanpercentile(direction * x, THRESHOLD_Q)
        return {"direction": direction, "threshold": float(threshold)}

    # Composite: z-score each component using Discovery only.
    a = discovery["Vol60"].astype(float)
    b = discovery["Vol252"].astype(float)
    za = (a - a.mean()) / a.std(ddof=1)
    zb = (b - b.mean()) / b.std(ddof=1)
    raw = za + zb
    rho = spearmanr(raw, discovery[f"Forward{horizon}"], nan_policy="omit").statistic
    direction = 1 if (np.isfinite(rho) and rho >= 0) else -1
    score = direction * raw
    threshold = np.nanpercentile(score, THRESHOLD_Q)
    return {
        "direction": direction,
        "threshold": float(threshold),
        "mean60": float(a.mean()),
        "std60": float(a.std(ddof=1)),
        "mean252": float(b.mean()),
        "std252": float(b.std(ddof=1)),
    }


def apply_frozen_signal(df: pd.DataFrame, signal: str, params: dict):
    if signal == "VOL60":
        return params["direction"] * df["Vol60"].to_numpy(float)
    if signal == "VOL252":
        return params["direction"] * df["Vol252"].to_numpy(float)

    z60 = (df["Vol60"] - params["mean60"]) / params["std60"]
    z252 = (df["Vol252"] - params["mean252"]) / params["std252"]
    return params["direction"] * (z60 + z252)


def select_names(day: pd.DataFrame, policy: str, score_col: str, n_universe: int):
    x = day.dropna(subset=[score_col]).sort_values(score_col, ascending=False).copy()
    if x.empty:
        return x

    if policy == "TOP5":
        return x.head(min(5, len(x)))
    if policy == "TOP10":
        return x.head(min(10, len(x)))
    if policy == "TOP15":
        return x.head(min(15, len(x)))
    if policy == "TOP20":
        return x.head(min(20, len(x)))

    # Threshold perturbation: the underlying Discovery threshold remains
    # frozen; only a fixed multiplicative stress around it is tested.
    if policy in THRESHOLD_MULTIPLIERS:
        mult = THRESHOLD_MULTIPLIERS[policy]
        return x[x[score_col] >= day.attrs["threshold"] * mult]

    raise ValueError(f"Política desconocida: {policy}")


def cap_and_normalize(weights: pd.Series, max_position=MAX_POSITION):
    if weights.empty:
        return weights

    w = weights.clip(lower=0).astype(float)
    if w.sum() <= 0:
        return pd.Series(dtype=float)

    w = w / w.sum()

    # Iterative cap-and-redistribute.
    active = pd.Series(True, index=w.index)
    out = pd.Series(0.0, index=w.index)
    remaining = 1.0

    while active.any():
        base = w[active]
        if base.sum() <= 0:
            break
        proposed = remaining * base / base.sum()
        over = proposed > max_position + 1e-12
        if not over.any():
            out.loc[active] = proposed
            break
        capped = proposed[over]
        out.loc[capped.index] = max_position
        active.loc[capped.index] = False
        remaining = 1.0 - out.sum()
        if remaining <= 1e-12:
            break

    return out[out > 0]


def build_ticker(ticker: str):
    try:
        df = yf.download(
            ticker, period=PERIOD, auto_adjust=True,
            progress=False, actions=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Open" not in df.columns or "Close" not in df.columns:
            return None
        df = df[["Open", "Close"]].dropna().copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        close = df["Close"].to_numpy(dtype=float)
        opn = df["Open"].to_numpy(dtype=float)
        max_offset = max(ENTRY_LAGS) + max(HOLDING_HORIZONS) - 1
        rows = []
        for i in range(MIN_STATE, len(df) - max_offset):
            st = canonical_state(close, i)
            if st is None:
                continue
            row = {"Ticker": ticker, "Date": df.index[i], "Price": float(close[i]), **st}
            for h in HOLDING_HORIZONS:
                for lag in ENTRY_LAGS:
                    exit_i = i + lag + h - 1
                    if exit_i < len(df) and i + lag < len(df):
                        row[f"Forward_H{h}_L{lag}"] = float(close[exit_i] / opn[i + lag] - 1.0)
                    else:
                        row[f"Forward_H{h}_L{lag}"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"[WARN] {ticker}: {e}")
        return None

def choose_dates(all_dates, min_gap):
    dates = sorted(pd.Timestamp(x) for x in all_dates)
    selected = []
    for d in dates:
        if not selected:
            selected.append(d)
            continue
        if sum(x > selected[-1] and x <= d for x in dates) >= min_gap:
            selected.append(d)
    return selected


def freeze_threshold(discovery):
    x = discovery["Vol252"].astype(float)
    rho = spearmanr(x, discovery["Forward_H20_L1"], nan_policy="omit").statistic
    direction = 1 if (np.isfinite(rho) and rho >= 0) else -1
    score = direction * x
    threshold = float(np.nanpercentile(score, THRESHOLD_Q))
    return {"direction": direction, "threshold": threshold, "rho": float(rho) if np.isfinite(rho) else np.nan}


def score_signal(df, params):
    return params["direction"] * df["Vol252"].to_numpy(float)


def cap_and_normalize(weights, max_position=MAX_POSITION):
    if weights is None or len(weights) == 0:
        return pd.Series(dtype=float)
    w = pd.Series(weights, dtype=float).clip(lower=0)
    w = w.replace([np.inf, -np.inf], np.nan).dropna()
    if w.empty or w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    active = pd.Series(True, index=w.index)
    out = pd.Series(0.0, index=w.index)
    remaining = 1.0
    while active.any():
        base = w[active]
        if base.sum() <= 0:
            break
        proposed = remaining * base / base.sum()
        over = proposed > max_position + 1e-12
        if not over.any():
            out.loc[active] = proposed
            break
        out.loc[proposed[over].index] = max_position
        active.loc[proposed[over].index] = False
        remaining = 1.0 - out.sum()
        if remaining <= 1e-12:
            break
    return out[out > 0]


def make_selection(day, kind, threshold, direction, top_n=None):
    x = day.dropna(subset=["Vol252", "Vol60"]).copy()
    if x.empty:
        return x
    raw = direction * x["Vol252"].astype(float)
    if kind == "CERE_THRESHOLD":
        return x[raw >= threshold].sort_values("Vol252", ascending=False)
    # Raw benchmark deliberately ignores the frozen CERE direction and threshold.
    x["RawVol252"] = x["Vol252"].astype(float)
    return x.sort_values("RawVol252", ascending=False).head(min(top_n, len(x)))


def weights_vol_adj(selected):
    if selected.empty:
        return pd.Series(dtype=float)
    raw = 1.0 / selected.set_index("Ticker")["Vol60"].replace(0, np.nan)
    return cap_and_normalize(raw)


def weights_equal(selected):
    if selected.empty:
        return pd.Series(dtype=float)
    return cap_and_normalize(pd.Series(1.0, index=selected["Ticker"].tolist()))


def metrics(returns, equity, turnover, cash_pct, horizon):
    r = pd.Series(returns, dtype=float).dropna()
    eq = pd.Series(equity, dtype=float).dropna()
    if r.empty or eq.empty:
        return {}
    ppy = 252.0 / horizon
    cagr = np.nan if (eq <= 0).any() else (eq.iloc[-1] / eq.iloc[0]) ** (ppy / len(r)) - 1
    total = eq.iloc[-1] / eq.iloc[0] - 1
    sd = r.std(ddof=1)
    vol = sd * math.sqrt(ppy) if len(r) > 1 else np.nan
    sharpe = r.mean() / sd * math.sqrt(ppy) if sd > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = r.mean() / downside * math.sqrt(ppy) if np.isfinite(downside) and downside > 0 else np.nan
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else np.inf
    return {
        "TotalReturn": total, "CAGR": cagr, "MaxDD": maxdd, "Vol": vol,
        "Sharpe": sharpe, "Sortino": sortino, "ProfitFactor": pf,
        "HitRate": float((r > 0).mean()), "MeanPeriodReturn": float(r.mean()),
        "Turnover": float(np.mean(turnover)) if turnover else 0.0,
        "CashPct": float(np.mean(cash_pct)) if cash_pct else 1.0,
        "Observations": int(len(r)),
    }


def simulate(valid, qdates, horizon, lag, selection_kind, threshold, direction,
             sizing, cost_bps, top_n=None):
    forward_col = f"Forward_H{horizon}_L{lag}"
    returns, equity = [], [1.0]
    turnovers, cash_pct = [], []
    previous = pd.Series(dtype=float)
    for qdate in qdates:
        day = valid[valid["Date"] == qdate].copy()
        if day.empty:
            continue
        selected = make_selection(day, selection_kind, threshold, direction, top_n)
        if sizing == "VOL_ADJ":
            weights = weights_vol_adj(selected)
        else:
            weights = weights_equal(selected)
        if weights.empty:
            gross, cash = 0.0, 1.0
        else:
            ret = selected.set_index("Ticker")[forward_col]
            gross = float((weights * ret.reindex(weights.index).fillna(0)).sum())
            cash = max(0.0, 1.0 - float(weights.sum()))
        union = previous.index.union(weights.index)
        turnover = float((weights.reindex(union, fill_value=0) - previous.reindex(union, fill_value=0)).abs().sum()) if len(union) else 0.0
        net = gross - turnover * (2.0 * cost_bps / 10000.0)
        returns.append(net)
        turnovers.append(turnover)
        cash_pct.append(cash)
        equity.append(equity[-1] * (1.0 + net))
        previous = weights
    return metrics(returns, equity, turnovers, cash_pct, horizon)


def benchmark_universe(valid, qdates, horizon, lag, cost_bps):
    return simulate(valid, qdates, horizon, lag, "RAW_TOP", threshold=np.inf,
                    direction=1, sizing="EQUAL_WEIGHT", cost_bps=cost_bps,
                    top_n=len(valid["Ticker"].unique()))


def date_shifted(base_dates, master_dates, offset):
    master = sorted(pd.Timestamp(x) for x in master_dates)
    pos = {d:i for i,d in enumerate(master)}
    out=[]
    for d in base_dates:
        i=pos.get(pd.Timestamp(d))
        if i is None:
            continue
        j=i+offset
        if 0 <= j < len(master):
            out.append(master[j])
    return out



def lowvol_selection(day, mode, cutoff=None, top_n=None):
    x = day.dropna(subset=["Vol252", "Vol60"]).copy()
    if x.empty:
        return x
    x["LowVol252"] = x["Vol252"].astype(float)
    if mode == "LOW_VOL_THRESHOLD":
        return x[x["LowVol252"] <= float(cutoff)].sort_values("LowVol252", ascending=True)
    return x.sort_values("LowVol252", ascending=True).head(min(int(top_n), len(x)))


def simulate_custom(valid, qdates, horizon, lag, selection_mode, cutoff, sizing, cost_bps, top_n=None):
    forward_col = f"Forward_H{horizon}_L{lag}"
    returns, equity = [], [1.0]
    turnovers, cash_pct = [], []
    previous = pd.Series(dtype=float)
    for qdate in qdates:
        day = valid[valid["Date"] == qdate].copy()
        if day.empty:
            continue
        selected = lowvol_selection(day, selection_mode, cutoff=cutoff, top_n=top_n)
        weights = weights_vol_adj(selected) if sizing == "VOL_ADJ" else weights_equal(selected)
        if weights.empty:
            gross, cash = 0.0, 1.0
        else:
            ret = selected.set_index("Ticker")[forward_col]
            gross = float((weights * ret.reindex(weights.index).fillna(0)).sum())
            cash = max(0.0, 1.0 - float(weights.sum()))
        union = previous.index.union(weights.index)
        turnover = float((weights.reindex(union, fill_value=0) - previous.reindex(union, fill_value=0)).abs().sum()) if len(union) else 0.0
        net = gross - turnover * (2.0 * cost_bps / 10000.0)
        returns.append(net)
        turnovers.append(turnover)
        cash_pct.append(cash)
        equity.append(equity[-1] * (1.0 + net))
        previous = weights
    return metrics(returns, equity, turnovers, cash_pct, horizon)


def compare_rows(a, b, keys):
    if a.empty or b.empty:
        return np.nan
    aa=a.set_index(keys)["CAGR"]
    bb=b.set_index(keys)["CAGR"]
    common=aa.index.intersection(bb.index)
    if len(common)==0:
        return np.nan
    return float((aa.loc[common]-bb.loc[common]).abs().max())


def main():
    if len(TICKERS) != 335:
        raise RuntimeError(f"Universo incorrecto: {len(TICKERS)} tickers; se esperaban 335.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print("=" * 72)
    print("CERE FASE 8D — CERE vs LOW-VOL FACTOR")
    print(f"RUN_ID: {run_id}")
    print(f"Tickers configurados: {len(TICKERS)}")
    print("Objetivo: determinar si CERE aporta valor incremental sobre LOW-VOL puro")
    print("=" * 72)

    datasets=[]; failures=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(build_ticker,t):t for t in TICKERS}
        for fut in as_completed(futures):
            t=futures[fut]
            try: df=fut.result()
            except Exception as e: print(f"[WARN] {t}: {e}"); df=None
            if df is None or df.empty: failures.append(t)
            else: datasets.append(df)
    data=pd.concat(datasets,ignore_index=True)
    data["Date"]=pd.to_datetime(data["Date"])
    print(f"Tickers OK: {data['Ticker'].nunique()}")
    print(f"Observaciones: {len(data):,}")
    print(f"Fallos: {len(failures)}")

    valid=data.dropna(subset=["Forward_H20_L1"]).copy()
    master_dates=sorted(valid["Date"].unique())
    base_dates=choose_dates(master_dates, BASE_HORIZON+EMBARGO_DAYS)
    discovery_dates=base_dates[:DISCOVERY_DATES]
    oos_dates=base_dates[DISCOVERY_DATES:]
    discovery=valid[valid["Date"].isin(discovery_dates)].copy()
    frozen=freeze_threshold(discovery)
    cutoff=float(-frozen["threshold"]) if frozen["direction"] < 0 else np.nan
    print(f"H20 calendar: total={len(base_dates)} discovery={len(discovery_dates)} OOS={len(oos_dates)}")
    print(f"Frozen CERE direction: {frozen['direction']} | threshold: {frozen['threshold']:.6f} | rho: {frozen['rho']:.6f}")
    print(f"Equivalent LOW_VOL cutoff when direction=-1: {cutoff:.6f}")

    results=[]
    # Fair comparison: identical dates, horizon, entry, sizing and costs.
    for cost in COST_BPS:
        for sizing in ["VOL_ADJ","EQUAL_WEIGHT"]:
            # CERE threshold selection.
            m=simulate(valid,oos_dates,20,1,"CERE_THRESHOLD",frozen["threshold"],frozen["direction"],sizing,cost)
            results.append({"Model":"CERE","Policy":"THRESHOLD","Sizing":sizing,"CostBps":cost,"TopN":np.nan,"Cutoff":cutoff,**m})
            # Pure low-vol threshold using mathematically equivalent cutoff if direction=-1.
            if frozen["direction"] < 0:
                m=simulate_custom(valid,oos_dates,20,1,"LOW_VOL_THRESHOLD",cutoff,sizing,cost)
                results.append({"Model":"LOW_VOL","Policy":"THRESHOLD_EQUIVALENT","Sizing":sizing,"CostBps":cost,"TopN":np.nan,"Cutoff":cutoff,**m})
            # Pure low-vol top N.
            for n in [5,10,15,20]:
                m=simulate_custom(valid,oos_dates,20,1,"LOW_VOL_TOPN",None,sizing,cost,top_n=n)
                results.append({"Model":"LOW_VOL","Policy":f"TOP{n}","Sizing":sizing,"CostBps":cost,"TopN":n,"Cutoff":np.nan,**m})

    dfres=pd.DataFrame(results)
    outdir=Path("cere_phase8d_lowvol")
    outdir.mkdir(exist_ok=True)
    dfres.to_csv(outdir/"phase8d_results.csv",index=False)

    # Structural equality audit: if CERE direction is -1, CERE threshold and equivalent low-vol threshold
    # must produce identical portfolio returns/metrics because the selection rule is mathematically identical.
    eq_pass=True
    eq_diffs=[]
    for sizing in ["VOL_ADJ","EQUAL_WEIGHT"]:
        for cost in COST_BPS:
            a=dfres[(dfres.Model=="CERE")&(dfres.Policy=="THRESHOLD")&(dfres.Sizing==sizing)&(dfres.CostBps==cost)].iloc[0]
            b=dfres[(dfres.Model=="LOW_VOL")&(dfres.Policy=="THRESHOLD_EQUIVALENT")&(dfres.Sizing==sizing)&(dfres.CostBps==cost)].iloc[0]
            for k in ["CAGR","MaxDD","Sharpe","Sortino","ProfitFactor","HitRate","Turnover","CashPct"]:
                d=abs(float(a[k])-float(b[k]))
                eq_diffs.append(d)
                if d > 1e-10: eq_pass=False

    # Benchmark comparison at 20 bps, VOL_ADJ.
    central=dfres[(dfres.CostBps==20)&(dfres.Sizing=="VOL_ADJ")].copy()
    print("\nRESULTADOS CENTRALES — 20 bps / VOL_ADJ")
    cols=["Model","Policy","CAGR","MaxDD","Sharpe","Sortino","ProfitFactor","HitRate","Turnover","CashPct","Observations"]
    print(central.sort_values("CAGR",ascending=False)[cols].to_string(index=False))

    core=central[(central.Model=="CERE")&(central.Policy=="THRESHOLD")].iloc[0]
    loweq=central[(central.Model=="LOW_VOL")&(central.Policy=="THRESHOLD_EQUIVALENT")].iloc[0]
    lowtops=central[central.Model=="LOW_VOL"].sort_values("CAGR",ascending=False)
    best_low=float(lowtops.iloc[0]["CAGR"]) if not lowtops.empty else np.nan
    core_better=bool(core["CAGR"] > best_low + 1e-12)

    audit={
        "RUN_ID":run_id,
        "ConfiguredTickers":len(TICKERS),
        "ValidTickers":int(data["Ticker"].nunique()),
        "Failures":len(failures),
        "BaseDates":len(base_dates),
        "DiscoveryDates":len(discovery_dates),
        "OOSDates":len(oos_dates),
        "FrozenDirection":int(frozen["direction"]),
        "FrozenThreshold":float(frozen["threshold"]),
        "EquivalentLowVolCutoff":cutoff,
        "ResultsRows":len(dfres),
        "NoEmptyResults":not dfres.empty,
        "NoNaNCoreMetrics":bool(dfres[["CAGR","MaxDD","Sharpe","ProfitFactor"]].notna().all().all()),
        "OOSCountsValid":len(oos_dates)>0,
        "CEREEqualsEquivalentLowVol":eq_pass,
        "MaxEqualityMetricDiff":max(eq_diffs) if eq_diffs else np.nan,
        "CoreCAGR20bps":float(core["CAGR"]),
        "EquivalentLowVolCAGR20bps":float(loweq["CAGR"]),
        "BestLowVolTopNCAGR20bps":best_low,
        "CoreBeatsBestLowVol":core_better,
    }
    pd.DataFrame([audit]).to_csv(outdir/"phase8d_audit.csv",index=False)
    print("\nAUDITORÍA FASE 8D")
    for k,v in audit.items(): print(f"{k}: {v}")

    # This phase is a diagnostic/refutation test: it passes only if structure is valid.
    structural_pass = all([
        audit["NoEmptyResults"], audit["NoNaNCoreMetrics"], audit["OOSCountsValid"],
        audit["CEREEqualsEquivalentLowVol"] if frozen["direction"] < 0 else True
    ])
    audit["STATUS"]="PASS" if structural_pass else "FAIL"
    pd.DataFrame([audit]).to_csv(outdir/"phase8d_audit.csv",index=False)
    print(f"STATUS: {audit['STATUS']}")
    print(f"Artifacts: {outdir.resolve()}")


if __name__ == "__main__":
    main()
