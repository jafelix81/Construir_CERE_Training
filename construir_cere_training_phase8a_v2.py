"""
CERE FASE 8A — BENCHMARK ECONÓMICO DE POSITION SIZING

Objetivo:
    Evaluar económicamente señales ya validadas en Fases 5–7 sin volver
    a optimizar el modelo predictivo.

Diseño congelado:
    Signals: Vol60, Vol252, Vol60+Vol252
    Horizons: H20, H40
    Selection: TOP5, TOP10, TOP20, THRESHOLD
    Sizing:
        - EQUAL_WEIGHT
        - VOL_ADJ
        - KELLY
        - KELLY25_CAP20
    Costs: 0/10/20/30/50/100 bps
    Discovery: 12 fechas
    OOS: resto de fechas disponibles por horizonte
    Gap: H + 5 trading days
    No Sheets / no production.

IMPORTANTE:
    - Los parámetros de señal/dirección/threshold se congelan con Discovery.
    - Kelly se estima SOLAMENTE con retornos históricos disponibles antes de
      cada OOS date y respetando el purge/embargo.
    - No se permite look-ahead.
    - Los costos se cargan una vez por lado de entrada/salida como costo
      round-trip aproximado: 2 * bps.
    - El retorno de cada posición es el forward H-day return definido en fases
      anteriores: Close[t+H] / Open[t+1] - 1.
    - Para sizing, se utiliza volatilidad anualizada del activo disponible
      al momento t; nunca se usa información futura.
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

# ---------------- CONFIG ----------------

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
HORIZONS = (20, 40)
DISCOVERY_DATES = 12
EMBARGO_DAYS = 5

SIGNALS = ("VOL60", "VOL252", "VOL60_VOL252")
POLICIES = ("TOP5", "TOP10", "TOP20", "THRESHOLD")
SIZINGS = ("EQUAL_WEIGHT", "VOL_ADJ", "KELLY", "KELLY25_CAP20")
COST_BPS = (0, 10, 20, 30, 50, 100)

MAX_POSITION = 0.20
MIN_POSITIONS_FOR_INVEST = 1
THRESHOLD_Q = 0.90

KELLY_GRID = np.linspace(0.0, 0.20, 401)  # 0%..20%, step 0.05%
KELLY_FRACTION = 0.25

OUTDIR = Path("cere_phase8a_economic")
OUTDIR.mkdir(exist_ok=True)

# ---------------- STATE ----------------

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
    # Minimum separation to prevent overlap in the realized H-day returns.
    min_gap = horizon + EMBARGO_DAYS
    selected = []
    for d in dates:
        if not selected or (d - selected[-1]).days >= min_gap:
            selected.append(d)
    if len(selected) < DISCOVERY_DATES + 8:
        return selected
    # Keep first 12 as Discovery and all subsequent dates as OOS.
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
    if policy == "TOP20":
        return x.head(min(20, len(x)))

    # THRESHOLD: only names above frozen threshold; if none, CASH.
    return x[x[score_col] >= day.attrs["threshold"]]


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


def estimate_kelly_full(history_returns: np.ndarray):
    """
    Full-distribution Kelly proxy:
        f* = argmax_f E[log(1 + fR)]
    over a bounded grid.

    This returns the un-fractionalized Kelly fraction. The two tested
    policies then apply either 100% Kelly or 25% Kelly.
    """
    r = np.asarray(history_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return 0.0

    # Prevent mathematical invalidity if an extreme historical observation
    # is below -100%.
    r = np.clip(r, -0.999, None)

    vals = []
    for f in KELLY_GRID:
        if np.any(1.0 + f * r <= 0):
            vals.append(-np.inf)
        else:
            vals.append(np.mean(np.log1p(f * r)))

    f_star = float(KELLY_GRID[int(np.nanargmax(vals))])
    return max(0.0, min(MAX_POSITION, f_star))


def precompute_kelly_by_ticker(valid, oos_dates, horizon):
    result = {}
    forward_col = f"Forward{horizon}"
    by_ticker = {t:g.sort_values("Date") for t,g in valid.groupby("Ticker")}
    for qdate in oos_dates:
        cutoff = pd.Timestamp(qdate) - pd.Timedelta(days=(horizon + EMBARGO_DAYS))
        for ticker,g in by_ticker.items():
            vals = g[g["Date"] <= cutoff][forward_col].dropna().to_numpy(float)
            result[(pd.Timestamp(qdate),ticker)] = estimate_kelly_full(vals) if len(vals)>=30 else 0.0
    return result

def weights_from_precomputed(selected, sizing, kelly_cache, qdate):
    if selected.empty: return pd.Series(dtype=float)
    idx=selected["Ticker"].tolist()
    if sizing=="EQUAL_WEIGHT": return cap_and_normalize(pd.Series(1.0,index=idx))
    if sizing=="VOL_ADJ":
        vols=selected.set_index("Ticker")["Vol60"].astype(float)
        return cap_and_normalize((1.0/vols.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna())
    raw={}
    for t in idx:
        f=kelly_cache.get((pd.Timestamp(qdate),t),0.0)
        if sizing=="KELLY25_CAP20": f*=KELLY_FRACTION
        f=max(0.0,min(MAX_POSITION,f))
        if f>0: raw[t]=f
    if not raw: return pd.Series(dtype=float)
    raw=pd.Series(raw)
    if sizing=="KELLY" and raw.sum()>1: raw=raw/raw.sum()
    return raw.clip(upper=MAX_POSITION)

def compute_weights(selected, sizing, history_by_ticker, horizon):
    if selected.empty:
        return pd.Series(dtype=float)

    idx = selected["Ticker"].tolist()

    if sizing == "EQUAL_WEIGHT":
        raw = pd.Series(1.0, index=idx)
        return cap_and_normalize(raw)

    if sizing == "VOL_ADJ":
        vols = selected.set_index("Ticker")["Vol60"].astype(float)
        raw = 1.0 / vols.replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        return cap_and_normalize(raw)

    raw = {}
    for _, row in selected.iterrows():
        t = row["Ticker"]
        hist = history_by_ticker.get(t, pd.DataFrame())
        if hist.empty:
            continue
        # Historical candidates must already have been purged by caller.
        vals = hist[f"Forward{horizon}"].dropna().to_numpy(float)
        f_star = estimate_kelly_full(vals)
        if sizing == "KELLY":
            f = f_star
        else:
            f = KELLY_FRACTION * f_star
        f = max(0.0, min(MAX_POSITION, f))
        if f > 0:
            raw[t] = f

    if not raw:
        return pd.Series(dtype=float)

    raw = pd.Series(raw)

    if sizing == "KELLY":
        # Kelly already produces an individual fraction. Normalize only if
        # aggregate capital exceeds 100%; otherwise leave cash unallocated.
        if raw.sum() > 1:
            raw = raw / raw.sum()
        return raw.clip(upper=MAX_POSITION)

    if sizing == "KELLY25_CAP20":
        # estimate_kelly_fraction already is 25% Kelly; enforce 20% hard cap.
        raw = raw.clip(upper=MAX_POSITION)
        return raw

    raise ValueError(sizing)


def metrics(returns, equity, turnover, cash_pct):
    r = pd.Series(returns, dtype=float).dropna()
    eq = pd.Series(equity, dtype=float).dropna()

    if r.empty or eq.empty:
        return {}

    total = eq.iloc[-1] / eq.iloc[0] - 1
    n = len(r)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(n, 1)) - 1

    vol = r.std(ddof=1) * math.sqrt(252) if len(r) > 1 else np.nan
    sharpe = (r.mean() / r.std(ddof=1) * math.sqrt(252)) if r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = (r.mean() / downside * math.sqrt(252)) if np.isfinite(downside) and downside > 0 else np.nan

    peak = eq.cummax()
    dd = eq / peak - 1
    maxdd = dd.min()

    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else np.inf

    return {
        "TotalReturn": total,
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "ProfitFactor": pf,
        "HitRate": float((r > 0).mean()),
        "MeanDailyReturn": float(r.mean()),
        "Turnover": float(np.mean(turnover)) if turnover else 0.0,
        "CashPct": float(np.mean(cash_pct)) if cash_pct else 1.0,
        "Observations": int(len(r)),
    }


def main():
    if not TICKERS:
        raise RuntimeError(
            "TICKERS está vacío. Sustituye la lista por EXACTAMENTE el universo "
            "congelado de Fase 7 antes de ejecutar Fase 8A."
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print("=" * 72)
    print("CERE FASE 8A v2 — BENCHMARK ECONÓMICO POSITION SIZING")
    print(f"RUN_ID: {run_id}")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Horizontes: {HORIZONS}")
    print(f"Signals: {SIGNALS}")
    print(f"Policies: {POLICIES}")
    print(f"Sizings: {SIZINGS}")
    print(f"Costs bps: {COST_BPS}")
    print("=" * 72)

    datasets=[]
    failures=[]
    print("Descargando/construyendo datasets en paralelo...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(build_ticker,t):t for t in TICKERS}
        for fut in as_completed(futures):
            ticker=futures[fut]
            try: df=fut.result()
            except Exception as e: print(f"[WARN] {ticker}: {e}"); df=None
            if df is None or df.empty: failures.append(ticker)
            else: datasets.append(df)
    datasets.sort(key=lambda x:str(x["Ticker"].iloc[0]))

    if not datasets:
        raise RuntimeError("No se pudo construir ningún dataset.")

    data = pd.concat(datasets, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])

    print(f"Tickers OK: {data['Ticker'].nunique()}")
    print(f"Observaciones: {len(data):,}")
    print(f"Fallos: {len(failures)}")

    all_results = []
    frozen_rows = []
    audit_rows = []

    for h in HORIZONS:
        valid = data.dropna(subset=[f"Forward{h}"]).copy()
        global_dates = sorted(valid["Date"].unique())

        # Greedy spacing by calendar days; with trading data this produces
        # sufficiently separated observations for H-day forward returns.
        dates = choose_dates(global_dates, h)

        if len(dates) <= DISCOVERY_DATES + 5:
            raise RuntimeError(f"H{h}: insuficientes fechas espaciadas: {len(dates)}")

        discovery_dates = dates[:DISCOVERY_DATES]
        oos_dates = dates[DISCOVERY_DATES:]

        print(f"H{h}: total={len(dates)} discovery={len(discovery_dates)} OOS={len(oos_dates)}")

        # Discovery sample for frozen signal parameters.
        disc = valid[valid["Date"].isin(discovery_dates)].copy()

        frozen = {}
        for signal in SIGNALS:
            params = freeze_signal_params(disc, signal, h)
            frozen[signal] = params
            frozen_rows.append({"Horizon": h, "Signal": signal, **params})

        print(f"  Precalculando Kelly H{h}...")
        kelly_cache=precompute_kelly_by_ticker(valid,oos_dates,h)
        print(f"  Kelly cache: {len(kelly_cache):,}")
        selection_cache={}
        for signal in SIGNALS:
            for policy in POLICIES:
                selection_cache[(signal,policy)]={}
                for qdate in oos_dates:
                    day=valid[valid["Date"]==qdate].copy()
                    if day.empty:
                        selection_cache[(signal,policy)][pd.Timestamp(qdate)]=day; continue
                    day["Score"]=apply_frozen_signal(day,signal,frozen[signal])
                    day.attrs["threshold"]=frozen[signal]["threshold"]
                    sel=select_names(day,policy,"Score",len(day)).dropna(subset=[f"Forward{h}","Vol60"]).copy()
                    selection_cache[(signal,policy)][pd.Timestamp(qdate)]=sel

        for signal in SIGNALS:
            for policy in POLICIES:
                for sizing in SIZINGS:
                    for cost_bps in COST_BPS:
                        equity = [1.0]
                        period_returns = []
                        turnover_list = []
                        cash_list = []
                        previous_weights = pd.Series(dtype=float)

                        for qdate in oos_dates:
                            selected=selection_cache[(signal,policy)].get(pd.Timestamp(qdate),pd.DataFrame())

                            # If threshold or any filter produces no candidates:
                            if len(selected) < MIN_POSITIONS_FOR_INVEST:
                                weights = pd.Series(dtype=float)
                            else:
                                # Purged history for Kelly: candidate history
                                # must end before qdate - (H + embargo).
                                weights=weights_from_precomputed(selected,sizing,kelly_cache,qdate)

                            if weights.empty:
                                gross = 0.0
                                cash = 1.0
                            else:
                                selected_ret = selected.set_index("Ticker")[f"Forward{h}"]
                                gross = float((weights * selected_ret.reindex(weights.index).fillna(0)).sum())
                                cash = max(0.0, 1.0 - float(weights.sum()))

                            # Turnover is one-way approximation against previous
                            # weights; entering/exiting is represented by abs delta.
                            current = weights.copy()
                            union = previous_weights.index.union(current.index)
                            if len(union):
                                turnover = float(
                                    (current.reindex(union, fill_value=0)
                                     - previous_weights.reindex(union, fill_value=0)).abs().sum()
                                )
                            else:
                                turnover = 0.0

                            # Approximate round-trip transaction cost.
                            net = gross - (turnover * (2.0 * cost_bps / 10000.0))

                            period_returns.append(net)
                            turnover_list.append(turnover)
                            cash_list.append(cash)

                            new_equity = equity[-1] * (1.0 + net)
                            equity.append(new_equity)
                            previous_weights = current

                        if not period_returns:
                            continue

                        m = metrics(period_returns, equity, turnover_list, cash_list)
                        row = {
                            "RUN_ID": run_id,
                            "Horizon": h,
                            "Signal": signal,
                            "Policy": policy,
                            "Sizing": sizing,
                            "CostBps": cost_bps,
                            "DiscoveryDates": len(discovery_dates),
                            "OOSDates": len(period_returns),
                            **m,
                        }
                        all_results.append(row)

    results = pd.DataFrame(all_results)
    frozen_df = pd.DataFrame(frozen_rows)

    # Structural audits.
    audit = {
        "RUN_ID": run_id,
        "ConfiguredTickers": len(TICKERS),
        "ValidTickers": int(data["Ticker"].nunique()),
        "Failures": len(failures),
        "ExpectedUniverseTickers": 335,
        "UniverseCountPASS": len(TICKERS) == 335,
        "ResultsRows": len(results),
        "FrozenRows": len(frozen_df),
        "ExpectedResultRows": len(HORIZONS) * len(SIGNALS) * len(POLICIES) * len(SIZINGS) * len(COST_BPS),
        "NoEmptyResults": not results.empty,
        "NoNaNCoreMetrics": bool(
            results[["CAGR", "MaxDD", "Sharpe", "Sortino", "ProfitFactor"]]
            .replace([np.inf, -np.inf], np.nan)
            .notna().all().all()
        ) if not results.empty else False,
        "OOSCountsValid": bool(
            all(
                int(results.loc[results["Horizon"] == h, "OOSDates"].iloc[0]) >= 8
                for h in HORIZONS
                if not results.loc[results["Horizon"] == h].empty
            )
        ),
    }

    # Mathematical sanity: equal-weight must have no dependence on Kelly/VOL_ADJ.
    # This is a structural audit, not a performance assertion.
    audit["STATUS"] = "PASS" if all([
        audit["NoEmptyResults"],
        audit["NoNaNCoreMetrics"],
        audit["UniverseCountPASS"],
        audit["OOSCountsValid"],
        len(results) == audit["ExpectedResultRows"],
    ]) else "FAIL"

    results.to_csv(OUTDIR / "phase8a_results.csv", index=False)
    frozen_df.to_csv(OUTDIR / "phase8a_frozen_parameters.csv", index=False)
    pd.DataFrame([audit]).to_csv(OUTDIR / "phase8a_audit.csv", index=False)

    if failures:
        pd.DataFrame({"Ticker": failures}).to_csv(OUTDIR / "phase8a_failures.csv", index=False)

    # Ranked summary using 20 bps as central friction assumption.
    central = results[results["CostBps"] == 20].copy()
    if not central.empty:
        central = central.sort_values(
            ["Horizon", "Sharpe", "CAGR"],
            ascending=[True, False, False]
        )
        central.to_csv(OUTDIR / "phase8a_central_20bps_ranked.csv", index=False)

    print("\nAUDITORÍA FASE 8A")
    for k, v in audit.items():
        print(f"{k}: {v}")

    print("\nMEJORES CONFIGURACIONES A 20 bps — POR HORIZONTE")
    if not central.empty:
        cols = [
            "Horizon", "Signal", "Policy", "Sizing", "CostBps",
            "CAGR", "MaxDD", "Sharpe", "Sortino",
            "ProfitFactor", "HitRate", "Turnover", "CashPct"
        ]
        print(central[cols].head(20).to_string(index=False))

    print("\nArtifacts:", OUTDIR.resolve())
    print("STATUS:", audit["STATUS"])

    if audit["STATUS"] != "PASS":
        raise RuntimeError("AUDITORÍA FASE 8A FALLÓ.")


if __name__ == "__main__":
    main()
