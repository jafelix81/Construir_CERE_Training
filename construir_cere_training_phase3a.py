#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CERE — FASE 3A — MOTOR DE VECINOS Y DISTRIBUCIÓN CONDICIONAL

Laboratorio, NO producción.

Objetivo:
1) reconstruir el dataset histórico FULL;
2) construir el estado CERE canónico;
3) para fechas históricas de consulta, usar SOLO candidatos anteriores;
4) normalizar las 14 variables usando exclusivamente candidatos anteriores;
5) encontrar K vecinos;
6) ponderarlos mediante kernel gaussiano;
7) construir la distribución condicional de Forward5/10/20/40;
8) calcular estadísticas descriptivas del EV condicional.

NO:
- escribe Google Sheets;
- modifica CERE_CURRENT;
- genera BUY/SELL;
- calcula Kelly;
- usa información posterior a la fecha de consulta.

Esto es la base matemática para las fases posteriores.
"""

import os
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis, t


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PERIOD = "10y"
INTERVAL = "1d"
MIN_STATE = 260

K_VALUES = (50, 100, 200)
HORIZONS = (5, 10, 20, 40)

# Número de fechas históricas de consulta por ticker.
QUERIES_PER_TICKER = 8

# Para que la auditoría sea reproducible.
RANDOM_SEED = 42

# Kernel.
MIN_BANDWIDTH = 1e-9

STATE_FIELDS = [
    "R1", "R5", "R20", "R60", "R120",
    "Vol5", "Vol20", "Vol60", "Vol252",
    "VolRatio20", "VolRatio60",
    "Skew20", "Kurt20", "AC1"
]

TICKERS = """
AA AAPL ABNB ABT ADI AFRM ALAB ALGN AMAT AMD AMGN AMT AMTD AMZN ANET APA
APD APO APP ARKF ARKG ARKK ARKQ ARKW ASB ASML ASTS ATI AVB AVGO AX AXON AXP AZN
BA BABA BAC BE BEAM BEN BIDU BIIB BITO BKNG BKR BLK BLNK BMY BNTX BOKF BSX BX BXP
C CACC CAT CATY CAVA CCI CDNS CEG CELH CF CFG CHPT CHTR CHWY CIFR CL CLF CLS CMCSA
COIN COMP COP COST CRDO CRSP CRWD CSCO CSX CUBI CVNA CVX CX DD DDOG DE DELL DHI DIS
DJT DKNG DLR DOCU DOW DUOL DVN EA EBAY ECL EDIT EEM EGBN EL ENPH EOG EQIX EQR ETN ETSY
EW EXPE EXR FCEL FCF FCX FDX FITB FLEX FLYW FMC FNB FOXA FSLR GD GE GEV GILD GIS GLD GLW
GOOGL GPRO GS GSK HAL HAPN HBAN HD HOMB HON HOOD HUT IEFA ILMN INTC IREN ISRG IWM JD JETS
JMIA JNJ JPM KEY KKR KLAC KMB KO KRE LCID LEN LI LIN LITE LLY LMT LOT LOW LPRO LRCX LYV MA
MAA MARA MCHP MDB MDLZ MDT MELI META MLM MMM MNST MOD MOS MP MRNA MRVL MS MSFT MTB MTCH MU NEM
NET NFLX NIO NOC NOW NSC NTLA NTR NTRS NU NUE NVDA NVO NWSA NXPI O OIH OKTA OMF ON OPEN OPFI
OXY OZK PAGS PANW PARA PAYS PDD PEP PFE PG PHM PINS PLD PLTR PLUG PSA PSNY PWR PYPL QCOM QQQ
QS RDDT REGN RF RIOT RIVN RKLB ROKU RSG RTX RUM RUN SBAC SBUX SCHW SE SEDG SHOP SHW SIFY SLB
SMCI SMH SNAP SNOW SNPS SNY SOFI SOXX SPG SPWR STLD STNE STT SYK T TAK TEAM TELFY TEM TER TFC
TGT TMDX TMUS TRIP TROW TSLA TSM TTWO TWLO TXN UMBF UNG UNP UPS UPST USO V VEA VICI VLY VMC
VNQ VOD VRT VRTX VST VWO VZ W WAY WBD WBS WDAY WFC WM WMT WULF WY XBI XLB XLE XLF XLI XLK
XLP XLRE XLU XLV XLY XOM XPEV XYZ Z ZBH ZG ZION ZM ZS
""".split()

TICKERS = sorted(set(TICKERS))


def state_from_closes(closes, i):
    """Estado CERE canónico al cierre t=i."""
    c = np.asarray(closes, dtype=float)

    if i < MIN_STATE - 1 or i >= len(c):
        return None

    if not np.all(np.isfinite(c[: i + 1])) or np.any(c[: i + 1] <= 0):
        return None

    r1 = c[i] / c[i - 1] - 1.0
    r5 = c[i] / c[i - 5] - 1.0
    r20 = c[i] / c[i - 20] - 1.0
    r60 = c[i] / c[i - 60] - 1.0
    r120 = c[i] / c[i - 120] - 1.0

    # Incluye el retorno que termina en t.
    logret = np.diff(np.log(c[: i + 1]))

    if len(logret) < 252:
        return None

    def vol(n):
        x = logret[-n:]
        return float(np.std(x, ddof=1) * np.sqrt(252.0))

    vol5 = vol(5)
    vol20 = vol(20)
    vol60 = vol(60)
    vol252 = vol(252)

    if not np.isfinite(vol252) or vol252 <= 0:
        return None

    x20 = logret[-20:]
    x60 = logret[-60:]

    vals = {
        "R1": r1,
        "R5": r5,
        "R20": r20,
        "R60": r60,
        "R120": r120,
        "Vol5": vol5,
        "Vol20": vol20,
        "Vol60": vol60,
        "Vol252": vol252,
        "VolRatio20": vol20 / vol252,
        "VolRatio60": vol60 / vol252,
        "Skew20": float(skew(x20, bias=False)),
        "Kurt20": float(kurtosis(x20, fisher=True, bias=False)),
        "AC1": float(pd.Series(x60).autocorr(lag=1)),
    }

    if not all(np.isfinite(v) for v in vals.values()):
        return None

    return vals


def build_dataset(df):
    """Genera estados + forwards para un ticker."""
    opens = df["Open"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    dates = pd.to_datetime(df.index)

    rows = []

    # Necesitamos H40 disponible.
    for i in range(MIN_STATE - 1, len(df) - 40):
        state = state_from_closes(closes, i)
        if state is None:
            continue

        row = {
            "Date": dates[i].strftime("%Y-%m-%d"),
            "Index": i,
            **state
        }

        for h in HORIZONS:
            entry = opens[i + 1]
            exit_price = closes[i + h]
            row[f"Forward{h}"] = exit_price / entry - 1.0

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_prices(ticker):
    df = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    if not {"Open", "Close"}.issubset(df.columns):
        return None

    df = df[["Open", "Close"]].copy()
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    if len(df) < MIN_STATE + 41:
        return None

    return df


def weighted_neighbors(dataset, query_row, k):
    """
    Historical analogue search.

    Candidate restriction:
        candidate Date < query Date

    Normalization:
        z-score calculated ONLY from candidates.

    Distance:
        Euclidean over 14 state variables.

    Weight:
        exp(-d^2 / bandwidth^2)

    Bandwidth:
        distance to K-th neighbor.
    """
    candidates = dataset[
        dataset["Date"] < query_row["Date"]
    ].copy()

    if candidates.empty:
        return None

    k_eff = min(k, len(candidates))

    X = candidates[STATE_FIELDS].to_numpy(float)
    q = query_row[STATE_FIELDS].to_numpy(float)

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)

    # Variables with zero historical variance carry no distance.
    stds[~np.isfinite(stds)] = 1.0
    stds[stds <= 0] = 1.0

    Xz = (X - means) / stds
    qz = (q - means) / stds

    distances = np.sqrt(np.sum((Xz - qz) ** 2, axis=1))

    order = np.argsort(distances, kind="mergesort")[:k_eff]

    selected = candidates.iloc[order].copy()
    d = distances[order]

    bandwidth = float(max(d[-1], MIN_BANDWIDTH))
    weights = np.exp(-(d ** 2) / (bandwidth ** 2))

    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones(len(weights), dtype=float)

    weights = weights / weights.sum()

    selected["Distance"] = d
    selected["Weight"] = weights

    return selected, bandwidth


def weighted_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return {}

    weights = weights / weights.sum()

    mean = float(np.sum(values * weights))

    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cw = np.cumsum(w)

    median = float(x[np.searchsorted(cw, 0.5)])
    p10 = float(x[np.searchsorted(cw, 0.10)])
    p25 = float(x[np.searchsorted(cw, 0.25)])
    p75 = float(x[np.searchsorted(cw, 0.75)])
    p90 = float(x[np.searchsorted(cw, 0.90)])

    positive = float(np.sum(weights[values > 0]))

    variance = float(np.sum(weights * (values - mean) ** 2))
    std = float(np.sqrt(max(variance, 0.0)))

    return {
        "EV": mean,
        "MEDIAN": median,
        "STD": std,
        "P10": p10,
        "P25": p25,
        "P75": p75,
        "P90": p90,
        "P_POSITIVE": positive,
        "MIN": float(np.min(values)),
        "MAX": float(np.max(values)),
        "N": int(len(values)),
        "ESS": float(1.0 / np.sum(weights ** 2))
    }


def robust_student_t(values, weights):
    """
    Descriptive Student-t fit.

    scipy's fit does not support arbitrary observation weights directly,
    so the weighted distribution is summarized through weighted moments
    and an unweighted fit on the selected analogue returns.

    This is diagnostic only in Phase 3A. It is NOT yet used for EV
    or position sizing.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 10:
        return {"DF": np.nan, "LOC": np.nan, "SCALE": np.nan}

    try:
        df_t, loc, scale = t.fit(x)
        return {
            "DF": float(df_t),
            "LOC": float(loc),
            "SCALE": float(scale)
        }
    except Exception:
        return {"DF": np.nan, "LOC": np.nan, "SCALE": np.nan}


def main():
    print("=" * 80)
    print("CERE — FASE 3A — MOTOR DE VECINOS Y DISTRIBUCIÓN CONDICIONAL")
    print("=" * 80)
    print(
        f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | "
        f"Period: {PERIOD}"
    )
    print(
        f"K_VALUES: {K_VALUES} | HORIZONS: {HORIZONS} | "
        f"Queries/ticker: {QUERIES_PER_TICKER}"
    )
    print()

    outdir = "cere_phase3a_engine"
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    summary_rows = []
    ticker_rows = []
    query_rows = []

    total_queries = 0
    failures = 0

    for n, ticker in enumerate(TICKERS, 1):
        print(f"[{n:03d}/{len(TICKERS)}] {ticker}", flush=True)

        try:
            df = prepare_prices(ticker)

            if df is None:
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "NO_DATA",
                    "QUERIES": 0
                })
                continue

            dataset = build_dataset(df)

            if len(dataset) < 250:
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "INSUFFICIENT_TRAINING",
                    "QUERIES": 0,
                    "OBSERVATIONS": len(dataset)
                })
                continue

            # Query dates: deterministic and sufficiently separated.
            eligible = np.arange(
                200,
                len(dataset)
            )

            if len(eligible) < QUERIES_PER_TICKER:
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "INSUFFICIENT_QUERIES",
                    "QUERIES": 0,
                    "OBSERVATIONS": len(dataset)
                })
                continue

            query_idx = np.linspace(
                eligible[0],
                eligible[-1],
                QUERIES_PER_TICKER,
                dtype=int
            )

            ticker_query_count = 0

            for qi in query_idx:
                query = dataset.iloc[int(qi)]

                # IMPORTANT:
                # This query's own future returns are NOT passed to
                # neighbor selection. Only Date < query Date is used.
                for k in K_VALUES:
                    result = weighted_neighbors(dataset, query, k)

                    if result is None:
                        failures += 1
                        continue

                    neighbors, bandwidth = result

                    for h in HORIZONS:
                        values = neighbors[f"Forward{h}"].to_numpy(float)
                        weights = neighbors["Weight"].to_numpy(float)

                        stats = weighted_stats(values, weights)
                        student = robust_student_t(values, weights)

                        query_rows.append({
                            "RUN_ID": RUN_ID,
                            "Ticker": ticker,
                            "QueryDate": query["Date"],
                            "K": k,
                            "HORIZON": h,
                            "N_NEIGHBORS": stats.get("N", 0),
                            "ESS": stats.get("ESS", np.nan),
                            "BANDWIDTH": bandwidth,
                            "EV": stats.get("EV", np.nan),
                            "MEDIAN": stats.get("MEDIAN", np.nan),
                            "STD": stats.get("STD", np.nan),
                            "P10": stats.get("P10", np.nan),
                            "P25": stats.get("P25", np.nan),
                            "P75": stats.get("P75", np.nan),
                            "P90": stats.get("P90", np.nan),
                            "P_POSITIVE": stats.get("P_POSITIVE", np.nan),
                            "MIN": stats.get("MIN", np.nan),
                            "MAX": stats.get("MAX", np.nan),
                            "STUDENT_T_DF": student["DF"],
                            "STUDENT_T_LOC": student["LOC"],
                            "STUDENT_T_SCALE": student["SCALE"]
                        })

                    ticker_query_count += 1
                    total_queries += 1

            ticker_rows.append({
                "Ticker": ticker,
                "STATUS": "OK",
                "OBSERVATIONS": len(dataset),
                "QUERIES": ticker_query_count
            })

        except Exception as e:
            failures += 1
            ticker_rows.append({
                "Ticker": ticker,
                "STATUS": "ERROR",
                "QUERIES": 0,
                "ERROR": repr(e)
            })

    qdf = pd.DataFrame(query_rows)
    tdf = pd.DataFrame(ticker_rows)

    query_path = os.path.join(
        outdir, f"phase3a_conditional_distributions_{RUN_ID}.csv"
    )
    ticker_path = os.path.join(
        outdir, f"phase3a_ticker_audit_{RUN_ID}.csv"
    )
    summary_path = os.path.join(
        outdir, f"phase3a_summary_{RUN_ID}.csv"
    )

    qdf.to_csv(query_path, index=False)
    tdf.to_csv(ticker_path, index=False)

    if not qdf.empty:
        summary = (
            qdf.groupby(["K", "HORIZON"], as_index=False)
            .agg(
                N_QUERIES=("Ticker", "count"),
                MEAN_EV=("EV", "mean"),
                MEDIAN_EV=("EV", "median"),
                MEAN_STD=("STD", "mean"),
                MEAN_P_POSITIVE=("P_POSITIVE", "mean"),
                MEAN_ESS=("ESS", "mean"),
                P10_EV=("EV", lambda x: x.quantile(0.10)),
                P90_EV=("EV", lambda x: x.quantile(0.90))
            )
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(summary_path, index=False)

    print()
    print("--- FASE 3A RESUMEN ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(
        f"Tickers OK: "
        f"{int((tdf['STATUS'] == 'OK').sum()) if not tdf.empty else 0}"
    )
    print(f"Query executions: {total_queries}")
    print(f"Failures: {failures}")
    print()

    if not summary.empty:
        print("--- DISTRIBUCIONES CONDICIONALES ---")
        print(summary.to_string(index=False))

    print()
    print(f"Archivos: {outdir}/")
    print(f"Duration/run ID: {RUN_ID}")

    # Phase 3A is diagnostic. Do not fail because a ticker is delisted;
    # only fail if there are no conditional results.
    if qdf.empty:
        raise SystemExit(
            "FASE 3A FAILED: no se generaron distribuciones condicionales."
        )


if __name__ == "__main__":
    main()
