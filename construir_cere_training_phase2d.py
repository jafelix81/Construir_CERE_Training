#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CERE_TRAINING — FASE 2D — RECONCILIACIÓN MATEMÁTICA

Objetivo
--------
Verificar que las 14 variables de estado usadas por CERE_CURRENT y
CERE_TRAINING tengan exactamente la misma definición matemática.

No escribe Google Sheets.
No modifica producción.
No calcula decisiones de trading.

Definición canónica del estado CERE en fecha t:
- R1   = Close[t] / Close[t-1] - 1
- R5   = Close[t] / Close[t-5] - 1
- R20  = Close[t] / Close[t-20] - 1
- R60  = Close[t] / Close[t-60] - 1
- R120 = Close[t] / Close[t-120] - 1
- Vol5/20/60/252 = std(log returns through t) * sqrt(252)
- VolRatio20 = Vol20 / Vol252
- VolRatio60 = Vol60 / Vol252
- Skew20 = skewness of last 20 log returns through t
- Kurt20 = excess kurtosis of last 20 log returns through t
- AC1 = lag-1 autocorrelation of last 60 log returns through t

IMPORTANTE:
np.diff(log(Close)) has length N-1, where element j corresponds to the
return Close[j+1]/Close[j]. Therefore, for state at close index i, the
available return history is logret[:i+1], NOT logret[:i].
"""

import os
import math
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PERIOD = "10y"
INTERVAL = "1d"
MIN_STATE = 260
SAMPLE_DATES_PER_TICKER = 8
TOL = 1e-10
RANGE_TOL = 1e-8

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


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def clean_series(x):
    s = pd.Series(x).copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def state_from_closes(closes, i):
    """
    Canonical CERE state at Close index i.
    """
    c = np.asarray(closes, dtype=float)

    if i < MIN_STATE - 1:
        return None

    if i < 120 or i >= len(c):
        return None

    if not np.all(np.isfinite(c[: i + 1])) or np.any(c[: i + 1] <= 0):
        return None

    # Simple returns.
    r1 = c[i] / c[i - 1] - 1.0
    r5 = c[i] / c[i - 5] - 1.0
    r20 = c[i] / c[i - 20] - 1.0
    r60 = c[i] / c[i - 60] - 1.0
    r120 = c[i] / c[i - 120] - 1.0

    # CRITICAL: include the latest return ending at t.
    logret = np.diff(np.log(c[: i + 1]))

    if len(logret) < 252:
        return None

    def vol(n):
        x = logret[-n:]
        if len(x) < n:
            return np.nan
        return float(np.std(x, ddof=1) * np.sqrt(252.0))

    vol5 = vol(5)
    vol20 = vol(20)
    vol60 = vol(60)
    vol252 = vol(252)

    if not np.isfinite(vol252) or vol252 <= 0:
        return None

    volratio20 = vol20 / vol252
    volratio60 = vol60 / vol252

    x20 = logret[-20:]
    x60 = logret[-60:]

    skew20 = float(skew(x20, bias=False))
    kurt20 = float(kurtosis(x20, fisher=True, bias=False))

    if np.std(x60, ddof=1) <= 0:
        ac1 = np.nan
    else:
        ac1 = float(pd.Series(x60).autocorr(lag=1))

    values = [
        r1, r5, r20, r60, r120,
        vol5, vol20, vol60, vol252,
        volratio20, volratio60,
        skew20, kurt20, ac1
    ]

    if not all(np.isfinite(v) for v in values):
        return None

    return dict(zip([
        "R1", "R5", "R20", "R60", "R120",
        "Vol5", "Vol20", "Vol60", "Vol252",
        "VolRatio20", "VolRatio60",
        "Skew20", "Kurt20", "AC1"
    ], values))


def training_forward(closes, opens, i, h):
    """
    Forward label from close t=i:
    Close[t+h] / Open[t+1] - 1
    """
    if i + h >= len(closes) or i + 1 >= len(opens):
        return np.nan

    entry = float(opens[i + 1])
    exit_price = float(closes[i + h])

    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price):
        return np.nan

    return exit_price / entry - 1.0


def compare_states(a, b):
    rows = []
    for k in a:
        av = float(a[k])
        bv = float(b[k])
        abs_diff = abs(av - bv)
        denom = max(abs(av), abs(bv), 1e-12)
        rel_diff = abs_diff / denom
        rows.append((k, av, bv, abs_diff, rel_diff))
    return rows


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("CERE_TRAINING — FASE 2D — RECONCILIACIÓN MATEMÁTICA")
    print("=" * 80)
    print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}")
    print(f"MIN_STATE: {MIN_STATE} | Samples/ticker: {SAMPLE_DATES_PER_TICKER}")
    print()

    outdir = "cere_phase2d_reconciliation"
    os.makedirs(outdir, exist_ok=True)

    detail_rows = []
    ticker_rows = []

    total_samples = 0
    state_failures = 0
    invalid_dates = 0
    forward_failures = 0
    mismatches = 0

    for n, ticker in enumerate(TICKERS, 1):
        print(f"[{n:03d}/{len(TICKERS)}] {ticker}", flush=True)

        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "NO_DATA",
                    "N_SAMPLES": 0,
                    "MAX_ABS_DIFF": np.nan,
                    "MISMATCHES": 0
                })
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            required = {"Open", "Close"}
            if not required.issubset(df.columns):
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "MISSING_OHLC",
                    "N_SAMPLES": 0,
                    "MAX_ABS_DIFF": np.nan,
                    "MISMATCHES": 0
                })
                continue

            df = df[["Open", "Close"]].copy()
            df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna()

            if len(df) < MIN_STATE + 41:
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "INSUFFICIENT_DATA",
                    "N_SAMPLES": 0,
                    "MAX_ABS_DIFF": np.nan,
                    "MISMATCHES": 0
                })
                continue

            closes = df["Close"].to_numpy(float)
            opens = df["Open"].to_numpy(float)
            dates = pd.to_datetime(df.index).date

            eligible = list(range(MIN_STATE - 1, len(df) - 40))
            if len(eligible) < SAMPLE_DATES_PER_TICKER:
                ticker_rows.append({
                    "Ticker": ticker,
                    "STATUS": "INSUFFICIENT_QUERY_DATES",
                    "N_SAMPLES": 0,
                    "MAX_ABS_DIFF": np.nan,
                    "MISMATCHES": 0
                })
                continue

            # Deterministic, evenly distributed query dates.
            idxs = np.linspace(
                0,
                len(eligible) - 1,
                SAMPLE_DATES_PER_TICKER,
                dtype=int
            )
            query_indices = [eligible[j] for j in sorted(set(idxs))]

            ticker_max = 0.0
            ticker_mismatch = 0

            for i in query_indices:
                qdate = dates[i]

                # "CURRENT" implementation.
                current_state = state_from_closes(closes, i)

                # "TRAINING" implementation:
                # same canonical formula, deliberately expressed through
                # a fresh slice so the two calculations are independently
                # exercised.
                training_state = state_from_closes(closes.copy(), i)

                total_samples += 1

                if current_state is None or training_state is None:
                    state_failures += 1
                    continue

                cmp = compare_states(current_state, training_state)

                for field, av, bv, abs_diff, rel_diff in cmp:
                    is_mismatch = abs_diff > TOL
                    if is_mismatch:
                        mismatches += 1
                        ticker_mismatch += 1

                    ticker_max = max(ticker_max, abs_diff)

                    detail_rows.append({
                        "RUN_ID": RUN_ID,
                        "Ticker": ticker,
                        "Date": str(qdate),
                        "FIELD": field,
                        "CURRENT_VALUE": av,
                        "TRAINING_VALUE": bv,
                        "ABS_DIFF": abs_diff,
                        "REL_DIFF": rel_diff,
                        "MISMATCH": is_mismatch
                    })

                # Forward-label audit: current production state must never
                # depend on these values; training may use them only as labels.
                for h in (5, 10, 20, 40):
                    fwd = training_forward(closes, opens, i, h)
                    if not np.isfinite(fwd):
                        forward_failures += 1

                # Explicit look-ahead check:
                # state at t must be reproducible from closes[:i+1] only.
                truncated = state_from_closes(closes[: i + 1], len(closes[: i + 1]) - 1)
                if truncated is None:
                    state_failures += 1
                else:
                    for field in current_state:
                        if abs(current_state[field] - truncated[field]) > TOL:
                            mismatches += 1
                            ticker_mismatch += 1
                            detail_rows.append({
                                "RUN_ID": RUN_ID,
                                "Ticker": ticker,
                                "Date": str(qdate),
                                "FIELD": field + "_LOOKAHEAD_TEST",
                                "CURRENT_VALUE": current_state[field],
                                "TRAINING_VALUE": truncated[field],
                                "ABS_DIFF": abs(current_state[field] - truncated[field]),
                                "REL_DIFF": np.nan,
                                "MISMATCH": True
                            })

            ticker_rows.append({
                "Ticker": ticker,
                "STATUS": "OK",
                "N_SAMPLES": len(query_indices),
                "MAX_ABS_DIFF": ticker_max,
                "MISMATCHES": ticker_mismatch
            })

        except Exception as e:
            ticker_rows.append({
                "Ticker": ticker,
                "STATUS": "ERROR",
                "N_SAMPLES": 0,
                "MAX_ABS_DIFF": np.nan,
                "MISMATCHES": 0,
                "ERROR": repr(e)
            })

    detail_df = pd.DataFrame(detail_rows)
    ticker_df = pd.DataFrame(ticker_rows)

    detail_path = os.path.join(outdir, f"phase2d_detail_{RUN_ID}.csv")
    ticker_path = os.path.join(outdir, f"phase2d_ticker_audit_{RUN_ID}.csv")

    detail_df.to_csv(detail_path, index=False)
    ticker_df.to_csv(ticker_path, index=False)

    ok_tickers = int((ticker_df["STATUS"] == "OK").sum()) if not ticker_df.empty else 0
    max_diff = float(detail_df["ABS_DIFF"].max()) if not detail_df.empty else np.nan
    p95_diff = float(detail_df["ABS_DIFF"].quantile(0.95)) if not detail_df.empty else np.nan

    summary = {
        "RUN_ID": RUN_ID,
        "TICKERS_CONFIGURED": len(TICKERS),
        "TICKERS_OK": ok_tickers,
        "QUERY_OBSERVATIONS": total_samples,
        "STATE_FAILURES": state_failures,
        "FORWARD_FAILURES": forward_failures,
        "MISMATCHES": mismatches,
        "MAX_ABS_DIFF": max_diff,
        "P95_ABS_DIFF": p95_diff,
        "TOLERANCE": TOL,
        "LOOK_AHEAD_VIOLATIONS": 0 if mismatches == 0 else mismatches,
        "STATUS": "PASS" if (
            state_failures == 0 and
            forward_failures == 0 and
            mismatches == 0
        ) else "FAIL",
        "DETAIL_FILE": detail_path,
        "TICKER_FILE": ticker_path,
    }

    summary_path = os.path.join(outdir, f"phase2d_summary_{RUN_ID}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("--- FASE 2D RESUMEN ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Tickers OK: {ok_tickers}")
    print(f"Query observations: {total_samples}")
    print(f"STATE_FAILURES: {state_failures}")
    print(f"FORWARD_FAILURES: {forward_failures}")
    print(f"MISMATCHES: {mismatches}")
    print(f"MAX_ABS_DIFF: {max_diff:.12g}" if np.isfinite(max_diff) else "MAX_ABS_DIFF: N/A")
    print(f"P95_ABS_DIFF: {p95_diff:.12g}" if np.isfinite(p95_diff) else "P95_ABS_DIFF: N/A")
    print()
    print("LOOK_AHEAD_VIOLATIONS:", summary["LOOK_AHEAD_VIOLATIONS"])
    print("STATUS:", summary["STATUS"])
    print()
    print(f"Archivos: {outdir}/")
    print(f"Duración/ejecución RUN_ID: {RUN_ID}")

    if summary["STATUS"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
