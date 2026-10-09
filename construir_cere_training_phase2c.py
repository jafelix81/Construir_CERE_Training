#!/usr/bin/env python3
"""
CERE — FASE 2C — BENCHMARK FULL vs COMPACT

Purpose
-------
Test whether sampling 1/5 of the historical observations changes the
conditional EV produced by the historical-neighbor engine.

This is a BENCHMARK ONLY:
- no Google Sheets
- no production decisions
- no Spark
- no look-ahead

Method
------
For each ticker, reconstruct the historical state/forward dataset exactly
as in Phase 2A/2B. Select deterministic query dates from the valid history.
For each query date:
  1) use only observations strictly BEFORE the query date as candidates
  2) exclude the query observation itself
  3) calculate Euclidean distance over the 14 state variables using
     normalization estimated from the candidate history only
  4) compare FULL vs COMPACT candidate sets
  5) calculate weighted conditional EV for H=5,10,20,40
  6) compare K=50,100,200

Important:
The future return of the query date is NOT used in finding its neighbors.
It is retained only as an out-of-sample reference for later evaluation.
"""

import os, math, json, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
PERIOD = "10y"
HORIZONS = (5, 10, 20, 40)
K_VALUES = (50, 100, 200)
SAMPLE_STEP = 5
MIN_STATE = 260
N_QUERY_PER_TICKER = 8
RANDOM_SEED = 20261009

STATE_COLS = [
    "R1","R5","R20","R60","R120",
    "Vol5","Vol20","Vol60","Vol252",
    "VolRatio20","VolRatio60","Skew20","Kurt20","AC1"
]

# Same universe used by the validated Phase 2A/2B run.
TICKERS = """AA AAPL ABNB ABT ADI AFRM ALAB ALGN AMAT AMD AMGN AMT AMTD AMZN ANET APA APD APO APP ARKF ARKG ARKK ARKQ ARKW ASB ASML ASTS ATI AVB AVGO AX AXON AXP AZN BA BABA BAC BE BEAM BEN BIDU BIIB BITO BKNG BKR BLK BLNK BMY BNTX BOKF BSX BX BXP C CACC CAT CATY CAVA CCI CDNS CEG CELH CF CFG CHPT CHTR CHWY CIFR CL CLF CLS CMCSA COIN COMP COP COST CRDO CRSP CRWD CSCO CSX CUBI CVNA CVX CX DD DDOG DE DELL DHI DIS DJT DKNG DLR DOCU DOW DUOL DVN EA EBAY ECL EDIT EEM EGBN EL ENPH EOG EQIX EQR ETN ETSY EW EXPE EXR FCEL FCF FCX FDX FITB FLEX FLYW FMC FNB FOXA FSLR GD GE GEV GILD GIS GLD GLW GOOGL GPRO GS GSK HAL HAPN HBAN HD HOMB HON HOOD HUT IEFA ILMN INTC IREN ISRG IWM JD JETS JMIA JNJ JPM KEY KKR KLAC KMB KO KRE LCID LEN LI LIN LITE LLY LMT LOT LOW LPRO LRCX LYV MA MAA MARA MCHP MDB MDLZ MDT MELI META MLM MMM MNST MOD MOS MP MRNA MRVL MS MSFT MTB MTCH MU NEM NET NFLX NIO NOC NOW NSC NTLA NTR NTRS NU NUE NVDA NVO NWSA NXPI O OIH OKTA OMF ON OPEN OPFI OXY OZK PAGS PANW PARA PAYS PDD PEP PFE PG PHM PINS PLD PLTR PLUG PSA PSNY PWR PYPL QCOM QQQ QS RDDT REGN RF RIOT RIVN RKLB ROKU RSG RTX RUM RUN SBAC SBUX SCHW SE SEDG SHOP SHW SIFY SLB SMCI SMH SNAP SNOW SNPS SNY SOFI SOXX SPG SPWR STLD STNE STT SYK T TAK TEAM TELFY TEM TER TFC TGT TMDX TMUS TRIP TROW TSLA TSM TTWO TWLO TXN UMBF UNG UNP UPS UPST USO V VEA VICI VLY VMC VNQ VOD VRT VRTX VST VWO VZ W WAY WBD WBS WDAY WFC WM WMT WULF WY XBI XLB XLE XLF XLI XLK XLP XLRE XLU XLV XLY XOM XPEV XYZ Z ZBH ZG ZION ZM ZS""".split()

OUT = Path("cere_phase2c_benchmark")
OUT.mkdir(exist_ok=True)

def clean_series(s):
    return pd.to_numeric(s, errors="coerce").dropna()

def make_dataset(ticker):
    try:
        df = yf.download(
            ticker, period=PERIOD, interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        if df is None or df.empty:
            return None, "NO_DATA"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Open" not in df.columns or "Close" not in df.columns:
            return None, "MISSING_OHLC"

        df = df[["Open","Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated()].sort_index()
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna()

        if len(df) < MIN_STATE + 41:
            return None, "INSUFFICIENT_DATA"

        close = df["Close"].to_numpy(float)
        op = df["Open"].to_numpy(float)
        logret = np.diff(np.log(close))

        rows = []
        dates = df.index

        for i in range(MIN_STATE - 1, len(df) - 40):
            # State uses Close through date i only.
            lr = logret[:i]  # returns ending at i; no future data
            if len(lr) < 252:
                continue

            def ret_n(n):
                return close[i] / close[i-n] - 1.0

            r1, r5, r20, r60, r120 = [ret_n(n) for n in (1,5,20,60,120)]
            vol5 = np.std(lr[-5:], ddof=1) * math.sqrt(252)
            vol20 = np.std(lr[-20:], ddof=1) * math.sqrt(252)
            vol60 = np.std(lr[-60:], ddof=1) * math.sqrt(252)
            vol252 = np.std(lr[-252:], ddof=1) * math.sqrt(252)
            vr20 = vol20 / vol252 if vol252 > 0 else np.nan
            vr60 = vol60 / vol252 if vol252 > 0 else np.nan
            w20 = lr[-20:]
            skew20 = pd.Series(w20).skew()
            kurt20 = pd.Series(w20).kurt()
            w60 = lr[-60:]
            ac1 = pd.Series(w60).autocorr(lag=1)

            vals = [r1,r5,r20,r60,r120,vol5,vol20,vol60,vol252,
                    vr20,vr60,skew20,kurt20,ac1]
            if not np.all(np.isfinite(vals)):
                continue

            row = {"Ticker": ticker, "Date": dates[i]}
            row.update(dict(zip(STATE_COLS, vals)))
            for h in HORIZONS:
                entry = op[i+1]
                exitp = close[i+h]
                row[f"Forward{h}"] = exitp / entry - 1.0
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return None, "NO_VALID_ROWS"
        return out, "OK"
    except Exception as e:
        return None, f"ERROR:{type(e).__name__}"

def weighted_ev(candidates, query, k, h):
    if candidates.empty:
        return np.nan, 0
    x = candidates[STATE_COLS].to_numpy(float)
    q = query[STATE_COLS].to_numpy(float)
    # Candidate-only scaling; query is transformed using candidate statistics.
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    qz = (q - mu) / sd
    d = np.sqrt(np.sum((z - qz) ** 2, axis=1))
    kk = min(k, len(candidates))
    idx = np.argpartition(d, kk-1)[:kk]
    d_k = d[idx]
    # Bandwidth = distance to kth neighbor, with a floor for stability.
    bandwidth = max(float(np.max(d_k)), 1e-9)
    w = np.exp(-(d_k**2) / (bandwidth**2))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(w)
    y = candidates.iloc[idx][f"Forward{h}"].to_numpy(float)
    ev = float(np.sum(w*y) / np.sum(w))
    return ev, int(len(y))

def choose_queries(df):
    if len(df) <= N_QUERY_PER_TICKER:
        return df
    # Deterministic evenly spaced queries, with enough prior history to form K=200.
    eligible = df.iloc[200:].copy()
    if len(eligible) <= N_QUERY_PER_TICKER:
        return eligible
    positions = np.linspace(0, len(eligible)-1, N_QUERY_PER_TICKER).astype(int)
    return eligible.iloc[positions]

def main():
    print("="*80)
    print("CERE_TRAINING — FASE 2C — FULL vs COMPACT BENCHMARK")
    print("="*80)
    print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}")
    print(f"K_VALUES: {K_VALUES} | COMPACT STEP: {SAMPLE_STEP}")
    print(f"Queries/ticker: {N_QUERY_PER_TICKER}")

    all_rows = []
    audit = []
    t0 = time.time()

    for n, ticker in enumerate(TICKERS, 1):
        print(f"[{n:03d}/{len(TICKERS)}] {ticker}")
        full, status = make_dataset(ticker)
        if full is None:
            audit.append({"Ticker":ticker, "STATUS":status, "FULL_ROWS":0})
            continue

        compact = full.iloc[::SAMPLE_STEP].reset_index(drop=True)
        queries = choose_queries(full)

        # Candidate history must be strictly earlier than query date.
        for _, q in queries.iterrows():
            qdate = q["Date"]
            cand_full = full[full["Date"] < qdate].copy()
            cand_comp = compact[compact["Date"] < qdate].copy()
            if len(cand_full) < 200:
                continue

            rec = {
                "Ticker": ticker,
                "QueryDate": qdate,
                "FullCandidates": len(cand_full),
                "CompactCandidates": len(cand_comp),
            }

            for k in K_VALUES:
                for h in HORIZONS:
                    evf, nf = weighted_ev(cand_full, q, k, h)
                    evc, nc = weighted_ev(cand_comp, q, k, h)
                    rec[f"FULL_EV{h}_K{k}"] = evf
                    rec[f"COMPACT_EV{h}_K{k}"] = evc
                    rec[f"ABS_DIFF_EV{h}_K{k}"] = abs(evf-evc) if np.isfinite(evf) and np.isfinite(evc) else np.nan
                    rec[f"QUERY_FORWARD{h}"] = q[f"Forward{h}"]
            all_rows.append(rec)

        audit.append({
            "Ticker": ticker, "STATUS":"OK",
            "FULL_ROWS":len(full), "COMPACT_ROWS":len(compact),
            "QUERIES":len(queries)
        })

    result = pd.DataFrame(all_rows)
    audit_df = pd.DataFrame(audit)

    # Summary: compare FULL vs COMPACT absolute EV difference.
    summary_rows = []
    for k in K_VALUES:
        for h in HORIZONS:
            col = f"ABS_DIFF_EV{h}_K{k}"
            s = pd.to_numeric(result[col], errors="coerce").dropna()
            if len(s):
                summary_rows.append({
                    "K":k, "HORIZON":h, "N_QUERIES":len(s),
                    "MEAN_ABS_DIFF":s.mean(),
                    "MEDIAN_ABS_DIFF":s.median(),
                    "P90_ABS_DIFF":s.quantile(.90),
                    "P95_ABS_DIFF":s.quantile(.95),
                    "MAX_ABS_DIFF":s.max()
                })
    summary = pd.DataFrame(summary_rows)

    result_path = OUT / f"phase2c_queries_{RUN_ID}.csv"
    audit_path = OUT / f"phase2c_ticker_audit_{RUN_ID}.csv"
    summary_path = OUT / f"phase2c_summary_{RUN_ID}.csv"
    result.to_csv(result_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n--- FASE 2C RESUMEN ---")
    print(f"Tickers OK: {sum(audit_df.STATUS == 'OK') if not audit_df.empty else 0}")
    print(f"Query observations: {len(result)}")
    print("\n--- EV ABSOLUTE DIFFERENCE: FULL vs COMPACT ---")
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"\nArchivos: {OUT}/")
    print(f"Duración: {(time.time()-t0)/60:.2f} min")

if __name__ == "__main__":
    main()
