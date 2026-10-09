#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CERE — FASE 3B — ROBUSTEZ, INCERTIDUMBRE Y ESTABILIDAD DEL EV

Laboratorio, NO producción.

Objetivo:
1) reconstruir el mismo dataset FULL y estado canónico de Fase 3A;
2) reproducir exactamente las consultas históricas de Fase 3A;
3) construir la distribución condicional con K=50/100/200;
4) medir incertidumbre del EV mediante bootstrap ponderado;
5) comparar estimadores robustos del EV;
6) medir estabilidad temporal por bloques;
7) producir una tabla de calidad estadística.

NO:
- escribe Google Sheets;
- modifica CERE_CURRENT;
- genera BUY/SELL;
- calcula Kelly;
- usa datos posteriores a la fecha de consulta.

IMPORTANTE:
- La normalización de estados usa SOLO candidatos anteriores a la consulta.
- La consulta no participa en sus propios vecinos.
- Las observaciones futuras NO participan en la selección de vecinos.
"""

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t, trim_mean


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PERIOD = "10y"
INTERVAL = "1d"
MIN_STATE = 260

K_VALUES = (50, 100, 200)
HORIZONS = (5, 10, 20, 40)
QUERIES_PER_TICKER = 8

# Bootstrap deliberadamente moderado para laboratorio.
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42

MIN_BANDWIDTH = 1e-9
BLOCKS = 5

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


def state_from_closes(closes):
    c = np.asarray(closes, dtype=float)
    if len(c) < MIN_STATE or np.any(~np.isfinite(c)) or np.any(c <= 0):
        return None

    lr = np.diff(np.log(c))  # incluye el retorno más reciente

    def ret_n(n):
        return c[-1] / c[-1-n] - 1.0

    def vol_n(n):
        x = lr[-n:]
        if len(x) < n or len(x) < 2:
            return np.nan
        return float(np.std(x, ddof=1) * np.sqrt(252.0))

    def skew_n(n):
        x = lr[-n:]
        return float(pd.Series(x).skew())

    def kurt_n(n):
        x = lr[-n:]
        return float(pd.Series(x).kurt())  # excess kurtosis

    def ac1_n(n):
        x = lr[-n:]
        if len(x) < 3:
            return np.nan
        a, b = x[:-1], x[1:]
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    v20 = vol_n(20)
    v60 = vol_n(60)
    v252 = vol_n(252)

    s = {
        "R1": ret_n(1), "R5": ret_n(5), "R20": ret_n(20),
        "R60": ret_n(60), "R120": ret_n(120),
        "Vol5": vol_n(5), "Vol20": v20, "Vol60": v60,
        "Vol252": v252,
        "VolRatio20": v20 / v252 if np.isfinite(v252) and v252 != 0 else np.nan,
        "VolRatio60": v60 / v252 if np.isfinite(v252) and v252 != 0 else np.nan,
        "Skew20": skew_n(20), "Kurt20": kurt_n(20), "AC1": ac1_n(60)
    }
    if not all(np.isfinite(s[k]) for k in STATE_FIELDS):
        return None
    return s


def build_dataset(ticker):
    try:
        df = yf.download(
            ticker, period=PERIOD, interval=INTERVAL,
            auto_adjust=True, progress=False, threads=False
        )
        if df is None or df.empty:
            return None, "EMPTY"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Open" not in df.columns or "Close" not in df.columns:
            return None, "MISSING_OHLC"

        df = df[["Open", "Close"]].copy()
        df = df.dropna()
        df = df[(df["Open"] > 0) & (df["Close"] > 0)].reset_index()

        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None).dt.normalize()
        df = df[["Date", "Open", "Close"]].drop_duplicates("Date").reset_index(drop=True)

        rows = []
        for i in range(MIN_STATE - 1, len(df) - 40):
            state = state_from_closes(df["Close"].to_numpy()[:i+1])
            if state is None:
                continue
            row = {"Date": df.loc[i, "Date"]}
            row.update(state)
            for h in HORIZONS:
                row[f"Forward{h}"] = (
                    float(df.loc[i+h, "Close"]) / float(df.loc[i+1, "Open"]) - 1.0
                )
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            return None, "NO_ELIGIBLE_ROWS"
        return out, "OK"
    except Exception as e:
        return None, f"ERROR:{type(e).__name__}"


def weighted_neighbors(candidates, query_state, k):
    X = candidates[STATE_FIELDS].to_numpy(dtype=float)
    q = np.array([query_state[f] for f in STATE_FIELDS], dtype=float)

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0

    Xz = (X - mu) / sd
    qz = (q - mu) / sd

    d = np.sqrt(np.sum((Xz - qz) ** 2, axis=1))
    idx = np.argpartition(d, k-1)[:k]
    idx = idx[np.argsort(d[idx])]

    ds = d[idx]
    bandwidth = max(float(ds[-1]), MIN_BANDWIDTH)
    w = np.exp(-(ds ** 2) / (bandwidth ** 2))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(w))
    w = w / w.sum()

    return candidates.iloc[idx].copy(), w


def weighted_median(x, w):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    w = np.asarray(w)[order]
    cs = np.cumsum(w)
    return float(x[np.searchsorted(cs, 0.5, side="left")])


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    x = np.asarray(x)[order]
    w = np.asarray(w)[order]
    cs = np.cumsum(w)
    return float(x[np.searchsorted(cs, q, side="left")])


def weighted_trimmed_mean(x, w, trim=0.10):
    lo = weighted_quantile(x, w, trim)
    hi = weighted_quantile(x, w, 1.0-trim)
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        return float(np.average(x, weights=w))
    ww = w[mask]
    ww = ww / ww.sum()
    return float(np.average(x[mask], weights=ww))


def bootstrap_weighted_mean(x, w, rng, reps):
    """
    Bootstrap ponderado: cada réplica remuestrea K observaciones
    con probabilidad proporcional a los pesos kernel.
    """
    n = len(x)
    # Batch para evitar una matriz gigante.
    batch = 250
    means = np.empty(reps, dtype=float)
    pos = 0
    while pos < reps:
        b = min(batch, reps-pos)
        choices = rng.choice(n, size=(b, n), replace=True, p=w)
        means[pos:pos+b] = np.mean(x[choices], axis=1)
        pos += b
    return means


def effective_sample_size(w):
    return float(1.0 / np.sum(np.square(w)))


def robust_stats(x, w):
    mean = float(np.average(x, weights=w))
    median = weighted_median(x, w)
    trimmed = weighted_trimmed_mean(x, w, 0.10)
    return mean, median, trimmed


def main():
    print("=" * 80)
    print("CERE — FASE 3B — ROBUSTEZ, INCERTIDUMBRE Y ESTABILIDAD DEL EV")
    print("=" * 80)
    print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}")
    print(f"K_VALUES: {K_VALUES} | HORIZONS: {HORIZONS} | Queries/ticker: {QUERIES_PER_TICKER}")
    print(f"BOOTSTRAP_REPS: {BOOTSTRAP_REPS} | BLOCKS: {BLOCKS}")
    print()

    datasets = {}
    audit = []

    for n, ticker in enumerate(TICKERS, 1):
        print(f"[{n:03d}/{len(TICKERS)}] {ticker}")
        ds, status = build_dataset(ticker)
        if ds is None:
            audit.append({"Ticker": ticker, "STATUS": status, "ROWS": 0})
            continue
        datasets[ticker] = ds
        audit.append({"Ticker": ticker, "STATUS": "OK", "ROWS": len(ds)})

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    query_rows = []
    block_rows = []
    ticker_query_counts = []

    for ticker, dataset in datasets.items():
        eligible = np.arange(200, len(dataset))
        if len(eligible) == 0:
            continue

        q_idx = np.linspace(
            eligible[0], eligible[-1],
            num=min(QUERIES_PER_TICKER, len(eligible)),
            dtype=int
        )
        q_idx = np.unique(q_idx)

        raw_count = 0

        for qi in q_idx:
            qrow = dataset.iloc[int(qi)]
            qdate = pd.Timestamp(qrow["Date"])
            candidates = dataset[dataset["Date"] < qdate].copy()

            if len(candidates) < max(K_VALUES):
                continue

            qstate = {f: float(qrow[f]) for f in STATE_FIELDS}
            raw_count += 1

            # 5 temporal blocks, assigned by the query date relative to
            # the ticker's full eligible sample. This measures whether
            # conditional EV behaves similarly through time.
            date_min = dataset["Date"].min()
            date_max = dataset["Date"].max()
            span_days = max((date_max - date_min).days, 1)
            block = int(
                min(
                    BLOCKS - 1,
                    ((qdate - date_min).days * BLOCKS) / span_days
                )
            ) + 1

            for k in K_VALUES:
                neigh, w = weighted_neighbors(candidates, qstate, k)
                ess = effective_sample_size(w)

                for h in HORIZONS:
                    x = neigh[f"Forward{h}"].to_numpy(dtype=float)

                    mean, median, trimmed = robust_stats(x, w)

                    # Student-t is diagnostic only in this phase.
                    try:
                        df_t, loc_t, scale_t = t.fit(x)
                        t_loc = float(loc_t)
                    except Exception:
                        df_t, t_loc, scale_t = np.nan, np.nan, np.nan

                    boot = bootstrap_weighted_mean(x, w, rng, BOOTSTRAP_REPS)
                    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
                    p_ev_pos = float(np.mean(boot > 0.0))

                    query_rows.append({
                        "RUN_ID": RUN_ID,
                        "Ticker": ticker,
                        "QueryDate": qdate.date().isoformat(),
                        "K": k,
                        "HORIZON": h,
                        "EV_MEAN": mean,
                        "EV_MEDIAN": median,
                        "EV_TRIMMED10": trimmed,
                        "EV_STUDENT_T": t_loc,
                        "BOOT_CI95_LOW": float(ci_lo),
                        "BOOT_CI95_HIGH": float(ci_hi),
                        "P_EV_POSITIVE": p_ev_pos,
                        "ESS": ess,
                        "N_NEIGHBORS": len(x),
                        "P_POSITIVE_RET": float(np.average((x > 0).astype(float), weights=w)),
                        "Q10": weighted_quantile(x, w, .10),
                        "Q90": weighted_quantile(x, w, .90),
                        "STD": float(np.sqrt(np.average((x-mean)**2, weights=w))),
                        "BLOCK": block
                    })

        ticker_query_counts.append({
            "Ticker": ticker,
            "QUERY_DATES": raw_count
        })

    qdf = pd.DataFrame(query_rows)

    # Aggregate overall quality.
    summary_rows = []
    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        # EV robust agreement: how far the alternative estimators are
        # from the primary weighted mean.
        robust_disagreement = np.mean([
            np.abs(g["EV_MEAN"] - g["EV_MEDIAN"]),
            np.abs(g["EV_MEAN"] - g["EV_TRIMMED10"]),
            np.abs(g["EV_MEAN"] - g["EV_STUDENT_T"])
        ], axis=0)

        summary_rows.append({
            "RUN_ID": RUN_ID,
            "K": int(k),
            "HORIZON": int(h),
            "N_QUERIES": len(g),
            "MEAN_EV": g["EV_MEAN"].mean(),
            "MEDIAN_EV": g["EV_MEAN"].median(),
            "MEAN_BOOT_CI_LOW": g["BOOT_CI95_LOW"].mean(),
            "MEAN_BOOT_CI_HIGH": g["BOOT_CI95_HIGH"].mean(),
            "MEAN_P_EV_POSITIVE": g["P_EV_POSITIVE"].mean(),
            "PCT_CI_EXCLUDES_ZERO": float(
                np.mean((g["BOOT_CI95_LOW"] > 0) | (g["BOOT_CI95_HIGH"] < 0))
            ),
            "PCT_CI_ENTIRELY_POSITIVE": float(np.mean(g["BOOT_CI95_LOW"] > 0)),
            "MEAN_ESS": g["ESS"].mean(),
            "MEAN_STD": g["STD"].mean(),
            "MEAN_ROBUST_DISAGREEMENT": float(np.mean(robust_disagreement)),
        })

    summary = pd.DataFrame(summary_rows)

    # Temporal stability.
    stability_rows = []
    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        blocks = []
        for b, gb in g.groupby("BLOCK"):
            blocks.append({
                "RUN_ID": RUN_ID,
                "K": int(k),
                "HORIZON": int(h),
                "BLOCK": int(b),
                "N_QUERIES": len(gb),
                "MEAN_EV": gb["EV_MEAN"].mean(),
                "MEDIAN_EV": gb["EV_MEAN"].median(),
                "MEAN_P_EV_POSITIVE": gb["P_EV_POSITIVE"].mean(),
                "PCT_CI_ENTIRELY_POSITIVE": float(np.mean(gb["BOOT_CI95_LOW"] > 0)),
                "MEAN_ESS": gb["ESS"].mean()
            })
            if len(gb):
                blocks.append
        # no-op; rows already appended
        stability_rows.extend([r for r in blocks if isinstance(r, dict)])

    stability = pd.DataFrame(stability_rows)

    outdir = "cere_phase3b_robustness"
    os.makedirs(outdir, exist_ok=True)

    qdf.to_csv(
        os.path.join(outdir, f"phase3b_query_robustness_{RUN_ID}.csv"),
        index=False
    )
    summary.to_csv(
        os.path.join(outdir, f"phase3b_summary_{RUN_ID}.csv"),
        index=False
    )
    stability.to_csv(
        os.path.join(outdir, f"phase3b_stability_{RUN_ID}.csv"),
        index=False
    )
    pd.DataFrame(audit).to_csv(
        os.path.join(outdir, f"phase3b_ticker_audit_{RUN_ID}.csv"),
        index=False
    )
    pd.DataFrame(ticker_query_counts).to_csv(
        os.path.join(outdir, f"phase3b_query_counts_{RUN_ID}.csv"),
        index=False
    )

    print()
    print("--- FASE 3B RESUMEN ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Tickers OK: {len(datasets)}")
    print(f"Query dates: {qdf[['Ticker','QueryDate']].drop_duplicates().shape[0]}")
    print(f"Query executions: {len(qdf)}")
    print()

    print("--- ROBUSTEZ / INCERTIDUMBRE ---")
    cols = [
        "K","HORIZON","N_QUERIES","MEAN_EV",
        "MEAN_BOOT_CI_LOW","MEAN_BOOT_CI_HIGH",
        "MEAN_P_EV_POSITIVE","PCT_CI_ENTIRELY_POSITIVE",
        "MEAN_ESS","MEAN_ROBUST_DISAGREEMENT"
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print()
    print("--- ESTABILIDAD TEMPORAL ---")
    print(
        stability.groupby(["K","HORIZON"])
        .agg(
            BLOCKS_OBSERVED=("BLOCK","nunique"),
            BLOCK_MEAN_EV=("MEAN_EV","mean"),
            BLOCK_EV_STD=("MEAN_EV","std"),
            BLOCKS_POSITIVE=("MEAN_EV", lambda x: int(np.sum(x > 0))),
        )
        .reset_index()
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print()
    print(f"Archivos: {outdir}/")
    print(f"Duration/run ID: {RUN_ID}")
    print("FASE 3B COMPLETADA SIN ESCRITURA A GOOGLE SHEETS.")


if __name__ == "__main__":
    main()
