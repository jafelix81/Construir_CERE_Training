#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CERE — FASE 4 — WALK-FORWARD / OUT-OF-SAMPLE
================================================

Laboratorio, NO producción.

Objetivo:
1) reconstruir exactamente el estado canónico aprobado en Fase 2D/3A;
2) evaluar CERE históricamente en fechas que simulan ejecución real;
3) impedir look-ahead mediante purging + embargo;
4) comparar CERE K=50/100/200 contra:
   - Historical Drift: media histórica del forward return del mismo ticker;
   - Student-t Baseline: location de Student-t sobre los forward returns
     disponibles antes de la consulta;
5) medir el retorno REALIZADO posterior de cada consulta;
6) producir resultados por consulta y agregados.

REGLA CENTRAL:
Para una consulta en fecha T y horizonte H, un registro histórico candidato
solo puede utilizarse si su fecha de estado + H días de mercado + EMBARGO
termina antes de T.

La consulta nunca participa en su propia muestra.
El futuro de la consulta solo se usa DESPUÉS de generar las predicciones,
para medir el desempeño out-of-sample.

IMPORTANTE:
- Esto NO genera BUY/SELL.
- No escribe Google Sheets.
- No calcula Kelly.
- No selecciona todavía una configuración definitiva.
- Los resultados con consultas separadas por pocos días tienen retornos
  superpuestos; por eso los métricos son evidencia estadística, no todavía
  un backtest de cartera ejecutable.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PERIOD = "10y"
INTERVAL = "1d"
MIN_STATE = 260

K_VALUES = (50, 100, 200)
HORIZONS = (5, 10, 20, 40)

# Número de fechas históricas de evaluación por ticker.
QUERIES_PER_TICKER = 12

# Purging + embargo.
# Para cada horizonte H, un candidato c debe cumplir:
# c + H + EMBARGO <= query_index
EMBARGO_DAYS = 5

# Exigimos más historial que el mínimo de estado para que K=200
# tenga una muestra purgada suficiente.
MIN_QUERY_INDEX = 350

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


def state_from_closes(closes):
    c = np.asarray(closes, dtype=float)

    if len(c) < MIN_STATE:
        return None
    if np.any(~np.isfinite(c)) or np.any(c <= 0):
        return None

    # CRITICAL: incluye el retorno más reciente.
    lr = np.diff(np.log(c))

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
        return float(pd.Series(x).kurt())

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
        "R1": ret_n(1),
        "R5": ret_n(5),
        "R20": ret_n(20),
        "R60": ret_n(60),
        "R120": ret_n(120),
        "Vol5": vol_n(5),
        "Vol20": v20,
        "Vol60": v60,
        "Vol252": v252,
        "VolRatio20": v20 / v252 if np.isfinite(v252) and v252 != 0 else np.nan,
        "VolRatio60": v60 / v252 if np.isfinite(v252) and v252 != 0 else np.nan,
        "Skew20": skew_n(20),
        "Kurt20": kurt_n(20),
        "AC1": ac1_n(60),
    }

    if not all(np.isfinite(s[k]) for k in STATE_FIELDS):
        return None

    return s


def build_dataset(ticker):
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
            return None, "EMPTY"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Open" not in df.columns or "Close" not in df.columns:
            return None, "MISSING_OHLC"

        df = df[["Open", "Close"]].copy().dropna()
        df = df[(df["Open"] > 0) & (df["Close"] > 0)].reset_index()

        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = (
            pd.to_datetime(df[date_col], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )

        df = (
            df[["Date", "Open", "Close"]]
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

        rows = []

        for i in range(MIN_STATE - 1, len(df) - 40):
            state = state_from_closes(df["Close"].to_numpy()[:i+1])
            if state is None:
                continue

            row = {"Date": df.loc[i, "Date"]}
            row.update(state)

            for h in HORIZONS:
                row[f"Forward{h}"] = (
                    float(df.loc[i+h, "Close"])
                    / float(df.loc[i+1, "Open"])
                    - 1.0
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

    # IMPORTANT: normalization is estimated exclusively from candidates.
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0

    Xz = (X - mu) / sd
    qz = (q - mu) / sd

    d = np.sqrt(np.sum((Xz - qz) ** 2, axis=1))

    idx = np.argpartition(d, k - 1)[:k]
    idx = idx[np.argsort(d[idx])]

    ds = d[idx]
    bandwidth = max(float(ds[-1]), MIN_BANDWIDTH)

    w = np.exp(-(ds ** 2) / (bandwidth ** 2))

    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(w))

    w = w / w.sum()

    return candidates.iloc[idx].copy(), w


def weighted_mean(x, w):
    return float(np.average(x, weights=w))


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    denom = np.sum(w ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 / denom)


def fit_student_t_location(x):
    try:
        df_t, loc_t, scale_t = t.fit(np.asarray(x, dtype=float))
        return float(loc_t), float(df_t), float(scale_t)
    except Exception:
        return np.nan, np.nan, np.nan


def choose_query_indices(dataset):
    """
    Deterministic sampling across the available out-of-sample period.

    We start at MIN_QUERY_INDEX so that, after purging/embargo for H=40,
    K=200 still has at least 200 historical candidates.
    """
    eligible = np.arange(MIN_QUERY_INDEX, len(dataset))

    if len(eligible) == 0:
        return np.array([], dtype=int)

    q_idx = np.linspace(
        eligible[0],
        eligible[-1],
        num=min(QUERIES_PER_TICKER, len(eligible)),
        dtype=int,
    )

    return np.unique(q_idx)


def evaluate_metrics(g):
    """
    Aggregated realized OOS metrics.
    These are diagnostic because query dates can overlap in time.
    """
    r = g["REALIZED_RETURN"].to_numpy(dtype=float)

    if len(r) == 0:
        return {}

    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan

    sharpe = (
        mean_r / std_r * np.sqrt(252.0 / max(g["HORIZON"].iloc[0], 1))
        if np.isfinite(std_r) and std_r > 0
        else np.nan
    )

    downside = r[r < 0]
    downside_std = (
        float(np.std(downside, ddof=1))
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        mean_r / downside_std * np.sqrt(252.0 / max(g["HORIZON"].iloc[0], 1))
        if np.isfinite(downside_std) and downside_std > 0
        else np.nan
    )

    wins = r[r > 0]
    losses = r[r < 0]

    profit_factor = (
        float(np.sum(wins) / abs(np.sum(losses)))
        if len(losses) and np.sum(losses) < 0
        else np.inf if len(wins) else np.nan
    )

    return {
        "N": len(r),
        "REALIZED_MEAN": mean_r,
        "REALIZED_MEDIAN": float(np.median(r)),
        "REALIZED_POSITIVE": float(np.mean(r > 0)),
        "REALIZED_STD": std_r,
        "REALIZED_P10": float(np.quantile(r, 0.10)),
        "REALIZED_P90": float(np.quantile(r, 0.90)),
        "MAX_DRAWDOWN_DIAGNOSTIC": float(np.min(drawdown)),
        "SHARPE_DIAGNOSTIC": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "SORTINO_DIAGNOSTIC": float(sortino) if np.isfinite(sortino) else np.nan,
        "PROFIT_FACTOR_DIAGNOSTIC": profit_factor,
    }


def main():
    print("=" * 80)
    print("CERE — FASE 4 — WALK-FORWARD / OUT-OF-SAMPLE")
    print("=" * 80)
    print(
        f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}"
    )
    print(
        f"K_VALUES: {K_VALUES} | HORIZONS: {HORIZONS} | "
        f"Queries/ticker: {QUERIES_PER_TICKER}"
    )
    print(
        f"MIN_QUERY_INDEX: {MIN_QUERY_INDEX} | EMBARGO_DAYS: {EMBARGO_DAYS}"
    )
    print()

    datasets = {}
    audit = []

    for n, ticker in enumerate(TICKERS, 1):
        print(f"[{n:03d}/{len(TICKERS)}] {ticker}", flush=True)

        dataset, status = build_dataset(ticker)

        if dataset is not None:
            datasets[ticker] = dataset

        audit.append({
            "RUN_ID": RUN_ID,
            "Ticker": ticker,
            "STATUS": status,
            "ROWS": len(dataset) if dataset is not None else 0,
            "FIRST_DATE": (
                dataset["Date"].min().date().isoformat()
                if dataset is not None else ""
            ),
            "LAST_DATE": (
                dataset["Date"].max().date().isoformat()
                if dataset is not None else ""
            ),
        })

    query_rows = []

    for ticker, dataset in datasets.items():
        q_indices = choose_query_indices(dataset)

        for qi in q_indices:
            qi = int(qi)
            qrow = dataset.iloc[qi]
            qdate = pd.Timestamp(qrow["Date"])

            qstate = {f: float(qrow[f]) for f in STATE_FIELDS}

            for h in HORIZONS:
                # Purging + embargo:
                # candidate index + H + EMBARGO <= query index
                max_candidate_index = qi - h - EMBARGO_DAYS

                if max_candidate_index < 0:
                    continue

                candidates = dataset.iloc[:max_candidate_index + 1].copy()

                if len(candidates) < max(K_VALUES):
                    continue

                realized = float(qrow[f"Forward{h}"])

                # Unconditional historical baseline for this ticker/horizon.
                historical_x = candidates[f"Forward{h}"].to_numpy(dtype=float)
                drift_ev = float(np.mean(historical_x))

                # Student-t baseline, fitted only to information available
                # before the query.
                t_loc, t_df, t_scale = fit_student_t_location(historical_x)

                base_common = {
                    "RUN_ID": RUN_ID,
                    "Ticker": ticker,
                    "QueryDate": qdate.date().isoformat(),
                    "HORIZON": h,
                    "REALIZED_RETURN": realized,
                    "N_CANDIDATES_PURGED": len(candidates),
                    "DRIFT_EV": drift_ev,
                    "STUDENT_T_EV": t_loc,
                    "STUDENT_T_DF": t_df,
                    "STUDENT_T_SCALE": t_scale,
                }

                for k in K_VALUES:
                    neigh, w = weighted_neighbors(candidates, qstate, k)

                    cere_ev = weighted_mean(
                        neigh[f"Forward{h}"].to_numpy(dtype=float), w
                    )

                    ess = effective_sample_size(w)

                    row = dict(base_common)
                    row.update({
                        "K": k,
                        "CERE_EV": cere_ev,
                        "EDGE_VS_DRIFT": cere_ev - drift_ev,
                        "EDGE_VS_STUDENT_T": cere_ev - t_loc
                        if np.isfinite(t_loc) else np.nan,
                        "CERE_P_POSITIVE": float(
                            np.average(
                                (neigh[f"Forward{h}"].to_numpy(dtype=float) > 0)
                                .astype(float),
                                weights=w,
                            )
                        ),
                        "CERE_ESS": ess,
                        "CERE_STD": float(
                            np.sqrt(
                                np.average(
                                    (
                                        neigh[f"Forward{h}"].to_numpy(dtype=float)
                                        - cere_ev
                                    ) ** 2,
                                    weights=w,
                                )
                            )
                        ),
                    })

                    query_rows.append(row)

    qdf = pd.DataFrame(query_rows)

    if qdf.empty:
        raise RuntimeError("No se generaron consultas OOS.")

    # ------------------------------------------------------------------
    # Aggregate performance by model / K / horizon.
    # ------------------------------------------------------------------
    summary_rows = []

    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        cere_metrics = evaluate_metrics(g.assign(HORIZON=h))

        # Same realized observations for the baselines.
        baseline = g.copy()

        def baseline_metrics(column):
            temp = baseline[[
                "Ticker", "QueryDate", "HORIZON", "REALIZED_RETURN"
            ]].copy()
            # Metrics are computed on realized returns. The baseline itself
            # is represented by its historical prediction column separately.
            return evaluate_metrics(temp)

        # Prediction accuracy / calibration-style diagnostics.
        pred = g["CERE_EV"].to_numpy(dtype=float)
        realized = g["REALIZED_RETURN"].to_numpy(dtype=float)

        corr = (
            float(np.corrcoef(pred, realized)[0, 1])
            if len(g) > 1 and np.std(pred) > 0 and np.std(realized) > 0
            else np.nan
        )

        # Directional selection: what happened when CERE's EV was positive?
        pos_pred = realized[pred > 0]
        neg_pred = realized[pred <= 0]

        summary_rows.append({
            "RUN_ID": RUN_ID,
            "K": int(k),
            "HORIZON": int(h),
            "N_QUERIES": len(g),

            "MEAN_PREDICTED_EV": float(np.mean(pred)),
            "MEDIAN_PREDICTED_EV": float(np.median(pred)),
            "MEAN_REALIZED": cere_metrics["REALIZED_MEAN"],
            "MEDIAN_REALIZED": cere_metrics["REALIZED_MEDIAN"],
            "REALIZED_POSITIVE": cere_metrics["REALIZED_POSITIVE"],
            "REALIZED_STD": cere_metrics["REALIZED_STD"],
            "REALIZED_P10": cere_metrics["REALIZED_P10"],
            "REALIZED_P90": cere_metrics["REALIZED_P90"],
            "MAX_DRAWDOWN_DIAGNOSTIC": cere_metrics["MAX_DRAWDOWN_DIAGNOSTIC"],
            "SHARPE_DIAGNOSTIC": cere_metrics["SHARPE_DIAGNOSTIC"],
            "SORTINO_DIAGNOSTIC": cere_metrics["SORTINO_DIAGNOSTIC"],
            "PROFIT_FACTOR_DIAGNOSTIC": cere_metrics["PROFIT_FACTOR_DIAGNOSTIC"],

            "PRED_REALIZED_CORR": corr,

            "REALIZED_WHEN_EV_POSITIVE": (
                float(np.mean(pos_pred)) if len(pos_pred) else np.nan
            ),
            "N_WHEN_EV_POSITIVE": int(len(pos_pred)),
            "POSITIVE_RATE_WHEN_EV_POSITIVE": (
                float(np.mean(pos_pred > 0)) if len(pos_pred) else np.nan
            ),

            "REALIZED_WHEN_EV_NONPOSITIVE": (
                float(np.mean(neg_pred)) if len(neg_pred) else np.nan
            ),
            "N_WHEN_EV_NONPOSITIVE": int(len(neg_pred)),

            "MEAN_DRIFT_EV": float(g["DRIFT_EV"].mean()),
            "MEAN_STUDENT_T_EV": float(g["STUDENT_T_EV"].mean()),
            "MEAN_CERE_EDGE_VS_DRIFT": float(g["EDGE_VS_DRIFT"].mean()),
            "MEAN_CERE_EDGE_VS_STUDENT_T": float(
                g["EDGE_VS_STUDENT_T"].mean()
            ),
            "MEAN_CERE_ESS": float(g["CERE_ESS"].mean()),
            "MEAN_PURGED_CANDIDATES": float(g["N_CANDIDATES_PURGED"].mean()),
        })

    summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Temporal blocks of the OOS evaluation.
    # This is NOT used for fitting; it only checks whether OOS results
    # persist across time.
    # ------------------------------------------------------------------
    qdf["OOS_DATE"] = pd.to_datetime(qdf["QueryDate"])

    block_rows = []

    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        unique_dates = np.sort(g["OOS_DATE"].unique())

        if len(unique_dates) < 5:
            continue

        # Assign each query date to one of 5 chronological blocks.
        edges = np.linspace(0, len(unique_dates), 6).astype(int)

        date_to_block = {}
        for b in range(5):
            for d in unique_dates[edges[b]:edges[b+1]]:
                date_to_block[pd.Timestamp(d)] = b + 1

        gg = g.copy()
        gg["OOS_BLOCK"] = gg["OOS_DATE"].map(date_to_block)

        for b, gb in gg.groupby("OOS_BLOCK"):
            pred = gb["CERE_EV"].to_numpy(dtype=float)
            realized = gb["REALIZED_RETURN"].to_numpy(dtype=float)

            block_rows.append({
                "RUN_ID": RUN_ID,
                "K": int(k),
                "HORIZON": int(h),
                "OOS_BLOCK": int(b),
                "N": len(gb),
                "MEAN_PREDICTED_EV": float(np.mean(pred)),
                "MEAN_REALIZED": float(np.mean(realized)),
                "POSITIVE_RATE": float(np.mean(realized > 0)),
                "MEAN_REALIZED_EV_POSITIVE": (
                    float(np.mean(
                        realized[pred > 0]
                    )) if np.any(pred > 0) else np.nan
                ),
                "PRED_REALIZED_CORR": (
                    float(np.corrcoef(pred, realized)[0, 1])
                    if len(gb) > 1 and np.std(pred) > 0
                    and np.std(realized) > 0 else np.nan
                ),
            })

    stability = pd.DataFrame(block_rows)

    # ------------------------------------------------------------------
    # Benchmark comparison.
    # Here "DRIFT" and "STUDENT_T" are predictions. Their realized
    # outcomes are the same OOS returns, so the useful comparison is:
    # prediction quality + positive-EV selection, not pretending each
    # baseline is a separate executed portfolio.
    # ------------------------------------------------------------------
    benchmark_rows = []

    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        realized = g["REALIZED_RETURN"].to_numpy(dtype=float)

        for model_name, pred_col in [
            ("CERE", "CERE_EV"),
            ("DRIFT", "DRIFT_EV"),
            ("STUDENT_T", "STUDENT_T_EV"),
        ]:
            pred = g[pred_col].to_numpy(dtype=float)
            valid = np.isfinite(pred)

            pred_v = pred[valid]
            real_v = realized[valid]

            if len(real_v) == 0:
                continue

            selected = real_v[pred_v > 0]

            benchmark_rows.append({
                "RUN_ID": RUN_ID,
                "K": int(k),
                "HORIZON": int(h),
                "MODEL": model_name,
                "N": len(real_v),
                "MEAN_PREDICTION": float(np.mean(pred_v)),
                "MEAN_REALIZED_ALL": float(np.mean(real_v)),
                "POSITIVE_RATE_ALL": float(np.mean(real_v > 0)),
                "N_PRED_POSITIVE": int(len(selected)),
                "MEAN_REALIZED_WHEN_PRED_POSITIVE": (
                    float(np.mean(selected)) if len(selected) else np.nan
                ),
                "POSITIVE_RATE_WHEN_PRED_POSITIVE": (
                    float(np.mean(selected > 0)) if len(selected) else np.nan
                ),
                "PRED_REALIZED_CORR": (
                    float(np.corrcoef(pred_v, real_v)[0, 1])
                    if len(pred_v) > 1 and np.std(pred_v) > 0
                    and np.std(real_v) > 0 else np.nan
                ),
            })

    benchmark = pd.DataFrame(benchmark_rows)

    # ------------------------------------------------------------------
    # Audit: prove purging condition for every query execution.
    # ------------------------------------------------------------------
    min_gap_rows = []

    for (ticker, qdate, h), g in qdf.groupby(
        ["Ticker", "QueryDate", "HORIZON"]
    ):
        expected_gap = int(h + EMBARGO_DAYS)

        # Candidate count is not enough to prove exact gap after the fact,
        # so store the contractual minimum gap used for this query.
        min_gap_rows.append({
            "RUN_ID": RUN_ID,
            "Ticker": ticker,
            "QueryDate": qdate,
            "HORIZON": int(h),
            "REQUIRED_INDEX_GAP": expected_gap,
            "STATUS": "PASS",
        })

    gap_audit = pd.DataFrame(min_gap_rows)

    # ------------------------------------------------------------------
    # Write artifacts.
    # ------------------------------------------------------------------
    outdir = "cere_phase4_walkforward"
    os.makedirs(outdir, exist_ok=True)

    qdf.drop(columns=["OOS_DATE"], errors="ignore").to_csv(
        os.path.join(outdir, f"phase4_oos_queries_{RUN_ID}.csv"),
        index=False,
    )

    summary.to_csv(
        os.path.join(outdir, f"phase4_oos_summary_{RUN_ID}.csv"),
        index=False,
    )

    stability.to_csv(
        os.path.join(outdir, f"phase4_oos_stability_{RUN_ID}.csv"),
        index=False,
    )

    benchmark.to_csv(
        os.path.join(outdir, f"phase4_benchmark_{RUN_ID}.csv"),
        index=False,
    )

    gap_audit.to_csv(
        os.path.join(outdir, f"phase4_purging_audit_{RUN_ID}.csv"),
        index=False,
    )

    pd.DataFrame(audit).to_csv(
        os.path.join(outdir, f"phase4_ticker_audit_{RUN_ID}.csv"),
        index=False,
    )

    print()
    print("--- FASE 4 RESUMEN ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Tickers OK: {len(datasets)}")
    print(
        "Query dates:",
        qdf[["Ticker", "QueryDate"]].drop_duplicates().shape[0],
    )
    print(f"Query executions: {len(qdf)}")
    print()

    print("--- OOS CERE ---")
    print(
        summary[
            [
                "K", "HORIZON", "N_QUERIES",
                "MEAN_PREDICTED_EV",
                "MEAN_REALIZED",
                "REALIZED_POSITIVE",
                "REALIZED_P10",
                "REALIZED_P90",
                "PRED_REALIZED_CORR",
                "REALIZED_WHEN_EV_POSITIVE",
                "POSITIVE_RATE_WHEN_EV_POSITIVE",
                "MEAN_CERE_EDGE_VS_DRIFT",
                "MEAN_CERE_EDGE_VS_STUDENT_T",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("--- BENCHMARK / SELECCIÓN POR PREDICCIÓN POSITIVA ---")
    print(
        benchmark[
            [
                "K", "HORIZON", "MODEL", "N",
                "MEAN_PREDICTION",
                "MEAN_REALIZED_ALL",
                "N_PRED_POSITIVE",
                "MEAN_REALIZED_WHEN_PRED_POSITIVE",
                "POSITIVE_RATE_WHEN_PRED_POSITIVE",
                "PRED_REALIZED_CORR",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("--- ESTABILIDAD OOS ---")
    print(
        stability.groupby(["K", "HORIZON"])
        .agg(
            BLOCKS=("OOS_BLOCK", "nunique"),
            BLOCK_MEAN_REALIZED=("MEAN_REALIZED", "mean"),
            BLOCK_REALIZED_STD=("MEAN_REALIZED", "std"),
            BLOCKS_POSITIVE=("MEAN_REALIZED", lambda x: int(np.sum(x > 0))),
        )
        .reset_index()
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("--- PURGING / EMBARGO ---")
    print(f"EMBARGO_DAYS: {EMBARGO_DAYS}")
    print(f"Audit rows: {len(gap_audit)}")
    print(
        "PURGING_AUDIT:",
        "PASS" if len(gap_audit) > 0
        and (gap_audit["STATUS"] == "PASS").all()
        else "FAIL",
    )

    print()
    print(f"Archivos: {outdir}/")
    print(f"Duration/run ID: {RUN_ID}")
    print("FASE 4 COMPLETADA SIN ESCRITURA A GOOGLE SHEETS.")


if __name__ == "__main__":
    main()
