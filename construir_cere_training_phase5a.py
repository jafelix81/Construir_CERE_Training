#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CERE — FASE 5A — CROSS-SECTIONAL WALK-FORWARD OOS
=================================================

Laboratorio, NO producción.

Objetivo:
1) mantener exactamente el estado canónico de Fase 2D/3A/4;
2) cambiar únicamente la fuente de análogos:
   - Fase 4: historial del MISMO ticker;
   - Fase 5A: historial de TODO el universo;
3) evaluar si el estado estadístico aporta información para ordenar
   oportunidades por encima de Historical Drift;
4) medir Rank IC / Spearman y spreads cross-sectional;
5) mantener purging + embargo y evaluación estrictamente OOS.

IMPORTANTE:
- No escribe Google Sheets.
- No calcula Kelly.
- No genera BUY/SELL.
- No selecciona todavía configuración definitiva.
- No usa información futura para formar predicciones.

Diseño cross-sectional:
Para cada fecha de consulta T y horizonte H:
- se toman los estados disponibles de todos los tickers en T;
- los candidatos históricos pertenecen a TODOS los tickers;
- un candidato C solo puede entrar si su fecha de estado es anterior
  a T por al menos H + EMBARGO días de calendario hábiles aproximados.
- la normalización z-score se estima exclusivamente sobre candidatos.
- se construye un KDTree con los candidatos y se consultan todos los
  tickers de la fecha T de una sola vez.

Para evitar que la consulta use una observación demasiado cercana en
tiempo, el corte histórico se hace por fecha usando pandas BusinessDay.
Es deliberadamente conservador: puede descartar algunos candidatos
válidos, pero nunca incorpora datos posteriores al embargo.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t, spearmanr
from scipy.spatial import cKDTree


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PERIOD = "10y"
INTERVAL = "1d"
MIN_STATE = 260

K_VALUES = (50, 100, 200)
HORIZONS = (5, 10, 20, 40)

# Fechas globales de evaluación.
# Cada fecha representa un "corte de mercado" cross-sectional.
QUERY_DATES = 12

EMBARGO_DAYS = 5
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
        return float(pd.Series(lr[-n:]).skew())

    def kurt_n(n):
        return float(pd.Series(lr[-n:]).kurt())

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

            row = {
                "Date": df.loc[i, "Date"],
                "INDEX": i,
            }
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


def choose_global_query_dates(datasets):
    """
    Uses QQQ as the reference trading calendar. This creates 12 global
    market dates, rather than 12 dates independently per ticker.
    """
    ref = datasets.get("QQQ")
    if ref is None:
        ref = next(iter(datasets.values()))

    eligible = ref.loc[ref["INDEX"] >= MIN_QUERY_INDEX, "Date"]
    eligible = pd.Series(pd.to_datetime(eligible).unique()).sort_values()

    if len(eligible) == 0:
        return []

    idx = np.linspace(
        0,
        len(eligible) - 1,
        num=min(QUERY_DATES, len(eligible)),
        dtype=int,
    )

    return [pd.Timestamp(eligible.iloc[i]) for i in np.unique(idx)]


def eligible_candidates(all_rows, query_date, horizon):
    """
    Conservative calendar purging:
    candidate_date <= query_date - (H + EMBARGO) business days.

    This is intentionally stricter than a same-ticker index test.
    """
    cutoff = query_date - pd.offsets.BDay(horizon + EMBARGO_DAYS)
    return all_rows.loc[all_rows["Date"] <= cutoff].copy()


def make_tree(candidates):
    X = candidates[STATE_FIELDS].to_numpy(dtype=float)

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0

    Xz = (X - mu) / sd

    tree = cKDTree(Xz)
    return tree, mu, sd


def query_neighbors(tree, mu, sd, query_states, k):
    q = np.asarray(query_states, dtype=float)
    qz = (q - mu) / sd

    distances, indices = tree.query(
        qz,
        k=k,
        workers=-1,
    )

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    return distances, indices


def weighted_stats(values, distances):
    """
    Gaussian-like kernel used in earlier CERE phases.
    Bandwidth = distance to kth neighbor.
    """
    ds = np.asarray(distances, dtype=float)
    x = np.asarray(values, dtype=float)

    bandwidth = max(float(ds[-1]), MIN_BANDWIDTH)
    w = np.exp(-(ds ** 2) / (bandwidth ** 2))

    if not np.isfinite(w).all() or np.sum(w) <= 0:
        w = np.ones(len(w))

    w = w / np.sum(w)

    ev = float(np.average(x, weights=w))
    ess = float(1.0 / np.sum(w ** 2))

    var = float(np.average((x - ev) ** 2, weights=w))

    return ev, ess, float(np.sqrt(var)), w


def fit_student_t_location(x):
    try:
        df_t, loc_t, scale_t = t.fit(np.asarray(x, dtype=float))
        return float(loc_t), float(df_t), float(scale_t)
    except Exception:
        return np.nan, np.nan, np.nan


def cross_sectional_metrics(g):
    """
    Computes metrics for one K/H configuration.

    Rank IC is calculated date-by-date across tickers and then averaged.
    This is the central metric of Phase 5A.
    """
    ic_values = []
    top_bottom_spreads = []
    positive_ic = 0

    for date, gd in g.groupby("QueryDate"):
        gd = gd.dropna(subset=["CERE_EV", "REALIZED_RETURN"])

        if len(gd) < 10:
            continue

        rho, _ = spearmanr(
            gd["CERE_EV"].to_numpy(float),
            gd["REALIZED_RETURN"].to_numpy(float),
        )

        if np.isfinite(rho):
            ic_values.append(float(rho))
            positive_ic += int(rho > 0)

        n = len(gd)
        q = max(1, n // 5)

        ranked = gd.sort_values("CERE_EV")
        bottom = ranked.iloc[:q]["REALIZED_RETURN"].mean()
        top = ranked.iloc[-q:]["REALIZED_RETURN"].mean()

        top_bottom_spreads.append(float(top - bottom))

    return {
        "N_CROSS_DATES": int(len(ic_values)),
        "MEAN_RANK_IC": float(np.mean(ic_values)) if ic_values else np.nan,
        "MEDIAN_RANK_IC": float(np.median(ic_values)) if ic_values else np.nan,
        "RANK_IC_STD": float(np.std(ic_values, ddof=1))
        if len(ic_values) > 1 else np.nan,
        "POSITIVE_IC_RATE": float(np.mean(np.asarray(ic_values) > 0))
        if ic_values else np.nan,
        "MEAN_TOP_MINUS_BOTTOM": float(np.mean(top_bottom_spreads))
        if top_bottom_spreads else np.nan,
        "MEDIAN_TOP_MINUS_BOTTOM": float(np.median(top_bottom_spreads))
        if top_bottom_spreads else np.nan,
    }


def main():
    print("=" * 80)
    print("CERE — FASE 5A — CROSS-SECTIONAL WALK-FORWARD OOS")
    print("=" * 80)
    print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}")
    print(
        f"K_VALUES: {K_VALUES} | HORIZONS: {HORIZONS} | "
        f"Global query dates: {QUERY_DATES}"
    )
    print(f"MIN_QUERY_INDEX: {MIN_QUERY_INDEX} | EMBARGO_DAYS: {EMBARGO_DAYS}")
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

    if not datasets:
        raise RuntimeError("No se pudo construir ningún dataset.")

    query_dates = choose_global_query_dates(datasets)

    print()
    print("GLOBAL QUERY DATES:")
    for d in query_dates:
        print(" ", d.date().isoformat())
    print()

    # One combined historical table.
    all_rows = []
    for ticker, df in datasets.items():
        x = df.copy()
        x["Ticker"] = ticker
        all_rows.append(x)

    all_rows = pd.concat(all_rows, ignore_index=True)
    all_rows["Date"] = pd.to_datetime(all_rows["Date"])

    query_rows = []
    purging_audit = []

    for date_no, qdate in enumerate(query_dates, 1):
        print(
            f"QUERY DATE [{date_no:02d}/{len(query_dates)}] "
            f"{qdate.date().isoformat()}",
            flush=True,
        )

        # Current cross-section: all tickers with an exact observation date.
        current = all_rows.loc[all_rows["Date"] == qdate].copy()

        if current.empty:
            print("  sin observaciones; se omite.", flush=True)
            continue

        for h in HORIZONS:
            candidates = eligible_candidates(all_rows, qdate, h)

            if len(candidates) < max(K_VALUES):
                print(
                    f"  H={h}: candidatos insuficientes ({len(candidates)})",
                    flush=True,
                )
                continue

            # Candidate normalization is strictly historical.
            tree, mu, sd = make_tree(candidates)

            qstates = current[STATE_FIELDS].to_numpy(dtype=float)

            for k in K_VALUES:
                distances, indices = query_neighbors(
                    tree, mu, sd, qstates, k
                )

                for row_pos, (_, qrow) in enumerate(current.iterrows()):
                    ticker = qrow["Ticker"]

                    # Future return of the query is used ONLY now, after
                    # the prediction has been generated.
                    realized = float(qrow[f"Forward{h}"])

                    # Same-ticker historical baseline, using only purged
                    # candidates for this query.
                    ticker_candidates = candidates.loc[
                        candidates["Ticker"] == ticker
                    ]

                    if len(ticker_candidates) < 20:
                        drift_ev = np.nan
                        student_ev = np.nan
                        student_df = np.nan
                        student_scale = np.nan
                    else:
                        hx = ticker_candidates[
                            f"Forward{h}"
                        ].to_numpy(dtype=float)

                        drift_ev = float(np.mean(hx))
                        student_ev, student_df, student_scale = (
                            fit_student_t_location(hx)
                        )

                    neigh_idx = indices[row_pos]
                    neigh_dist = distances[row_pos]

                    neigh = candidates.iloc[neigh_idx]
                    values = neigh[f"Forward{h}"].to_numpy(dtype=float)

                    cere_ev, ess, cere_std, weights = weighted_stats(
                        values,
                        neigh_dist,
                    )

                    p_positive = float(
                        np.average(
                            (values > 0).astype(float),
                            weights=weights,
                        )
                    )

                    row = {
                        "RUN_ID": RUN_ID,
                        "QueryDate": qdate.date().isoformat(),
                        "Ticker": ticker,
                        "HORIZON": int(h),
                        "K": int(k),
                        "REALIZED_RETURN": realized,

                        "CERE_CROSS_EV": cere_ev,
                        "CERE_CROSS_P_POSITIVE": p_positive,
                        "CERE_CROSS_ESS": ess,
                        "CERE_CROSS_STD": cere_std,

                        "DRIFT_EV": drift_ev,
                        "STUDENT_T_EV": student_ev,
                        "STUDENT_T_DF": student_df,
                        "STUDENT_T_SCALE": student_scale,

                        "EDGE_CROSS_VS_DRIFT": (
                            cere_ev - drift_ev
                            if np.isfinite(drift_ev) else np.nan
                        ),
                        "EDGE_CROSS_VS_STUDENT_T": (
                            cere_ev - student_ev
                            if np.isfinite(student_ev) else np.nan
                        ),

                        "N_GLOBAL_CANDIDATES": len(candidates),
                    }

                    query_rows.append(row)

            # Audit once per date/horizon.
            cutoff = qdate - pd.offsets.BDay(h + EMBARGO_DAYS)
            max_candidate_date = (
                candidates["Date"].max() if len(candidates) else pd.NaT
            )

            purging_audit.append({
                "RUN_ID": RUN_ID,
                "QueryDate": qdate.date().isoformat(),
                "HORIZON": int(h),
                "EMBARGO_DAYS": EMBARGO_DAYS,
                "CUTOFF_DATE": cutoff.date().isoformat(),
                "MAX_CANDIDATE_DATE": (
                    max_candidate_date.date().isoformat()
                    if pd.notna(max_candidate_date) else ""
                ),
                "N_GLOBAL_CANDIDATES": len(candidates),
                "STATUS": (
                    "PASS"
                    if pd.notna(max_candidate_date)
                    and max_candidate_date <= cutoff
                    else "FAIL"
                ),
            })

    qdf = pd.DataFrame(query_rows)

    if qdf.empty:
        raise RuntimeError("No se generaron observaciones cross-sectional.")

    # ---------------------------------------------------------------
    # Aggregate summary.
    # ---------------------------------------------------------------
    summary_rows = []

    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        pred = g["CERE_CROSS_EV"].to_numpy(float)
        realized = g["REALIZED_RETURN"].to_numpy(float)

        valid = np.isfinite(pred) & np.isfinite(realized)
        pred = pred[valid]
        realized = realized[valid]

        pos = realized[pred > 0]

        metrics = cross_sectional_metrics(
            g.loc[valid].copy()
        )

        summary_rows.append({
            "RUN_ID": RUN_ID,
            "K": int(k),
            "HORIZON": int(h),
            "N_OBSERVATIONS": len(g),
            "N_TICKERS_MEAN": float(g.groupby("QueryDate")["Ticker"].nunique().mean()),

            "MEAN_PREDICTED_EV": float(np.mean(pred)),
            "MEAN_REALIZED": float(np.mean(realized)),
            "REALIZED_POSITIVE": float(np.mean(realized > 0)),

            "MEAN_REALIZED_WHEN_CROSS_EV_POSITIVE": (
                float(np.mean(pos)) if len(pos) else np.nan
            ),
            "POSITIVE_RATE_WHEN_CROSS_EV_POSITIVE": (
                float(np.mean(pos > 0)) if len(pos) else np.nan
            ),

            "MEAN_EDGE_VS_DRIFT": float(
                g["EDGE_CROSS_VS_DRIFT"].mean()
            ),
            "MEAN_EDGE_VS_STUDENT_T": float(
                g["EDGE_CROSS_VS_STUDENT_T"].mean()
            ),

            "MEAN_CROSS_ESS": float(g["CERE_CROSS_ESS"].mean()),
            "MEAN_GLOBAL_CANDIDATES": float(
                g["N_GLOBAL_CANDIDATES"].mean()
            ),

            "MEAN_RANK_IC": metrics["MEAN_RANK_IC"],
            "MEDIAN_RANK_IC": metrics["MEDIAN_RANK_IC"],
            "RANK_IC_STD": metrics["RANK_IC_STD"],
            "POSITIVE_IC_RATE": metrics["POSITIVE_IC_RATE"],
            "MEAN_TOP_MINUS_BOTTOM": metrics["MEAN_TOP_MINUS_BOTTOM"],
            "MEDIAN_TOP_MINUS_BOTTOM": metrics["MEDIAN_TOP_MINUS_BOTTOM"],
        })

    summary = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------------
    # Direct benchmark by positive prediction.
    # ---------------------------------------------------------------
    benchmark_rows = []

    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        for model_name, pred_col in [
            ("CERE_CROSS", "CERE_CROSS_EV"),
            ("DRIFT", "DRIFT_EV"),
            ("STUDENT_T", "STUDENT_T_EV"),
        ]:
            gg = g.dropna(subset=[pred_col, "REALIZED_RETURN"])
            pred = gg[pred_col].to_numpy(float)
            real = gg["REALIZED_RETURN"].to_numpy(float)

            selected = real[pred > 0]

            benchmark_rows.append({
                "RUN_ID": RUN_ID,
                "K": int(k),
                "HORIZON": int(h),
                "MODEL": model_name,
                "N": len(gg),
                "MEAN_PREDICTION": float(np.mean(pred)),
                "MEAN_REALIZED_ALL": float(np.mean(real)),
                "N_PRED_POSITIVE": int(len(selected)),
                "MEAN_REALIZED_WHEN_PRED_POSITIVE": (
                    float(np.mean(selected))
                    if len(selected) else np.nan
                ),
                "POSITIVE_RATE_WHEN_PRED_POSITIVE": (
                    float(np.mean(selected > 0))
                    if len(selected) else np.nan
                ),
                "PRED_REALIZED_SPEARMAN": (
                    float(spearmanr(pred, real).statistic)
                    if len(gg) > 2
                    and np.std(pred) > 0
                    and np.std(real) > 0
                    else np.nan
                ),
            })

    benchmark = pd.DataFrame(benchmark_rows)

    # ---------------------------------------------------------------
    # Time-block stability.
    # ---------------------------------------------------------------
    qdf["OOS_DATE"] = pd.to_datetime(qdf["QueryDate"])
    stability_rows = []

    for (k, h), g in qdf.groupby(["K", "HORIZON"]):
        dates = np.sort(g["OOS_DATE"].unique())

        if len(dates) < 5:
            continue

        edges = np.linspace(0, len(dates), 6).astype(int)

        for b in range(5):
            block_dates = dates[edges[b]:edges[b+1]]
            gb = g[g["OOS_DATE"].isin(block_dates)]

            stability_rows.append({
                "RUN_ID": RUN_ID,
                "K": int(k),
                "HORIZON": int(h),
                "OOS_BLOCK": b + 1,
                "N": len(gb),
                "MEAN_RANK_IC": cross_sectional_metrics(gb)[
                    "MEAN_RANK_IC"
                ],
                "MEAN_REALIZED": float(
                    gb["REALIZED_RETURN"].mean()
                ),
                "MEAN_TOP_MINUS_BOTTOM": cross_sectional_metrics(gb)[
                    "MEAN_TOP_MINUS_BOTTOM"
                ],
            })

    stability = pd.DataFrame(stability_rows)

    # ---------------------------------------------------------------
    # Purging audit.
    # ---------------------------------------------------------------
    gap_audit = pd.DataFrame(purging_audit)

    if gap_audit.empty or not (gap_audit["STATUS"] == "PASS").all():
        raise RuntimeError("PURGING_AUDIT FAIL")

    # ---------------------------------------------------------------
    # Write artifacts.
    # ---------------------------------------------------------------
    outdir = "cere_phase5a_cross_sectional"
    os.makedirs(outdir, exist_ok=True)

    qdf.drop(columns=["OOS_DATE"], errors="ignore").to_csv(
        os.path.join(outdir, f"phase5a_oos_queries_{RUN_ID}.csv"),
        index=False,
    )

    summary.to_csv(
        os.path.join(outdir, f"phase5a_oos_summary_{RUN_ID}.csv"),
        index=False,
    )

    benchmark.to_csv(
        os.path.join(outdir, f"phase5a_benchmark_{RUN_ID}.csv"),
        index=False,
    )

    stability.to_csv(
        os.path.join(outdir, f"phase5a_stability_{RUN_ID}.csv"),
        index=False,
    )

    gap_audit.to_csv(
        os.path.join(outdir, f"phase5a_purging_audit_{RUN_ID}.csv"),
        index=False,
    )

    pd.DataFrame(audit).to_csv(
        os.path.join(outdir, f"phase5a_ticker_audit_{RUN_ID}.csv"),
        index=False,
    )

    # ---------------------------------------------------------------
    # Console summary: this is the only text the user needs to paste.
    # ---------------------------------------------------------------
    print()
    print("--- FASE 5A RESUMEN ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Tickers OK: {len(datasets)}")
    print(f"Global query dates: {len(query_dates)}")
    print(
        "Cross-sectional observations:",
        len(qdf),
    )
    print()

    print("--- OOS CERE CROSS-SECTIONAL ---")
    print(
        summary[
            [
                "K", "HORIZON",
                "N_OBSERVATIONS",
                "MEAN_PREDICTED_EV",
                "MEAN_REALIZED",
                "REALIZED_POSITIVE",
                "MEAN_EDGE_VS_DRIFT",
                "MEAN_EDGE_VS_STUDENT_T",
                "MEAN_RANK_IC",
                "MEDIAN_RANK_IC",
                "POSITIVE_IC_RATE",
                "MEAN_TOP_MINUS_BOTTOM",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("--- BENCHMARK / PREDICCIÓN POSITIVA ---")
    print(
        benchmark[
            [
                "K", "HORIZON", "MODEL",
                "N",
                "MEAN_PREDICTION",
                "MEAN_REALIZED_ALL",
                "N_PRED_POSITIVE",
                "MEAN_REALIZED_WHEN_PRED_POSITIVE",
                "POSITIVE_RATE_WHEN_PRED_POSITIVE",
                "PRED_REALIZED_SPEARMAN",
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
            MEAN_RANK_IC=("MEAN_RANK_IC", "mean"),
            BLOCKS_POSITIVE_IC=(
                "MEAN_RANK_IC",
                lambda x: int(np.sum(x > 0)),
            ),
            MEAN_TOP_MINUS_BOTTOM=("MEAN_TOP_MINUS_BOTTOM", "mean"),
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
        "PASS" if (gap_audit["STATUS"] == "PASS").all()
        else "FAIL",
    )

    print()
    print(f"Archivos: {outdir}/")
    print(f"Duration/run ID: {RUN_ID}")
    print("FASE 5A COMPLETADA SIN ESCRITURA A GOOGLE SHEETS.")


if __name__ == "__main__":
    main()
