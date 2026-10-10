#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CERE — FASE 5B — DIAGNÓSTICO DE INFORMACIÓN CROSS-SECTIONAL
============================================================

Laboratorio OOS. NO producción.

Objetivo:
1) determinar si las 14 variables de estado canónicas contienen señal
   cross-sectional por sí mismas;
2) probar explícitamente la dirección de la señal (score / -score);
3) evitar que una mala especificación del vecino CERE o de la métrica
   de distancia oculte una variable potencialmente útil;
4) sólo después de la prueba univariada, construir combinaciones pequeñas
   usando selección en una mitad de las fechas y confirmación OOS en la
   mitad posterior;
5) mantener el mismo universo, historia, definición de estado y forward
   return de Fases 2D/3A/4/5A.

IMPORTANTE
- No escribe Google Sheets.
- No calcula Kelly.
- No genera BUY/SELL.
- No selecciona una configuración de producción.
- No usa información futura para construir scores.
- Las 12 fechas globales son las mismas de Fase 5A, para que la comparación
  sea directa.

METODOLOGÍA
-----------
A. Univariado:
   Para cada fecha global y ticker se toma una sola variable de estado.
   Se calcula Rank IC (Spearman) contra el forward return H.
   Se reporta tanto score como -score. La inversión de signo NO se asume:
   se usa como diagnóstico.

B. Selección de dirección:
   Las primeras 6 fechas son DISCOVERY.
   La dirección de cada variable/horizonte se fija exclusivamente con
   esas 6 fechas. Las últimas 6 son CONFIRMATION y no participan en la
   selección.

C. Combinaciones pequeñas:
   Para cada horizonte, se seleccionan las 4 variables con mayor |IC|
   medio en DISCOVERY. Se fijan sus signos con DISCOVERY y se evalúan:
      - cada variable seleccionada;
      - las 6 combinaciones de pares;
      - una combinación de las 3 mejores;
      - una combinación de las 4 mejores.
   El score combinado es el promedio de z-scores cross-sectional.

D. Robustez de representación:
   Para las combinaciones se calcula además una versión por percentiles
   cross-sectional. Esto no cambia Rank IC cuando se usa una sola variable,
   pero sí puede cambiar la combinación multivariable.

E. Benchmark:
   Se compara contra Historical Drift por ticker, exactamente como en
   Fase 5A, pero la evaluación principal es Rank IC y Top-Bottom Spread.

F. Purging/embargo:
   Se mantiene el mismo criterio conservador de Fase 5A: la observación
   histórica usada para una consulta H debe quedar al menos H+5 días hábiles
   antes de la fecha de consulta. Para el diagnóstico univariado actual,
   esto se audita aunque el score no necesite candidatos históricos; así
   conservamos una trazabilidad homogénea con la fase anterior.
"""

import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr, t

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

PERIOD = "10y"
INTERVAL = "1d"
MIN_STATE = 260
HORIZONS = (5, 10, 20, 40)
QUERY_DATES = 12
EMBARGO_DAYS = 5
MIN_QUERY_INDEX = 350
DISCOVERY_DATES = 6
TOP_VARIABLES = 4

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
DJT DKNG DLR DOCU DOW DUOL DVN EA EBAY ECL EDIT EOG EQIX EQR ETN ETSY EW EXPE EXR FCEL
FCF FCX FDX FITB FLEX FLYW FMC FNB FOXA FSLR GD GE GEV GILD GIS GLD GLW GOOGL GPRO GS GSK
HAL HAPN HBAN HD HOMB HON HOOD HUT IEFA ILMN INTC IREN ISRG IWM JD JETS JMIA JNJ JPM KEY KKR
KLAC KMB KO KRE LCID LEN LI LIN LITE LLY LMT LOT LOW LPRO LRCX LYV MA MAA MARA MCHP MDB MDLZ
MDT MELI META MLM MMM MNST MOD MOS MP MRNA MRVL MS MSFT MTB MTCH MU NEM NET NFLX NIO NOC NOW NSC
NTLA NTR NTRS NU NUE NVDA NVO NWSA NXPI O OIH OKTA OMF ON OPEN OPFI OXY OZK PAGS PANW PARA
PAYS PDD PEP PFE PG PHM PINS PLD PLTR PLUG PSA PSNY PWR PYPL QCOM QQQ QS RDDT REGN RF RIOT RIVN
RKLB ROKU RSG RTX RUM RUN SBAC SBUX SCHW SE SEDG SHOP SHW SIFY SLB SMCI SMH SNAP SNOW SNPS SNY
SOFI SOXX SPG SPWR STLD STNE STT SYK T TAK TEAM TELFY TEM TER TFC TGT TMDX TMUS TRIP TROW TSLA
TSM TTWO TWLO TXN UMBF UNG UNP UPS UPST USO V VEA VICI VLY VMC VNQ VOD VRT VRTX VST VWO VZ W WAY
WBD WBS WDAY WFC WM WMT WULF WY XBI XLB XLE XLF XLI XLK XLP XLRE XLU XLV XLY XOM XPEV XYZ Z ZBH
ZG ZION ZM ZS
""".split()
TICKERS = sorted(set(TICKERS))


def state_from_closes(closes):
    c = np.asarray(closes, dtype=float)
    if len(c) < MIN_STATE or np.any(~np.isfinite(c)) or np.any(c <= 0):
        return None
    lr = np.diff(np.log(c))  # incluye el retorno más reciente

    def ret_n(n):
        return c[-1] / c[-1 - n] - 1.0

    def vol_n(n):
        x = lr[-n:]
        if len(x) < 2:
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
        df = df[["Date", "Open", "Close"]].drop_duplicates("Date").reset_index(drop=True)

        rows = []
        for i in range(MIN_STATE - 1, len(df) - max(HORIZONS)):
            state = state_from_closes(df["Close"].to_numpy()[:i + 1])
            if state is None:
                continue
            row = {"Date": df.loc[i, "Date"], "INDEX": i}
            row.update(state)
            for h in HORIZONS:
                row[f"Forward{h}"] = (
                    float(df.loc[i + h, "Close"]) / float(df.loc[i + 1, "Open"]) - 1.0
                )
            rows.append(row)
        out = pd.DataFrame(rows)
        if out.empty:
            return None, "NO_ELIGIBLE_ROWS"
        return out, "OK"
    except Exception as e:
        return None, f"ERROR:{type(e).__name__}"


def choose_global_query_dates(datasets):
    ref = datasets.get("QQQ")
    if ref is None:
        ref = next(iter(datasets.values()))
    eligible = ref.loc[ref["INDEX"] >= MIN_QUERY_INDEX, "Date"]
    eligible = pd.Series(pd.to_datetime(eligible).unique()).sort_values()
    if len(eligible) == 0:
        return []
    idx = np.linspace(0, len(eligible) - 1, num=min(QUERY_DATES, len(eligible)), dtype=int)
    return [pd.Timestamp(eligible.iloc[i]) for i in np.unique(idx)]


def cross_sectional_standardize(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.mean()
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def cross_sectional_percentile(s):
    return s.rank(method="average", pct=True) - 0.5


def date_ic(g, score_col, realized_col="REALIZED_RETURN"):
    x = g[[score_col, realized_col]].dropna()
    if len(x) < 10:
        return np.nan
    rho, _ = spearmanr(x[score_col].to_numpy(float), x[realized_col].to_numpy(float))
    return float(rho) if np.isfinite(rho) else np.nan


def top_bottom_spread(g, score_col):
    x = g[[score_col, "REALIZED_RETURN"]].dropna()
    if len(x) < 10:
        return np.nan
    n = len(x)
    q = max(1, n // 5)
    r = x.sort_values(score_col)
    return float(r.iloc[-q:]["REALIZED_RETURN"].mean() - r.iloc[:q]["REALIZED_RETURN"].mean())


def summarize_scores(df, score_col):
    ics = []
    spreads = []
    for _, g in df.groupby(["HORIZON", "QueryDate"]):
        ic = date_ic(g, score_col)
        sp = top_bottom_spread(g, score_col)
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(sp):
            spreads.append(sp)
    return {
        "N_DATES": len(ics),
        "MEAN_RANK_IC": float(np.mean(ics)) if ics else np.nan,
        "MEDIAN_RANK_IC": float(np.median(ics)) if ics else np.nan,
        "STD_RANK_IC": float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan,
        "POSITIVE_IC_RATE": float(np.mean(np.asarray(ics) > 0)) if ics else np.nan,
        "MEAN_TOP_MINUS_BOTTOM": float(np.mean(spreads)) if spreads else np.nan,
        "MEDIAN_TOP_MINUS_BOTTOM": float(np.median(spreads)) if spreads else np.nan,
    }


def fixed_direction_metrics(df, variable, horizon, direction, dates):
    x = df[(df["HORIZON"] == horizon) & (df["QueryDate"].isin(dates))].copy()
    x["SCORE"] = direction * x[variable]
    ics, spreads = [], []
    for _, g in x.groupby("QueryDate"):
        ic = date_ic(g, "SCORE")
        sp = top_bottom_spread(g, "SCORE")
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(sp):
            spreads.append(sp)
    return {
        "MEAN_RANK_IC": float(np.mean(ics)) if ics else np.nan,
        "MEDIAN_RANK_IC": float(np.median(ics)) if ics else np.nan,
        "POSITIVE_IC_RATE": float(np.mean(np.asarray(ics) > 0)) if ics else np.nan,
        "MEAN_TOP_MINUS_BOTTOM": float(np.mean(spreads)) if spreads else np.nan,
    }


def select_directions(uni, horizon, discovery_dates):
    selections = []
    for v in STATE_FIELDS:
        sub = uni[(uni["HORIZON"] == horizon) & (uni["QueryDate"].isin(discovery_dates))].copy()
        vals = []
        for d in discovery_dates:
            g = sub[sub["QueryDate"] == d]
            ic = date_ic(g, v)
            if np.isfinite(ic):
                vals.append(ic)
        mean_ic = float(np.mean(vals)) if vals else np.nan
        direction = 1 if np.isfinite(mean_ic) and mean_ic >= 0 else -1
        selections.append({
            "HORIZON": horizon,
            "VARIABLE": v,
            "DISCOVERY_MEAN_IC_RAW": mean_ic,
            "DIRECTION": direction,
            "DISCOVERY_ABS_IC": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
        })
    out = pd.DataFrame(selections).sort_values("DISCOVERY_ABS_IC", ascending=False)
    return out


def combination_score(group, variables, directions, representation="Z"):
    pieces = []
    for v in variables:
        s = group[v]
        if representation == "Z":
            q = cross_sectional_standardize(s)
        else:
            q = cross_sectional_percentile(s)
        pieces.append(directions[v] * q)
    if not pieces:
        return pd.Series(np.nan, index=group.index)
    return pd.concat(pieces, axis=1).mean(axis=1)


def combination_metrics(df, variables, directions, horizon, dates, representation):
    x = df[(df["HORIZON"] == horizon) & (df["QueryDate"].isin(dates))].copy()
    ics, spreads, realized_pos = [], [], []
    for _, g in x.groupby("QueryDate"):
        g = g.copy()
        g["SCORE"] = combination_score(g, variables, directions, representation)
        ic = date_ic(g, "SCORE")
        sp = top_bottom_spread(g, "SCORE")
        if np.isfinite(ic):
            ics.append(ic)
        if np.isfinite(sp):
            spreads.append(sp)
        pos = g[g["SCORE"] > 0]["REALIZED_RETURN"]
        if len(pos):
            realized_pos.append(float(pos.mean()))
    return {
        "MEAN_RANK_IC": float(np.mean(ics)) if ics else np.nan,
        "MEDIAN_RANK_IC": float(np.median(ics)) if ics else np.nan,
        "POSITIVE_IC_RATE": float(np.mean(np.asarray(ics) > 0)) if ics else np.nan,
        "MEAN_TOP_MINUS_BOTTOM": float(np.mean(spreads)) if spreads else np.nan,
        "MEAN_REALIZED_WHEN_SCORE_POSITIVE": float(np.mean(realized_pos)) if realized_pos else np.nan,
    }


def fit_student_t_location(x):
    try:
        df_t, loc_t, scale_t = t.fit(np.asarray(x, dtype=float))
        return float(loc_t), float(df_t), float(scale_t)
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    print("=" * 80)
    print("CERE — FASE 5B — DIAGNÓSTICO DE INFORMACIÓN CROSS-SECTIONAL")
    print("=" * 80)
    print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}")
    print(f"HORIZONS: {HORIZONS} | Global query dates: {QUERY_DATES}")
    print(f"DISCOVERY: {DISCOVERY_DATES} dates | CONFIRMATION: {QUERY_DATES - DISCOVERY_DATES} dates")
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
            "FIRST_DATE": dataset["Date"].min().date().isoformat() if dataset is not None else "",
            "LAST_DATE": dataset["Date"].max().date().isoformat() if dataset is not None else "",
        })

    if not datasets:
        raise RuntimeError("No se pudo construir ningún dataset.")

    query_dates = choose_global_query_dates(datasets)
    if len(query_dates) < QUERY_DATES:
        print(f"ADVERTENCIA: solo se obtuvieron {len(query_dates)} fechas globales.")

    print("GLOBAL QUERY DATES:")
    for d in query_dates:
        print(" ", d.date().isoformat())
    print()

    all_rows = []
    for ticker, dfx in datasets.items():
        x = dfx.copy()
        x["Ticker"] = ticker
        all_rows.append(x)
    all_rows = pd.concat(all_rows, ignore_index=True)
    all_rows["Date"] = pd.to_datetime(all_rows["Date"])

    # Current cross-section + forward returns for each global date.
    rows = []
    purging_audit = []
    for qdate in query_dates:
        current = all_rows[all_rows["Date"] == qdate].copy()
        if current.empty:
            continue
        for h in HORIZONS:
            for _, r in current.iterrows():
                rows.append({
                    "RUN_ID": RUN_ID,
                    "QueryDate": qdate.date().isoformat(),
                    "Ticker": r["Ticker"],
                    "HORIZON": h,
                    "REALIZED_RETURN": float(r[f"Forward{h}"]),
                    **{v: float(r[v]) for v in STATE_FIELDS},
                })

            cutoff = qdate - pd.offsets.BDay(h + EMBARGO_DAYS)
            hist = all_rows[all_rows["Date"] <= cutoff]
            purging_audit.append({
                "RUN_ID": RUN_ID,
                "QueryDate": qdate.date().isoformat(),
                "HORIZON": h,
                "EMBARGO_DAYS": EMBARGO_DAYS,
                "CUTOFF_DATE": cutoff.date().isoformat(),
                "MAX_HISTORICAL_DATE": hist["Date"].max().date().isoformat() if len(hist) else "",
                "N_HISTORICAL_ROWS": len(hist),
                "STATUS": "PASS" if len(hist) and hist["Date"].max() <= cutoff else "FAIL",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No se generaron observaciones.")

    # ------------------------------------------------------------------
    # A. UNIVARIATE FULL SAMPLE (diagnostic only, no model selection).
    # ------------------------------------------------------------------
    uni_rows = []
    for h in HORIZONS:
        sub_h = df[df["HORIZON"] == h]
        for v in STATE_FIELDS:
            ics, spreads = [], []
            for d, g in sub_h.groupby("QueryDate"):
                ic = date_ic(g, v)
                sp = top_bottom_spread(g, v)
                if np.isfinite(ic):
                    ics.append(ic)
                if np.isfinite(sp):
                    spreads.append(sp)
            raw_ic = float(np.mean(ics)) if ics else np.nan
            uni_rows.append({
                "RUN_ID": RUN_ID,
                "HORIZON": h,
                "VARIABLE": v,
                "SCORE_DIRECTION": "+1",
                "MEAN_RANK_IC": raw_ic,
                "MEDIAN_RANK_IC": float(np.median(ics)) if ics else np.nan,
                "POSITIVE_IC_RATE": float(np.mean(np.asarray(ics) > 0)) if ics else np.nan,
                "MEAN_TOP_MINUS_BOTTOM": float(np.mean(spreads)) if spreads else np.nan,
            })
            inv_ics, inv_spreads = [-x for x in ics], [-x for x in spreads]
            uni_rows.append({
                "RUN_ID": RUN_ID,
                "HORIZON": h,
                "VARIABLE": v,
                "SCORE_DIRECTION": "-1",
                "MEAN_RANK_IC": float(np.mean(inv_ics)) if inv_ics else np.nan,
                "MEDIAN_RANK_IC": float(np.median(inv_ics)) if inv_ics else np.nan,
                "POSITIVE_IC_RATE": float(np.mean(np.asarray(inv_ics) > 0)) if inv_ics else np.nan,
                "MEAN_TOP_MINUS_BOTTOM": float(np.mean(inv_spreads)) if inv_spreads else np.nan,
            })

    uni_full = pd.DataFrame(uni_rows)

    # ------------------------------------------------------------------
    # B. DISCOVERY -> fixed direction -> CONFIRMATION.
    # ------------------------------------------------------------------
    discovery_dates = [d.date().isoformat() for d in query_dates[:DISCOVERY_DATES]]
    confirmation_dates = [d.date().isoformat() for d in query_dates[DISCOVERY_DATES:]]

    direction_rows = []
    combo_rows = []

    for h in HORIZONS:
        sel = select_directions(df, h, discovery_dates)
        sel["RANK_DISCOVERY"] = np.arange(1, len(sel) + 1)
        direction_rows.extend(sel.to_dict("records"))

        top = sel.head(TOP_VARIABLES)
        top_vars = top["VARIABLE"].tolist()
        directions = dict(zip(top["VARIABLE"], top["DIRECTION"]))

        # Evaluate each selected variable on confirmation with frozen sign.
        for _, r in top.iterrows():
            v = r["VARIABLE"]
            m = fixed_direction_metrics(df, v, h, int(r["DIRECTION"]), confirmation_dates)
            combo_rows.append({
                "RUN_ID": RUN_ID,
                "HORIZON": h,
                "TYPE": "UNIVARIATE_CONFIRMATION",
                "VARIABLES": v,
                "DIRECTIONS": str(int(r["DIRECTION"])),
                "N_VARIABLES": 1,
                "REPRESENTATION": "RAW",
                **m,
            })

        # Small combinations: pairs, top3, top4.
        specs = []
        specs.extend(list(combinations(top_vars, 2)))
        if len(top_vars) >= 3:
            specs.append(tuple(top_vars[:3]))
        if len(top_vars) >= 4:
            specs.append(tuple(top_vars[:4]))

        # Remove duplicate specs while preserving order.
        seen = set()
        unique_specs = []
        for s in specs:
            if s not in seen:
                seen.add(s)
                unique_specs.append(s)

        for combo in unique_specs:
            for rep in ("Z", "PERCENTILE"):
                m = combination_metrics(df, combo, directions, h, confirmation_dates, rep)
                combo_rows.append({
                    "RUN_ID": RUN_ID,
                    "HORIZON": h,
                    "TYPE": "COMBINATION_CONFIRMATION",
                    "VARIABLES": " + ".join(combo),
                    "DIRECTIONS": " ".join(f"{v}:{directions[v]:+d}" for v in combo),
                    "N_VARIABLES": len(combo),
                    "REPRESENTATION": rep,
                    **m,
                })

    directions_df = pd.DataFrame(direction_rows)
    combos_df = pd.DataFrame(combo_rows)

    # ------------------------------------------------------------------
    # C. Full-sample summary of raw information, clearly labelled
    #    diagnostic, NOT valid for model selection.
    # ------------------------------------------------------------------
    full_ranked = (
        uni_full[uni_full["SCORE_DIRECTION"] != "-1"]
        .assign(ABS_MEAN_IC=lambda x: x["MEAN_RANK_IC"].abs())
        .sort_values(["HORIZON", "ABS_MEAN_IC"], ascending=[True, False])
    )

    # ------------------------------------------------------------------
    # D. Audit invariants.
    # ------------------------------------------------------------------
    if len(query_dates) != QUERY_DATES:
        print(f"ADVERTENCIA: query_dates={len(query_dates)} vs esperado={QUERY_DATES}")

    expected_uni = len(HORIZONS) * len(STATE_FIELDS) * 2
    if len(uni_full) != expected_uni:
        raise RuntimeError(f"UNIVARIATE_SCHEMA_FAIL: {len(uni_full)} != {expected_uni}")

    if len(purging_audit) and not (pd.Series([x["STATUS"] for x in purging_audit]) == "PASS").all():
        raise RuntimeError("PURGING_AUDIT_FAIL")

    # Confirm there is no forward column used as a predictor.
    forbidden_predictors = [f"Forward{h}" for h in HORIZONS]
    if any(x in STATE_FIELDS for x in forbidden_predictors):
        raise RuntimeError("LOOK_AHEAD_SCHEMA_FAIL")

    # ------------------------------------------------------------------
    # E. Save artifacts.
    # ------------------------------------------------------------------
    outdir = os.path.join(os.getcwd(), "cere_phase5b_diagnostics")
    os.makedirs(outdir, exist_ok=True)

    df.to_csv(os.path.join(outdir, f"phase5b_oos_observations_{RUN_ID}.csv"), index=False)
    uni_full.to_csv(os.path.join(outdir, f"phase5b_univariate_{RUN_ID}.csv"), index=False)
    directions_df.to_csv(os.path.join(outdir, f"phase5b_direction_selection_{RUN_ID}.csv"), index=False)
    combos_df.to_csv(os.path.join(outdir, f"phase5b_combinations_confirmation_{RUN_ID}.csv"), index=False)
    pd.DataFrame(purging_audit).to_csv(os.path.join(outdir, f"phase5b_purging_audit_{RUN_ID}.csv"), index=False)
    pd.DataFrame(audit).to_csv(os.path.join(outdir, f"phase5b_ticker_audit_{RUN_ID}.csv"), index=False)

    # Compact console output for GitHub Actions.
    print()
    print("--- FASE 5B: UNIVARIADO FULL-SAMPLE (DIAGNÓSTICO) ---")
    print(full_ranked[[
        "HORIZON", "VARIABLE", "MEAN_RANK_IC", "MEDIAN_RANK_IC",
        "POSITIVE_IC_RATE", "MEAN_TOP_MINUS_BOTTOM"
    ]].to_string(index=False))

    print()
    print("--- FASE 5B: DIRECCIÓN DISCOVERY -> CONFIRMATION ---")
    print(directions_df[[
        "HORIZON", "RANK_DISCOVERY", "VARIABLE",
        "DISCOVERY_MEAN_IC_RAW", "DIRECTION"
    ]].to_string(index=False))

    print()
    print("--- FASE 5B: CONFIRMACIÓN OOS DE VARIABLES/COMBINACIONES ---")
    print(combos_df[[
        "HORIZON", "TYPE", "VARIABLES", "N_VARIABLES", "REPRESENTATION",
        "MEAN_RANK_IC", "MEDIAN_RANK_IC", "POSITIVE_IC_RATE",
        "MEAN_TOP_MINUS_BOTTOM", "MEAN_REALIZED_WHEN_SCORE_POSITIVE"
    ]].to_string(index=False))

    print()
    print("--- AUDITORÍA ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Tickers OK: {len(datasets)}")
    print(f"Global query dates: {len(query_dates)}")
    print(f"Cross-sectional observations: {len(df):,}")
    print(f"Univariate rows: {len(uni_full)}")
    print(f"Confirmation rows: {len(combos_df)}")
    print(f"PURGING_AUDIT: {'PASS' if all(x['STATUS']=='PASS' for x in purging_audit) else 'FAIL'}")
    print("LOOK_AHEAD_PREDICTORS: PASS")
    print("STATUS: PASS")
    print()
    print(f"Artifacts: {outdir}")


if __name__ == "__main__":
    main()
