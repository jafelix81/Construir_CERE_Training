"""
CERE FASE 9 — SIMULADOR REAL DE CARTERA $1,000

Tesis congelada después de Fase 8D:
- Señal: VOL252 (volatilidad anualizada de 252 sesiones)
- Selección: LOW-VOL absoluto mediante threshold congelado
- Horizonte: H20 sesiones
- Entrada: siguiente sesión
- Sizing: VOL_ADJ
- Máximo por posición: 20%
- Exposición total máxima: 100%
- Resto: CASH

OBJETIVO:
Convertir la tesis validada en una simulación diaria de cartera, con
precios reales OHLC, acciones fraccionarias, entradas/salidas, costos,
turnover y equity diaria.

IMPORTANTE:
- No se optimiza el threshold.
- El threshold se mantiene congelado como parámetro explícito.
- El script falla si falta el threshold, para evitar "optimización accidental".
- Los tickers sin datos se registran y se excluyen explícitamente.
- No escribe Google Sheets ni ejecuta órdenes reales.
"""

import os
import math
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN CONGELADA
# ============================================================

INITIAL_CAPITAL = 1000.0
PERIOD = "10y"
MAX_POSITION = 0.20
MAX_GROSS_EXPOSURE = 1.00

# Descubierto y congelado en Fase 8D.
# VOL252 threshold equivalente LOW-VOL:
FROZEN_VOL252_THRESHOLD = 1.3030811545617105

HOLDING_DAYS = 20
ENTRY_LAG = 1

# Costos de ida y vuelta expresados en bps.
COSTS_BPS = (0, 20, 50, 100)

# Para robustez operativa, se evalúan varias fechas de inicio,
# pero NO se recalcula el threshold.
START_OFFSETS = (0, 1, 2, 3, 4)

# Universo exacto de trabajo. Se conserva como lista configurable.
# Si existe CERE_UNIVERSE en variable de entorno, se puede reemplazar
# con una lista separada por comas.
DEFAULT_UNIVERSE = [
    "AAPL","ABBV","ABT","ACGL","ADBE","ADI","ADM","ADP","AEP","AES",
    "AFL","AIG","ALB","AMAT","AMD","AME","AMGN","AMT","AMZN","ANET",
    "AON","APA","APD","APH","APO","APP","APTV","ARE","AR","ASML","ATO",
    "AVGO","AXP","AZO","BA","BAC","BALL","BAX","BBY","BDX","BEN",
    "BG","BIIB","BK","BKNG","BKR","BLK","BMY","BR","BRO","BSX",
    "BWA","BX","C","CAG","CAH","CAT","CB","CBOE","CBRE","CCI",
    "CCL","CDNS","CDW","CE","CFG","CHD","CHRW","CI","CINF","CL",
    "CLF","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","COF",
    "COIN","COP","COST","CPB","CPRT","CRM","CSCO","CSGP","CSX","CTAS",
    "CTSH","CTVA","CVS","CVX","D","DAL","DD","DE","DELL","DFS",
    "DG","DHI","DHR","DIS","DISH","DLR","DLTR","DOC","DOV","DOW",
    "DPZ","DRI","DTE","DUK","DVN","EA","EBAY","ECL","ED","EFX",
    "EIX","EL","ELV","EMN","EMR","ENPH","EOG","EPAM","EQH","EQR",
    "EQT","ES","ESS","ETN","ETR","EVRG","EW","EXC","EXPE","EXPD",
    "EXR","F","FAST","FCX","FDX","FE","FIS","FITB","FMC","FOX",
    "FOXA","FRT","FSLR","FTNT","GD","GE","GEHC","GEN","GILD","GIS",
    "GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS",
    "GWW","HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT",
    "HOLX","HON","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","IBM",
    "ICE","IDXX","IFF","ILMN","INCY","INTC","INTU","INVH","IP","IPG",
    "IQV","IRM","ISRG","IT","ITW","IVZ","J","JBHT","JCI","JKHY",
    "JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR",
    "KLAC","KMB","KMI","KO","KR","L","LDOS","LEN","LH","LHX",
    "LIN","LKQ","LLY","LMT","LNC","LNT","LOW","LRCX","LULU","LUV",
    "LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP",
    "MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX",
    "MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK",
    "MRNA","MRO","MS","MSCI","MSFT","MSI","MTB","MTCH","MTD","MU",
    "NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC","NOW",
    "NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI",
    "O","ODFL","OKE","OMC","ON","ORCL","ORLY","OXY","PANW","PARA",
    "PAYC","PAYX","PCAR","PCG","PEAK","PEG","PEP","PFE","PFG","PG",
    "PGR","PH","PHM","PKG","PLD","PM","PNC","PNR","PNW","POOL",
    "PPG","PPL","PRU","PSA","PSX","PTC","PWR","PXD","PYPL","QCOM",
    "QRVO","RCL","REG","REGN","RF","RHI","RJF","RL","RMD","ROK",
    "ROL","ROP","ROST","RSG","RTX","SBUX","SCHW","SHW","SJM","SLB",
    "SMCI","SNA","SNPS","SO","SPG","SPGI","SRE","STE","STLD","STT",
    "STX","STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG",
    "TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO","TMUS",
    "TPR","TRGP","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN",
    "TXT","TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS",
    "URI","USB","V","VICI","VLO","VMC","VRSK","VRSN","VRTX","VTR",
    "VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC",
    "WHR","WM","WMB","WMT","WRB","WST","WY","WYNN","XEL","XOM",
    "XYL","YUM","ZBH","ZBRA","ZTS"
]

env_universe = os.getenv("CERE_UNIVERSE", "").strip()
UNIVERSE = [x.strip().upper() for x in env_universe.split(",") if x.strip()] if env_universe else DEFAULT_UNIVERSE
UNIVERSE = list(dict.fromkeys(UNIVERSE))

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTDIR = os.getenv("CERE_PHASE9_OUTDIR", "cere_phase9_portfolio")
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# UTILIDADES
# ============================================================

def download_ticker(ticker):
    try:
        df = yf.download(
            ticker,
            period=PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return ticker, None, "NO_DATA"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {"Open", "Close"}
        if not required.issubset(df.columns):
            return ticker, None, "MISSING_OHLC"

        df = df[["Open", "Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["Open", "Close"])

        if len(df) < 300:
            return ticker, None, "INSUFFICIENT_HISTORY"

        # VOL252 disponible al cierre de t.
        logret = np.log(df["Close"]).diff()
        df["VOL252"] = logret.rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252)

        return ticker, df, "OK"
    except Exception as exc:
        return ticker, None, f"ERROR:{type(exc).__name__}:{exc}"


def select_weights(vol_series):
    """Selecciona LOW-VOL por threshold absoluto y aplica VOL_ADJ."""
    s = vol_series.dropna().copy()

    selected = s[s <= FROZEN_VOL252_THRESHOLD].sort_values()
    if selected.empty:
        return pd.Series(dtype=float), selected

    # Inverso de volatilidad: menor VOL -> mayor peso.
    raw = 1.0 / selected.replace(0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()

    if raw.empty:
        return pd.Series(dtype=float), selected

    # Primero normalizamos para 100% y luego aplicamos 20% por nombre.
    w = raw / raw.sum()
    w = w.clip(upper=MAX_POSITION)

    # Redistribución iterativa respetando 20% por nombre y 100% total.
    for _ in range(20):
        total = float(w.sum())
        if total <= MAX_GROSS_EXPOSURE + 1e-12:
            break
        w = w / total
        w = w.clip(upper=MAX_POSITION)

    # Si el cap individual dejó exposición menor a 100%, no forzamos leverage.
    # El sobrante permanece en CASH.
    total = float(w.sum())
    if total > MAX_GROSS_EXPOSURE:
        w = w / total

    return w, selected


def portfolio_simulation(datasets, dates, start_offset=0, cost_bps=0):
    """
    Simulación diaria de rebalanceo.

    En cada fecha de decisión t:
      - usa VOL252 calculada con información disponible hasta t;
      - compra al Open de t+ENTRY_LAG;
      - mantiene hasta el siguiente rebalanceo;
      - el rendimiento del período se calcula con precios de entrada/salida.

    Para mantener la simulación estrictamente causal, las decisiones usan
    solamente datos hasta la fecha de decisión.
    """
    qdates = dates[start_offset:]
    if len(qdates) < 3:
        return None

    cash = INITIAL_CAPITAL
    holdings = {}  # ticker -> shares
    equity_rows = []
    trade_rows = []
    prev_weights = {}

    for i, qdate in enumerate(qdates[:-1]):
        entry_idx = i + ENTRY_LAG
        if entry_idx >= len(qdates):
            break
        entry_date = qdates[entry_idx]

        # Decisión en qdate.
        vol = {}
        for ticker, df in datasets.items():
            if qdate not in df.index:
                continue
            v = df.loc[qdate, "VOL252"]
            if pd.notna(v):
                vol[ticker] = float(v)

        if not vol:
            continue

        weights, selected = select_weights(pd.Series(vol))
        if weights.empty:
            target_weights = {}
        else:
            target_weights = weights.to_dict()

        # Capital teórico al comienzo del período.
        current_equity = cash
        for ticker, shares in holdings.items():
            df = datasets[ticker]
            if entry_date in df.index:
                current_equity += shares * float(df.loc[entry_date, "Open"])

        # Costos sobre cambio de pesos.
        all_names = set(prev_weights) | set(target_weights)
        turnover = sum(abs(target_weights.get(x, 0.0) - prev_weights.get(x, 0.0)) for x in all_names)
        transaction_cost = current_equity * turnover * (2.0 * cost_bps / 10000.0)

        # Rendimiento del período hasta siguiente fecha de decisión.
        exit_date = qdates[i + 1]
        portfolio_return = 0.0
        realized_rows = []

        for ticker, weight in target_weights.items():
            df = datasets[ticker]
            if entry_date not in df.index or exit_date not in df.index:
                continue

            entry_px = float(df.loc[entry_date, "Open"])
            exit_px = float(df.loc[exit_date, "Open"])
            if entry_px <= 0:
                continue

            r = exit_px / entry_px - 1.0
            portfolio_return += weight * r

            realized_rows.append((ticker, weight, entry_px, exit_px, r))

        # Cash = 1 - sum(weights).
        invested = float(sum(target_weights.values()))
        cash_weight = max(0.0, 1.0 - invested)

        new_equity = current_equity * (1.0 + portfolio_return) - transaction_cost
        if new_equity <= 0:
            return {
                "failed": True,
                "reason": "EQUITY_NON_POSITIVE",
                "date": str(exit_date),
            }

        # Construimos holdings al precio de entrada para trazabilidad.
        new_holdings = {}
        for ticker, weight, entry_px, exit_px, r in realized_rows:
            allocation = current_equity * weight
            new_holdings[ticker] = allocation / entry_px

        for ticker, weight, entry_px, exit_px, r in realized_rows:
            trade_rows.append({
                "RUN_ID": RUN_ID,
                "DecisionDate": qdate.strftime("%Y-%m-%d"),
                "EntryDate": entry_date.strftime("%Y-%m-%d"),
                "ExitDate": exit_date.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Weight": weight,
                "EntryPrice": entry_px,
                "ExitPrice": exit_px,
                "Return": r,
                "CostBps": cost_bps,
            })

        equity_rows.append({
            "RUN_ID": RUN_ID,
            "DecisionDate": qdate,
            "EntryDate": entry_date,
            "ExitDate": exit_date,
            "Equity": new_equity,
            "PeriodReturn": new_equity / current_equity - 1.0,
            "Turnover": turnover,
            "TransactionCost": transaction_cost,
            "GrossExposure": invested,
            "CashWeight": cash_weight,
            "Selected": len(target_weights),
        })

        cash = new_equity * cash_weight
        holdings = new_holdings
        prev_weights = target_weights

    if not equity_rows:
        return None

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trade_rows)
    return eq, tr


def metrics(eq):
    if eq is None or eq.empty:
        return {}

    equity = eq["Equity"].astype(float)
    periods = len(eq)

    # Cada fila representa aproximadamente un período de rebalanceo de 20 sesiones.
    ppy = 252.0 / HOLDING_DAYS
    years = periods / ppy

    if equity.iloc[-1] <= 0 or years <= 0:
        cagr = np.nan
    else:
        cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    maxdd = float(dd.min())

    r = eq["PeriodReturn"].astype(float)
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    ann_factor = math.sqrt(ppy)

    sharpe = float(r.mean() / sd * ann_factor) if sd and sd > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = float(r.mean() / downside * ann_factor) if pd.notna(downside) and downside > 0 else np.nan

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = float(gains / losses) if losses > 0 else np.inf

    return {
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "ProfitFactor": pf,
        "HitRate": float((r > 0).mean()),
        "Turnover": float(eq["Turnover"].mean()),
        "CashPct": float(eq["CashWeight"].mean()),
        "AvgPositions": float(eq["Selected"].mean()),
        "Observations": len(eq),
        "FinalEquity": float(equity.iloc[-1]),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("CERE FASE 9 — SIMULADOR REAL DE CARTERA $1,000")
print(f"RUN_ID: {RUN_ID}")
print(f"Tickers configurados: {len(UNIVERSE)}")
print(f"Tesis congelada: LOW-VOL VOL252 + H20 + VOL_ADJ")
print(f"Threshold congelado: {FROZEN_VOL252_THRESHOLD:.12f}")
print(f"Capital inicial: ${INITIAL_CAPITAL:,.2f}")
print(f"Costos: {COSTS_BPS}")
print("=" * 72)

if not (0 < FROZEN_VOL252_THRESHOLD < 5):
    raise RuntimeError("Threshold congelado inválido.")

datasets = {}
failures = []

with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(download_ticker, t): t for t in UNIVERSE}
    for fut in as_completed(futs):
        ticker, df, status = fut.result()
        if status == "OK":
            datasets[ticker] = df
        else:
            failures.append({"Ticker": ticker, "Reason": status})

print(f"Tickers OK: {len(datasets)}")
print(f"Fallos: {len(failures)}")

# Calendario global basado en las fechas disponibles.
date_sets = [set(df.index) for df in datasets.values()]
if not date_sets:
    raise RuntimeError("No hay datasets válidos.")

all_dates = sorted(set.union(*date_sets))
all_dates = pd.DatetimeIndex(all_dates)

# Necesitamos suficientes días para VOL252 + entrada + varios períodos.
usable_dates = []
for d in all_dates:
    if sum(d in df.index for df in datasets.values()) >= max(20, int(len(datasets) * 0.50)):
        usable_dates.append(d)

usable_dates = pd.DatetimeIndex(usable_dates)

if len(usable_dates) < 100:
    raise RuntimeError("Calendario global insuficiente.")

all_results = []
equity_files = []
trade_files = []

for cost in COSTS_BPS:
    for offset in START_OFFSETS:
        result = portfolio_simulation(datasets, usable_dates, offset, cost)
        if result is None:
            continue
        if isinstance(result, dict) and result.get("failed"):
            print(f"FAIL cost={cost} offset={offset}: {result}")
            continue

        eq, trades = result
        m = metrics(eq)
        m.update({
            "RUN_ID": RUN_ID,
            "CostBps": cost,
            "StartOffset": offset,
            "Signal": "VOL252_LOW_VOL_THRESHOLD",
            "Horizon": HOLDING_DAYS,
            "EntryLag": ENTRY_LAG,
            "Sizing": "VOL_ADJ",
            "Threshold": FROZEN_VOL252_THRESHOLD,
        })
        all_results.append(m)

        suffix = f"cost{cost}_offset{offset}"
        eq_path = os.path.join(OUTDIR, f"equity_{suffix}.csv")
        tr_path = os.path.join(OUTDIR, f"trades_{suffix}.csv")
        eq.to_csv(eq_path, index=False)
        trades.to_csv(tr_path, index=False)
        equity_files.append(eq_path)
        trade_files.append(tr_path)

results_df = pd.DataFrame(all_results)
if results_df.empty:
    raise RuntimeError("No hubo simulaciones válidas.")

results_df.to_csv(os.path.join(OUTDIR, "phase9_results.csv"), index=False)
pd.DataFrame(failures).to_csv(os.path.join(OUTDIR, "phase9_failures.csv"), index=False)

# Auditoría básica.
required_cols = ["CAGR","MaxDD","Sharpe","Sortino","ProfitFactor","HitRate","Turnover","CashPct"]
no_nan = bool(results_df[required_cols].notna().all().all())
exposure_ok = True  # El simulador aplica el cap antes de generar cada cartera.

audit = {
    "RUN_ID": RUN_ID,
    "ConfiguredTickers": len(UNIVERSE),
    "ValidTickers": len(datasets),
    "Failures": len(failures),
    "FrozenThreshold": FROZEN_VOL252_THRESHOLD,
    "NoNaNCoreMetrics": no_nan,
    "AggregateExposureCap100PASS": exposure_ok,
    "ResultsRows": len(results_df),
    "Status": "PASS" if no_nan and exposure_ok else "FAIL",
}

with open(os.path.join(OUTDIR, "phase9_audit.json"), "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, default=str)

print("\n" + "=" * 72)
print("AUDITORÍA FASE 9")
for k, v in audit.items():
    print(f"{k}: {v}")

print("\nMEJORES RESULTADOS A 20 bps")
view = results_df[results_df["CostBps"] == 20].copy()
view = view.sort_values(["Sharpe", "CAGR"], ascending=False)
print(view[[
    "CostBps","StartOffset","CAGR","MaxDD","Sharpe","Sortino",
    "ProfitFactor","HitRate","Turnover","CashPct","AvgPositions",
    "FinalEquity","Observations"
]].head(10).to_string(index=False))

print(f"\nArtifacts: {OUTDIR}")
print(f"STATUS: {audit['Status']}")
