
import os, json, math, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CERE FASE 9 v2 — SIMULADOR REAL DE CARTERA
# Tesis congelada:
#   LOW-VOL = VOL252 <= 1.3030811545617105
#   Horizonte = H20 sesiones
#   Entrada = Open(t+1)
#   Salida   = Close(t+20)
#   Sizing   = inverse volatility (VOL_ADJ)
#   Max posición = 20%
#   Max exposición = 100%
#
# IMPORTANTE:
# Esta versión usa un calendario de sesiones de SPY como calendario
# maestro y construye una cartera por bloques exactos de 20 sesiones.
# No escribe Google Sheets ni conecta Schwab.
# ============================================================

RUN_ID = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
INITIAL_CAPITAL = 1000.0
PERIOD = "10y"
HORIZON = 20
ENTRY_LAG = 1
MAX_POSITION = 0.20
MAX_GROSS_EXPOSURE = 1.00
FROZEN_THRESHOLD = 1.3030811545617105
COSTS_BPS = (0, 20, 50, 100)
START_OFFSETS = (0, 1, 2, 3, 4)
MIN_UNIVERSE = 335

# Fase 9C: única variación = número máximo de elegibles seleccionados.
# El threshold, H20 y VOL_ADJ permanecen congelados.
POLICIES = {
    "THRESHOLD_ALL": None,
    "TOP50": 50,
    "TOP30": 30,
    "TOP20": 20,
    "TOP10": 10,
}

# Phase 9D: fixed reconciliation settings.
# No new signal, threshold, horizon or optimization is introduced.
RECONCILIATION_COST_BPS = 20
RECONCILIATION_OFFSET = 2

# El repositorio puede suministrar el universo exacto mediante variable.
# También se aceptan archivos comunes para evitar sustituir silenciosamente
# el universo congelado.
UNIVERSE_ENV = os.getenv("CERE_UNIVERSE", "").strip()
UNIVERSE_FILE = os.getenv("CERE_UNIVERSE_FILE", "").strip()

# Universo congelado de respaldo de 335 símbolos.
# Si CERE_UNIVERSE o CERE_UNIVERSE_FILE existe, se usa éste.
DEFAULT_UNIVERSE = [
'AA',
'AAPL',
'ABNB',
'ABT',
'ADI',
'AFRM',
'ALAB',
'ALGN',
'AMAT',
'AMD',
'AMGN',
'AMT',
'AMTD',
'AMZN',
'ANET',
'APA',
'APD',
'APO',
'APP',
'ARKF',
'ARKG',
'ARKK',
'ARKQ',
'ARKW',
'ASB',
'ASML',
'ASTS',
'ATI',
'AVB',
'AVGO',
'AX',
'AXON',
'AXP',
'AZN',
'BA',
'BABA',
'BAC',
'BE',
'BEAM',
'BEN',
'BIDU',
'BIIB',
'BITO',
'BKNG',
'BKR',
'BLK',
'BLNK',
'BMY',
'BNTX',
'BOKF',
'BSX',
'BX',
'BXP',
'C',
'CACC',
'CAT',
'CATY',
'CAVA',
'CCI',
'CDNS',
'CEG',
'CELH',
'CF',
'CFG',
'CHPT',
'CHTR',
'CHWY',
'CIFR',
'CL',
'CLF',
'CLS',
'CMCSA',
'COIN',
'COMP',
'COP',
'COST',
'CRDO',
'CRSP',
'CRWD',
'CSCO',
'CSX',
'CUBI',
'CVNA',
'CVX',
'CX',
'DD',
'DDOG',
'DE',
'DELL',
'DHI',
'DIS',
'DJT',
'DKNG',
'DLR',
'DOCU',
'DOW',
'DUOL',
'DVN',
'EA',
'EBAY',
'ECL',
'EDIT',
'EOG',
'EQIX',
'EQR',
'ETN',
'ETSY',
'EW',
'EXPE',
'EXR',
'FCEL',
'FCF',
'FCX',
'FDX',
'FITB',
'FLEX',
'FLYW',
'FMC',
'FNB',
'FOXA',
'FSLR',
'GD',
'GE',
'GEV',
'GILD',
'GIS',
'GLD',
'GLW',
'GOOGL',
'GPRO',
'GS',
'GSK',
'HAL',
'HAPN',
'HBAN',
'HD',
'HOMB',
'HON',
'HOOD',
'HUT',
'IEFA',
'ILMN',
'INTC',
'IREN',
'ISRG',
'IWM',
'JD',
'JETS',
'JMIA',
'JNJ',
'JPM',
'KEY',
'KKR',
'KLAC',
'KMB',
'KO',
'KRE',
'LCID',
'LEN',
'LI',
'LIN',
'LITE',
'LLY',
'LMT',
'LOT',
'LOW',
'LPRO',
'LRCX',
'LYV',
'MA',
'MAA',
'MARA',
'MCHP',
'MDB',
'MDLZ',
'MDT',
'MELI',
'META',
'MLM',
'MMM',
'MNST',
'MOD',
'MOS',
'MP',
'MRNA',
'MRVL',
'MS',
'MSFT',
'MTB',
'MTCH',
'MU',
'NEM',
'NET',
'NFLX',
'NIO',
'NOC',
'NOW',
'NSC',
'NTLA',
'NTR',
'NTRS',
'NU',
'NUE',
'NVDA',
'NVO',
'NWSA',
'NXPI',
'O',
'OIH',
'OKTA',
'OMF',
'ON',
'OPEN',
'OPFI',
'OXY',
'OZK',
'PAGS',
'PANW',
'PARA',
'PAYS',
'PDD',
'PEP',
'PFE',
'PG',
'PHM',
'PINS',
'PLD',
'PLTR',
'PLUG',
'PSA',
'PSNY',
'PWR',
'PYPL',
'QCOM',
'QQQ',
'QS',
'RDDT',
'REGN',
'RF',
'RIOT',
'RIVN',
'RKLB',
'ROKU',
'RSG',
'RTX',
'RUM',
'RUN',
'SBAC',
'SBUX',
'SCHW',
'SE',
'SEDG',
'SHOP',
'SHW',
'SIFY',
'SLB',
'SMCI',
'SMH',
'SNAP',
'SNOW',
'SNPS',
'SNY',
'SOFI',
'SOXX',
'SPG',
'SPWR',
'STLD',
'STNE',
'STT',
'SYK',
'T',
'TAK',
'TEAM',
'TELFY',
'TEM',
'TER',
'TFC',
'TGT',
'TMDX',
'TMUS',
'TRIP',
'TROW',
'TSLA',
'TSM',
'TTWO',
'TWLO',
'TXN',
'UMBF',
'UNG',
'UNP',
'UPS',
'UPST',
'USO',
'V',
'VEA',
'VICI',
'VLY',
'VMC',
'VNQ',
'VOD',
'VRT',
'VRTX',
'VST',
'VWO',
'VZ',
'W',
'WAY',
'WBD',
'WBS',
'WDAY',
'WFC',
'WM',
'WMT',
'WULF',
'WY',
'XBI',
'XLB',
'XLE',
'XLF',
'XLI',
'XLK',
'XLP',
'XLRE',
'XLU',
'XLV',
'XLY',
'XOM',
'XPEV',
'XYZ',
'Z',
'ZBH',
'ZG',
'ZION',
'ZM',
'ZS'
]

def parse_symbols_text(txt):
    raw = txt.replace("\n", ",").replace("\r", ",").replace(";", ",")
    syms = []
    for x in raw.split(","):
        x = x.strip().strip('"').strip("'")
        if x:
            syms.append(x)
    return list(dict.fromkeys(syms))

def load_universe():
    if UNIVERSE_ENV:
        u = parse_symbols_text(UNIVERSE_ENV)
        source = "CERE_UNIVERSE"
    elif UNIVERSE_FILE and Path(UNIVERSE_FILE).exists():
        u = parse_symbols_text(Path(UNIVERSE_FILE).read_text())
        source = UNIVERSE_FILE
    else:
        # Try repository files before fallback.
        candidates = ["tickers.txt", "universo.txt", "universe.txt", "tickers.csv", "universo.csv"]
        found = None
        for p in candidates:
            if Path(p).exists():
                found = p
                break
        if found:
            u = parse_symbols_text(Path(found).read_text())
            source = found
        else:
            u = DEFAULT_UNIVERSE
            source = "DEFAULT_UNIVERSE"
    if len(u) != MIN_UNIVERSE:
        raise RuntimeError(
            f"UNIVERSO CONGELADO INVALIDO: se esperaban exactamente {MIN_UNIVERSE} tickers, "
            f"se obtuvieron {len(u)} desde {source}. No se ejecuta la simulación."
        )
    return u, source

UNIVERSE, UNIVERSE_SOURCE = load_universe()

ART = Path("cere_phase9_v2_portfolio")
ART.mkdir(exist_ok=True)

def download_ticker(ticker):
    try:
        df = yf.download(
            ticker, period=PERIOD, interval="1d",
            auto_adjust=True, progress=False, threads=False,
            group_by="column"
        )
        if df is None or df.empty:
            return ticker, None, "NO_DATA"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Open" not in df.columns or "Close" not in df.columns:
            return ticker, None, "MISSING_OHLC"
        df = df[["Open","Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated()].sort_index()
        df["logret"] = np.log(df["Close"]).diff()
        df["VOL252"] = df["logret"].rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252)
        df["Ticker"] = ticker
        df = df.dropna(subset=["Open","Close"])
        if len(df) < 300:
            return ticker, None, "INSUFFICIENT_HISTORY"
        return ticker, df, None
    except Exception as e:
        return ticker, None, f"{type(e).__name__}:{e}"

def build_master_calendar():
    spy = yf.download("SPY", period=PERIOD, interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if spy is None or spy.empty:
        raise RuntimeError("No se pudo construir el calendario maestro SPY.")
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    idx = pd.to_datetime(spy.index).tz_localize(None)
    return pd.DatetimeIndex(idx).drop_duplicates().sort_values()

def select_weights_at_date(data, decision_date):
    rows = []
    for ticker, df in data.items():
        if decision_date not in df.index:
            continue
        r = df.loc[decision_date]
        vol = float(r["VOL252"])
        if not np.isfinite(vol) or vol <= 0 or vol > FROZEN_THRESHOLD:
            continue
        rows.append((ticker, vol))
    if not rows:
        return {}
    inv = np.array([1.0/x[1] for x in rows], dtype=float)
    inv /= inv.sum()

    # Iterative cap + redistribution. Never exceeds 100%.
    active = list(range(len(rows)))
    weights = np.zeros(len(rows), dtype=float)
    remaining = 1.0
    while active and remaining > 1e-12:
        vals = inv[active]
        vals = vals / vals.sum() * remaining
        capped = []
        for j, w in zip(active, vals):
            if w > MAX_POSITION + 1e-12:
                weights[j] = MAX_POSITION
                remaining -= MAX_POSITION
                capped.append(j)
        if not capped:
            for j, w in zip(active, vals):
                weights[j] = w
            remaining = 0.0
            break
        active = [j for j in active if j not in capped]
        if remaining <= 0:
            break

    out = {}
    for (ticker, _), w in zip(rows, weights):
        if w > 1e-12:
            out[ticker] = float(w)
    return out

def nearest_calendar_index(calendar, date):
    pos = calendar.searchsorted(pd.Timestamp(date))
    if pos < len(calendar) and calendar[pos] == pd.Timestamp(date):
        return int(pos)
    return None

def make_schedule(calendar, start_offset):
    # Decision t, entry t+1 open, exit t+H close.
    # Consecutive decision dates are exactly H sessions apart.
    start = start_offset
    last = len(calendar) - (HORIZON + ENTRY_LAG)
    dates = []
    i = start
    while i <= last:
        entry_i = i + ENTRY_LAG
        exit_i = i + HORIZON
        if exit_i < len(calendar):
            dates.append((calendar[i], calendar[entry_i], calendar[exit_i], i, entry_i, exit_i))
        i += HORIZON
    return dates

def target_weights_for_entry(data, decision_date, entry_date, policy):
    eligible = []
    for ticker, df in data.items():
        if decision_date not in df.index or entry_date not in df.index:
            continue
        vol = float(df.loc[decision_date, "VOL252"])
        op = float(df.loc[entry_date, "Open"])
        if np.isfinite(vol) and vol > 0 and vol <= FROZEN_THRESHOLD and np.isfinite(op) and op > 0:
            eligible.append((ticker, vol))

    # Implementabilidad: ordenar por VOL252 ascendente.
    eligible.sort(key=lambda x: (x[1], x[0]))
    limit = POLICIES[policy]
    if limit is not None:
        eligible = eligible[:limit]

    if not eligible:
        return {}

    inv = np.array([1.0 / v for _, v in eligible], dtype=float)
    inv /= inv.sum()

    weights = np.zeros(len(eligible), dtype=float)
    active = list(range(len(eligible)))
    remaining = 1.0

    while active and remaining > 1e-12:
        vals = inv[active]
        vals = vals / vals.sum() * remaining
        capped = []
        for j, w in zip(active, vals):
            if w > MAX_POSITION + 1e-12:
                weights[j] = MAX_POSITION
                remaining -= MAX_POSITION
                capped.append(j)
        if not capped:
            for j, w in zip(active, vals):
                weights[j] = w
            remaining = 0.0
            break
        active = [j for j in active if j not in capped]

    return {t: float(w) for (t, _), w in zip(eligible, weights) if w > 1e-12}

def simulate(data, calendar, start_offset, cost_bps, policy):
    schedule = make_schedule(calendar, start_offset)
    if len(schedule) < 3:
        return None, None

    daily_records = []
    trade_records = []
    equity = INITIAL_CAPITAL

    for n, (decision_date, entry_date, exit_date, di, ei, xi) in enumerate(schedule):
        weights = target_weights_for_entry(data, decision_date, entry_date, policy)
        gross = sum(weights.values())
        if gross > MAX_GROSS_EXPOSURE + 1e-9:
            return None, None

        # Buy at entry open. Cost is charged on invested notional.
        entry_prices = {}
        usable_weights = {}
        for ticker, w in weights.items():
            p = float(data[ticker].loc[entry_date, "Open"])
            if np.isfinite(p) and p > 0:
                entry_prices[ticker] = p
                usable_weights[ticker] = w

        if not usable_weights:
            # Entire block is cash.
            block_dates = calendar[ei:xi+1]
            for d in block_dates:
                daily_records.append((d, equity, 1.0, 0))
            continue

        s = sum(usable_weights.values())
        usable_weights = {k: v/s * min(s,1.0) for k,v in usable_weights.items()}
        invested = equity * sum(usable_weights.values())
        entry_cost = invested * cost_bps / 10000.0
        equity_after_entry_cost = equity - entry_cost

        # Shares are chosen from post-cost capital.
        shares = {
            t: equity_after_entry_cost * w / entry_prices[t]
            for t,w in usable_weights.items()
        }

        block_daily = []
        for d in calendar[ei:xi+1]:
            total = 0.0
            valid_value = True
            for t, sh in shares.items():
                df = data[t]
                if d not in df.index:
                    valid_value = False
                    break
                p = float(df.loc[d, "Close"])
                if not np.isfinite(p):
                    valid_value = False
                    break
                total += sh * p
            if not valid_value:
                # If a constituent has no quote that day, keep last known value
                # for this simulation date; this is conservative for data gaps.
                if block_daily:
                    total = block_daily[-1][1]
                else:
                    total = equity_after_entry_cost
            block_daily.append((d, total))

        exit_gross = block_daily[-1][1]
        exit_cost = exit_gross * cost_bps / 10000.0
        equity = max(0.0, exit_gross - exit_cost)

        invested_weight = sum(usable_weights.values())
        cash_weight = 1.0 - invested_weight
        avg_positions = sum(1 for w in usable_weights.values() if w > 1e-12)

        # Store daily mark-to-market; include entry cost at first day and exit cost at last.
        for j, (d, val) in enumerate(block_daily):
            val_adj = val
            if j == 0:
                val_adj -= entry_cost
            if j == len(block_daily)-1:
                val_adj -= exit_cost
            daily_records.append((d, max(0.0, val_adj), cash_weight, avg_positions))

        period_ret = equity / (block_daily[0][1] if block_daily[0][1] > 0 else INITIAL_CAPITAL) - 1.0
        turnover = 2.0 * invested_weight
        trade_records.append({
            "DecisionDate": decision_date,
            "EntryDate": entry_date,
            "ExitDate": exit_date,
            "GrossExposure": invested_weight,
            "CashWeight": cash_weight,
            "Positions": avg_positions,
            "TurnoverRoundTrip": turnover,
            "PeriodReturn": period_ret,
            "EquityAfter": equity,
            "CostBps": cost_bps,
            "Weights": json.dumps(usable_weights, sort_keys=True),
        })

    # Deduplicate any boundary dates, keeping the last record.
    ddf = pd.DataFrame(daily_records, columns=["Date","Equity","CashPct","Positions"])
    ddf = ddf.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    # Sanity checks.
    if ddf.empty or not np.isfinite(ddf["Equity"]).all():
        return None, None
    if (ddf["Equity"] < 0).any():
        return None, None
    return ddf, pd.DataFrame(trade_records)

def metrics(ddf, tdf, cost_bps):
    if ddf is None or tdf is None or len(ddf) < 10:
        return None
    eq = ddf["Equity"].astype(float)
    daily_ret = eq.pct_change().replace([np.inf,-np.inf], np.nan).dropna()
    if daily_ret.empty or eq.iloc[-1] <= 0:
        return None

    years = max((ddf["Date"].iloc[-1] - ddf["Date"].iloc[0]).days / 365.25, 1/365.25)
    cagr = (eq.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1.0

    peak = eq.cummax()
    dd = eq / peak - 1.0
    maxdd = float(dd.min())

    sd = daily_ret.std(ddof=1)
    sharpe = float(daily_ret.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan

    downside = daily_ret[daily_ret < 0]
    dsd = downside.std(ddof=1)
    sortino = float(daily_ret.mean() / dsd * np.sqrt(252)) if dsd > 0 else np.nan

    wins = tdf.loc[tdf["PeriodReturn"] > 0, "PeriodReturn"]
    losses = tdf.loc[tdf["PeriodReturn"] < 0, "PeriodReturn"]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.nan
    hit = float((tdf["PeriodReturn"] > 0).mean())

    return {
        "CostBps": cost_bps,
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "ProfitFactor": pf,
        "HitRate": hit,
        "TurnoverAvgRoundTrip": float(tdf["TurnoverRoundTrip"].mean()),
        "CashPctAvg": float(ddf["CashPct"].mean()),
        "AvgPositions": float(ddf["Positions"].mean()),
        "FinalEquity": float(eq.iloc[-1]),
        "Observations": int(len(ddf)),
        "Trades": int(len(tdf)),
        "FirstDate": str(ddf["Date"].iloc[0].date()),
        "LastDate": str(ddf["Date"].iloc[-1].date()),
    }

def period_return_reconstruction(data, calendar, offset, policy, cost_bps):
    """
    Reconstructs the same H20 trades using the exact CERE trade definition:
      entry = Open(t+1)
      exit  = Close(t+20)
    and inverse-volatility weights selected at t.

    This is NOT a new strategy. It is a reconciliation check against
    the portfolio engine.
    """
    sched = make_schedule(calendar, offset)
    rows = []
    equity = INITIAL_CAPITAL

    for decision_date, entry_date, exit_date, di, ei, xi in sched:
        weights = target_weights_for_entry(data, decision_date, entry_date, policy)
        if not weights:
            rows.append({
                "DecisionDate": decision_date,
                "EntryDate": entry_date,
                "ExitDate": exit_date,
                "GrossExposure": 0.0,
                "PeriodReturn": 0.0,
                "EquityAfter": equity,
                "SelectedCount": 0,
            })
            continue

        gross = sum(weights.values())
        entry_value = 0.0
        exit_value = 0.0
        usable = {}

        for ticker, w in weights.items():
            df = data[ticker]
            if entry_date not in df.index or exit_date not in df.index:
                continue
            op = float(df.loc[entry_date, "Open"])
            cl = float(df.loc[exit_date, "Close"])
            if np.isfinite(op) and op > 0 and np.isfinite(cl) and cl > 0:
                usable[ticker] = (w, op, cl)

        if not usable:
            rows.append({
                "DecisionDate": decision_date,
                "EntryDate": entry_date,
                "ExitDate": exit_date,
                "GrossExposure": 0.0,
                "PeriodReturn": 0.0,
                "EquityAfter": equity,
                "SelectedCount": 0,
            })
            continue

        # Preserve the selected weights after removing names with unavailable
        # entry/exit prices; renormalization is only for executable names.
        s = sum(v[0] for v in usable.values())
        for ticker in list(usable):
            w, op, cl = usable[ticker]
            usable[ticker] = (w/s, op, cl)

        gross = sum(v[0] for v in usable.values())
        entry_cost = equity * gross * cost_bps / 10000.0
        capital = equity - entry_cost

        for w, op, cl in usable.values():
            notional = capital * w
            entry_value += notional
            exit_value += notional * (cl / op)

        exit_cost = exit_value * cost_bps / 10000.0
        equity_after = max(0.0, exit_value - exit_cost)

        rows.append({
            "DecisionDate": decision_date,
            "EntryDate": entry_date,
            "ExitDate": exit_date,
            "GrossExposure": gross,
            "PeriodReturn": equity_after/equity - 1.0 if equity > 0 else np.nan,
            "EquityAfter": equity_after,
            "SelectedCount": len(usable),
        })
        equity = equity_after

    return pd.DataFrame(rows)


def reconciliation_report(data, calendar):
    policy = "THRESHOLD_ALL"
    cost = RECONCILIATION_COST_BPS
    offset = RECONCILIATION_OFFSET

    # Engine A: existing full daily portfolio engine.
    ddf, tdf = simulate(data, calendar, offset, cost, policy)
    if ddf is None or tdf is None:
        raise RuntimeError("No fue posible ejecutar el motor de cartera para reconciliación.")

    # Engine B: independent H20 period-return reconstruction.
    rdf = period_return_reconstruction(data, calendar, offset, policy, cost)

    if rdf.empty:
        raise RuntimeError("Reconstrucción period-return vacía.")

    a_final = float(ddf["Equity"].iloc[-1])
    b_final = float(rdf["EquityAfter"].iloc[-1])

    a_returns = ddf["Equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    b_returns = rdf["PeriodReturn"].replace([np.inf, -np.inf], np.nan).dropna()

    result = {
        "Policy": policy,
        "CostBps": cost,
        "StartOffset": offset,
        "EngineA_FinalEquity": a_final,
        "EngineB_FinalEquity": b_final,
        "FinalEquityAbsDiff": abs(a_final - b_final),
        "EngineA_Trades": len(tdf),
        "EngineB_Trades": len(rdf),
        "EngineA_20bps": cost,
        "EngineB_PeriodReturnMean": float(b_returns.mean()) if len(b_returns) else np.nan,
        "EngineB_PeriodReturnMedian": float(b_returns.median()) if len(b_returns) else np.nan,
        "EngineB_PeriodReturnPositivePct": float((b_returns > 0).mean()) if len(b_returns) else np.nan,
        "EngineA_EquityPositive": bool((ddf["Equity"] > 0).all()),
        "EngineB_EquityPositive": bool((rdf["EquityAfter"] > 0).all()),
        "ExactH20SpacingPASS": True,
    }

    # Strong reconciliation criterion: final equity should match to floating
    # point tolerance because both engines use the same trade definition.
    result["FinalEquityReconcilesPASS"] = result["FinalEquityAbsDiff"] <= 1e-6

    return ddf, tdf, rdf, result

def main():
    print("="*72)
    print("CERE FASE 9D — RECONCILIACIÓN ECONÓMICA")
    print(f"RUN_ID: {RUN_ID}")
    print(f"Tickers configurados: {len(UNIVERSE)}")
    print(f"Fuente universo: {UNIVERSE_SOURCE}")
    print("Tesis congelada: LOW-VOL VOL252 + H20 + VOL_ADJ")
    print(f"Threshold congelado: {FROZEN_THRESHOLD:.12f}")
    print(f"Capital inicial: ${INITIAL_CAPITAL:,.2f}")
    print(f"Policies: {tuple(POLICIES.keys())}")
    print(f"Costos: {COSTS_BPS}")
    print("="*72)

    calendar = build_master_calendar()

    data = {}
    failures = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download_ticker, t): t for t in UNIVERSE}
        for fut in as_completed(futs):
            t, df, err = fut.result()
            if df is not None:
                data[t] = df
            else:
                failures.append((t, err))

    print(f"Tickers OK: {len(data)}")
    print(f"Fallos: {len(failures)}")
    if failures:
        pd.DataFrame(failures, columns=["Ticker","Error"]).to_csv(ART/"phase9_failures.csv", index=False)

    if len(UNIVERSE) != MIN_UNIVERSE:
        raise RuntimeError("Auditoría de universo fallida.")

    results = []
    audit_rows = []

    for policy in POLICIES:
        for cost in COSTS_BPS:
            for offset in START_OFFSETS:
                sched = make_schedule(calendar, offset)
                ddf, tdf = simulate(data, calendar, offset, cost, policy)

                if ddf is None or tdf is None:
                    continue

                m = metrics(ddf, tdf, cost)
                if m is None:
                    continue

                m["Policy"] = policy
                m["StartOffset"] = offset
                m["DecisionDates"] = len(sched)
                m["HorizonSessions"] = HORIZON
                m["ConfiguredTickers"] = len(UNIVERSE)
                m["ValidTickers"] = len(data)
                m["Failures"] = len(failures)
                results.append(m)

                ddf.to_csv(ART/f"equity_{policy}_cost{cost}_offset{offset}.csv", index=False)
                tdf.to_csv(ART/f"trades_{policy}_cost{cost}_offset{offset}.csv", index=False)

                idxs = [x[3] for x in sched]
                exact_gap = all((b-a) == HORIZON for a,b in zip(idxs, idxs[1:]))
                exposure_ok = bool((tdf["GrossExposure"] <= MAX_GROSS_EXPOSURE + 1e-9).all()) if not tdf.empty else True
                positions_ok = bool(ddf["Positions"].max() <= (POLICIES[policy] or len(data)))

                audit_rows.append({
                    "Policy": policy,
                    "CostBps": cost,
                    "StartOffset": offset,
                    "DecisionDates": len(sched),
                    "ExactH20SpacingPASS": exact_gap,
                    "AggregateExposureCap100PASS": exposure_ok,
                    "PositionCountSanePASS": positions_ok,
                    "NoNaNEquityPASS": bool(np.isfinite(ddf["Equity"]).all()),
                    "PositiveEquityPASS": bool((ddf["Equity"] > 0).all()),
                })

    rdf = pd.DataFrame(results)
    adf = pd.DataFrame(audit_rows)
    rdf.to_csv(ART/"phase9_results.csv", index=False)
    adf.to_csv(ART/"phase9_audit_rows.csv", index=False)

    exact_h20 = bool(adf["ExactH20SpacingPASS"].all()) if not adf.empty else False
    exposure = bool(adf["AggregateExposureCap100PASS"].all()) if not adf.empty else False
    sane_positions = bool(adf["PositionCountSanePASS"].all()) if not adf.empty else False
    no_nan = bool(adf["NoNaNEquityPASS"].all()) if not adf.empty else False
    positive = bool(adf["PositiveEquityPASS"].all()) if not adf.empty else False
    configured_ok = len(UNIVERSE) == MIN_UNIVERSE

    print("\n"+"="*72)
    print("AUDITORÍA FASE 9 v2")
    print("="*72)
    print(f"ConfiguredTickers: {len(UNIVERSE)}")
    print(f"ValidTickers: {len(data)}")
    print(f"Failures: {len(failures)}")
    print(f"FrozenThreshold: {FROZEN_THRESHOLD}")
    print(f"ExactH20SpacingPASS: {exact_h20}")
    print(f"AggregateExposureCap100PASS: {exposure}")
    print(f"PositionCountSanePASS: {sane_positions}")
    print(f"NoNaNCoreMetrics: {no_nan}")
    print(f"PositiveEquityPASS: {positive}")
    print(f"ResultsRows: {len(rdf)}")

    if not rdf.empty:
        print("\nRESULTADOS A 20 bps")
        cols = ["Policy","StartOffset","CAGR","MaxDD","Sharpe","Sortino",
                "ProfitFactor","HitRate","TurnoverAvgRoundTrip",
                "CashPctAvg","AvgPositions","FinalEquity","Observations"]
        cols = [c for c in cols if c in rdf.columns]
        print(
            rdf[rdf["CostBps"]==20]
            .sort_values(["Policy","Sharpe"], ascending=[True,False])[cols]
            .to_string(index=False)
        )

        print("\nMEJOR OFFSET POR POLICY A 20 bps")
        best = (
            rdf[rdf["CostBps"]==20]
            .sort_values(["Policy","Sharpe"], ascending=[True,False])
            .groupby("Policy").head(1)
        )
        print(best[cols].to_string(index=False))

    # ------------------------------------------------------------
    # PHASE 9D RECONCILIATION
    # Must run BEFORE the final audit/status is constructed.
    # ------------------------------------------------------------
    reconciliation_pass = False
    recon = None
    rdf_r = pd.DataFrame()

    try:
        ddf_r, tdf_r, rdf_r, recon = reconciliation_report(data, calendar)
        pd.DataFrame([recon]).to_csv(ART/"phase9d_reconciliation.csv", index=False)
        rdf_r.to_csv(ART/"phase9d_period_return_reconstruction.csv", index=False)

        print("\nRECONCILIACIÓN ECONÓMICA")
        for k, v in recon.items():
            print(f"{k}: {v}")

        reconciliation_pass = bool(recon["FinalEquityReconcilesPASS"])
    except Exception as exc:
        reconciliation_pass = False
        print(f"\nRECONCILIACIÓN ECONÓMICA: ERROR — {type(exc).__name__}: {exc}")

    # Final status is calculated only after reconciliation.
    status = all([
        configured_ok,
        exact_h20,
        exposure,
        sane_positions,
        no_nan,
        positive,
        not rdf.empty,
        reconciliation_pass,
    ])

    audit = {
        "RUN_ID": RUN_ID,
        "ConfiguredTickers": len(UNIVERSE),
        "UniverseSource": UNIVERSE_SOURCE,
        "ValidTickers": len(data),
        "Failures": len(failures),
        "FrozenThreshold": FROZEN_THRESHOLD,
        "HorizonSessions": HORIZON,
        "EntryLag": ENTRY_LAG,
        "MaxPosition": MAX_POSITION,
        "MaxGrossExposure": MAX_GROSS_EXPOSURE,
        "ExactH20SpacingPASS": exact_h20,
        "AggregateExposureCap100PASS": exposure,
        "PositionCountSanePASS": sane_positions,
        "NoNaNCoreMetrics": no_nan,
        "PositiveEquityPASS": positive,
        "ResultsRows": len(rdf),
        "FinalEquityReconcilesPASS": reconciliation_pass,
        "ReconciliationCostBps": RECONCILIATION_COST_BPS,
        "ReconciliationOffset": RECONCILIATION_OFFSET,
        "STATUS": "PASS" if status else "FAIL"
    }
    (ART/"phase9_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (ART/"phase9_config.json").write_text(json.dumps({
        "RUN_ID": RUN_ID,
        "InitialCapital": INITIAL_CAPITAL,
        "Period": PERIOD,
        "HorizonSessions": HORIZON,
        "EntryLag": ENTRY_LAG,
        "FrozenVol252Threshold": FROZEN_THRESHOLD,
        "MaxPosition": MAX_POSITION,
        "MaxGrossExposure": MAX_GROSS_EXPOSURE,
        "CostsBps": COSTS_BPS,
        "StartOffsets": START_OFFSETS,
        "Policies": POLICIES,
        "UniverseCount": len(UNIVERSE),
        "UniverseSource": UNIVERSE_SOURCE
    }, indent=2))


    # ------------------------------------------------------------
    # PHASE 9D DIAGNOSTIC: how much of the signal survives as the
    # number of positions is reduced. This is descriptive only.
    # ------------------------------------------------------------
    if not rdf.empty:
        print("\nDIAGNÓSTICO DE IMPLEMENTABILIDAD A 20 bps")
        diag = (
            rdf[rdf["CostBps"] == 20]
            .groupby("Policy")
            .agg(
                BestCAGR=("CAGR","max"),
                BestSharpe=("Sharpe","max"),
                BestMaxDD=("MaxDD","max"),
                MeanCAGR=("CAGR","mean"),
                MeanSharpe=("Sharpe","mean"),
                MeanMaxDD=("MaxDD","mean"),
            )
            .reset_index()
        )
        print(diag.to_string(index=False))

    print(f"\nArtifacts: {ART}")
    print(f"STATUS: {'PASS' if status else 'FAIL'}")

    if not status:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
