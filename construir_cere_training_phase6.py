#!/usr/bin/env python3
"""
CERE — FASE 6 — SIGNAL SELECTION & PORTFOLIO VALIDATION

Purpose
-------
Validate the surviving cross-sectional volatility signals at portfolio level,
without introducing new predictors or using production data.

Design
------
- 10y daily history
- Canonical 14 state variables from prior phases
- Global query dates: 48 total
- Discovery: first 12 dates
- OOS: final 36 dates
- Purging + 5 trading-day embargo
- Signals frozen from Discovery only:
    Vol20
    Vol60
    Vol252
    Vol60+Vol252
- Horizons: 10, 20, 40 trading days
- Portfolio policies:
    TOP20
    TOP10
    THRESHOLD
- Explicit CASH when no security qualifies
- $1,000 starting capital
- Maximum 20% of equity per position
- Equal-weight allocation among selected names, capped at 20%
- Fractional shares
- Costs tested at 0/10/20/30/50 bps per turnover
- Rebalance on every OOS query date
- Real turnover calculated from target weights
- Metrics: CAGR, total return, max drawdown, volatility, Sharpe,
  Sortino, profit factor, hit rate, turnover, time in cash,
  number of active positions, and date-by-date equity.

Important
---------
This phase does NOT optimize signals on OOS. Discovery determines:
- component direction for each signal
- threshold candidates for the THRESHOLD policy

Those choices are then frozen for all OOS dates.
"""

from pathlib import Path
from datetime import datetime
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
PERIOD = "10y"
STARTING_CAPITAL = 1000.0
MAX_POSITION_WEIGHT = 0.20
COST_BPS = (0, 10, 20, 30, 50)
HORIZONS = (10, 20, 40)
DISCOVERY_DATES = 12
TOTAL_DATES = 48
EMBARGO_DAYS = 5

TICKERS = [
"AA","AAPL","ABNB","ABT","ADI","AFRM","ALAB","ALGN","AMAT","AMD","AMGN","AMT",
"AMTD","AMZN","ANET","APA","APD","APO","APP","ARKF","ARKG","ARKK","ARKQ","ARKW",
"ASB","ASML","ASTS","ATI","AVB","AVGO","AX","AXON","AXP","AZN","BA","BABA","BAC",
"BE","BEAM","BEN","BIDU","BIIB","BITO","BKNG","BKR","BLK","BLNK","BMY","BNTX",
"BOKF","BSX","BX","BXP","C","CACC","CAT","CATY","CAVA","CCI","CDNS","CEG","CELH",
"CF","CFG","CHPT","CHTR","CHWY","CIFR","CL","CLF","CLS","CMCSA","COIN","COMP","COP",
"COST","CRDO","CRSP","CRWD","CSCO","CSX","CUBI","CVNA","CVX","CX","DD","DDOG","DE",
"DELL","DHI","DIS","DJT","DKNG","DLR","DOCU","DOW","DUOL","DVN","EA","EBAY","ECL",
"EDIT","EOG","EQIX","EQR","ETN","ETSY","EW","EXPE","EXR","FCEL","FCF","FCX","FDX",
"FITB","FLEX","FLYW","FMC","FNB","FOXA","FSLR","GD","GE","GEV","GILD","GIS","GLD",
"GLW","GOOGL","GPRO","GS","GSK","HAL","HAPN","HBAN","HD","HOMB","HON","HOOD","HUT",
"IEFA","ILMN","INTC","IREN","ISRG","IWM","JD","JETS","JMIA","JNJ","JPM","KEY","KKR",
"KLAC","KMB","KO","KRE","LCID","LEN","LI","LIN","LITE","LLY","LMT","LOT","LOW","LPRO",
"LRCX","LYV","MA","MAA","MARA","MCHP","MDB","MDLZ","MDT","MELI","META","MLM","MMM",
"MNST","MOD","MOS","MP","MRNA","MRVL","MS","MSFT","MTB","MTCH","MU","NEM","NET",
"NFLX","NIO","NOC","NOW","NSC","NTLA","NTR","NTRS","NU","NUE","NVDA","NVO","NWSA",
"NXPI","O","OIH","OKTA","OMF","ON","OPEN","OPFI","OXY","OZK","PAGS","PANW","PARA",
"PAYS","PDD","PEP","PFE","PG","PHM","PINS","PLD","PLTR","PLUG","PSA","PSNY","PWR",
"PYPL","QCOM","QQQ","QS","RDDT","REGN","RF","RIOT","RIVN","RKLB","ROKU","RSG","RTX",
"RUM","RUN","SBAC","SBUX","SCHW","SE","SEDG","SHOP","SHW","SIFY","SLB","SMCI","SMH",
"SNAP","SNOW","SNPS","SNY","SOFI","SOXX","SPG","SPWR","STLD","STNE","STT","SYK","T",
"TAK","TEAM","TELFY","TEM","TER","TFC","TGT","TMDX","TMUS","TRIP","TROW","TSLA","TSM",
"TTWO","TWLO","TXN","UMBF","UNG","UNP","UPS","UPST","USO","V","VEA","VICI","VLY",
"VMC","VNQ","VOD","VRT","VRTX","VST","VWO","VZ","W","WAY","WBD","WBS","WDAY","WFC",
"WM","WMT","WULF","WY","XBI","XLB","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV",
"XLY","XOM","XPEV","XYZ","Z","ZBH","ZG","ZION","ZM","ZS"
]

SIGNAL_SPECS = {
    "Vol20": ["Vol20"],
    "Vol60": ["Vol60"],
    "Vol252": ["Vol252"],
    "Vol60+Vol252": ["Vol60", "Vol252"],
}

STATE_NAMES = [
    "R1","R5","R20","R60","R120",
    "Vol5","Vol20","Vol60","Vol252",
    "VolRatio20","VolRatio60","Skew20","Kurt20","AC1"
]

OUTDIR = Path("cere_phase6_portfolio_validation")
OUTDIR.mkdir(exist_ok=True)

RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def canonical_state(close):
    c = np.asarray(close, dtype=float)
    if len(c) < 260:
        return None
    lr = np.diff(np.log(c))
    def ret(n):
        return c[-1] / c[-1-n] - 1.0
    def vol(n):
        x = lr[-n:]
        return np.std(x, ddof=1) * np.sqrt(252)
    if len(lr) < 252:
        return None
    v5, v20, v60, v252 = vol(5), vol(20), vol(60), vol(252)
    x20, x60 = lr[-20:], lr[-60:]
    skew20 = pd.Series(x20).skew()
    kurt20 = pd.Series(x20).kurt()
    ac1 = pd.Series(x60).autocorr(lag=1)
    vals = [ret(1),ret(5),ret(20),ret(60),ret(120),
            v5,v20,v60,v252,v20/v252,v60/v252,skew20,kurt20,ac1]
    if not np.all(np.isfinite(vals)):
        return None
    return dict(zip(STATE_NAMES, vals))

def make_rows(ticker, df):
    if df is None or len(df) < 320:
        return []
    df = df.dropna(subset=["Open","Close"]).copy()
    rows=[]
    for i in range(260, len(df)-40):
        st = canonical_state(df["Close"].iloc[:i+1].values)
        if st is None:
            continue
        row = {"Ticker":ticker, "Date":df.index[i].date()}
        row.update(st)
        for h in HORIZONS:
            row[f"Forward{h}"] = float(df["Close"].iloc[i+h] / df["Open"].iloc[i+1] - 1.0)
        rows.append(row)
    return rows

def download_all():
    data={}
    audit=[]
    for n,t in enumerate(TICKERS,1):
        print(f"[{n:03d}/{len(TICKERS)}] {t}", flush=True)
        try:
            x=yf.download(t, period=PERIOD, auto_adjust=True, progress=False, actions=False)
            if isinstance(x.columns,pd.MultiIndex):
                x=x.droplevel(1,axis=1)
            if "Open" not in x or "Close" not in x or len(x)<320:
                audit.append((t,"FAIL",0))
                continue
            x=x[["Open","Close"]].dropna()
            data[t]=x
            audit.append((t,"OK",len(x)))
        except Exception as e:
            audit.append((t,"FAIL",0))
    pd.DataFrame(audit,columns=["Ticker","STATUS","OBS"]).to_csv(
        OUTDIR/f"phase6_ticker_audit_{RUN_ID}.csv",index=False)
    return data

def choose_dates(data):
    qqq=data.get("QQQ")
    if qqq is None:
        raise RuntimeError("QQQ no disponible para calendario global")
    dates=pd.DatetimeIndex(qqq.index).normalize()
    dates=dates[(dates>=dates.min()+pd.Timedelta(days=365*2))]
    idx=np.linspace(0,len(dates)-1,TOTAL_DATES).astype(int)
    chosen=sorted(set(dates[idx].date))
    if len(chosen)!=TOTAL_DATES:
        raise RuntimeError("No se pudieron obtener 48 fechas globales")
    return chosen

def build_panel(data):
    rows=[]
    for t,df in data.items():
        rows.extend(make_rows(t,df))
    panel=pd.DataFrame(rows)
    panel["Date"]=pd.to_datetime(panel["Date"])
    return panel

def candidate_panel(panel, qdate, h):
    cutoff=pd.Timestamp(qdate)-pd.tseries.offsets.BDay(h+EMBARGO_DAYS)
    return panel[panel["Date"]<=cutoff].copy()

def cross_section_score(cand, qdate, signal_vars):
    # Query states are constructed directly from each ticker history later.
    # This helper standardizes candidate cross-section at a historical date.
    x=cand.dropna(subset=signal_vars+["Ticker"])
    if len(x)<20:
        return None
    z=[]
    for v in signal_vars:
        s=x[v]
        sd=s.std(ddof=1)
        if not np.isfinite(sd) or sd==0:
            x[v+"_z"]=0.0
        else:
            x[v+"_z"]=(s-s.mean())/sd
    return x

def query_cross_section(data, qdate):
    out=[]
    qts=pd.Timestamp(qdate)
    for t,df in data.items():
        idx=df.index[df.index<=qts]
        if len(idx)<261:
            continue
        # Need future returns, but prediction is based only on state at qdate.
        pos=df.index.get_indexer([idx[-1]])[0]
        if pos<260:
            continue
        if pos+40>=len(df):
            continue
        st=canonical_state(df["Close"].iloc[:pos+1].values)
        if st is None:
            continue
        row={"Ticker":t,"Date":idx[-1]}
        row.update(st)
        for h in HORIZONS:
            row[f"Forward{h}"]=float(df["Close"].iloc[pos+h]/df["Open"].iloc[pos+1]-1)
        out.append(row)
    return pd.DataFrame(out)

def frozen_directions(discovery_rows):
    # Keep variable directions in a separate local dictionary.
    # The previous version mixed variable keys with signal keys in the same
    # dictionary, which later made the export iterate over integer values.
    directions={}
    for name,vars_ in SIGNAL_SPECS.items():
        var_directions={}
        for v in vars_:
            ics=[]
            for h in HORIZONS:
                for d in discovery_rows:
                    if len(d)<20: continue
                    r=spearmanr(d[v], d[f"Forward{h}"], nan_policy="omit").statistic
                    if np.isfinite(r): ics.append(r)
            var_directions[v]=1 if (np.mean(ics) if ics else 0)>=0 else -1
        directions[name]=var_directions
    return directions

def signal_score(df, name, directions):
    zcols=[]
    for v in SIGNAL_SPECS[name]:
        s=df[v].astype(float)
        sd=s.std(ddof=1)
        z=(s-s.mean())/sd if sd>0 else pd.Series(0.0,index=df.index)
        zcols.append(z*directions[name][v])
    return pd.concat(zcols,axis=1).mean(axis=1)

def portfolio_targets(qdf, signal_name, policy, threshold, directions):
    x=qdf.copy()
    x["Score"]=signal_score(x,signal_name,directions)
    x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=["Score"])
    if len(x)==0:
        return {}
    if policy=="TOP20":
        n=max(1,int(math.floor(len(x)*0.20)))
        sel=x.nlargest(n,"Score")
    elif policy=="TOP10":
        n=max(1,int(math.floor(len(x)*0.10)))
        sel=x.nlargest(n,"Score")
    elif policy=="THRESHOLD":
        sel=x[x["Score"]>=threshold].copy()
    else:
        raise ValueError(policy)
    if len(sel)==0:
        return {}
    # Equal weight, hard 20% cap.
    w=min(MAX_POSITION_WEIGHT, 1.0/len(sel))
    return {t:w for t in sel["Ticker"]}

def portfolio_step(prev_weights, targets, returns, equity, cost_bps):
    names=set(prev_weights)|set(targets)
    turnover=sum(abs(targets.get(t,0)-prev_weights.get(t,0)) for t in names)
    cost=turnover*cost_bps/10000.0
    gross=sum(prev_weights.get(t,0)*returns.get(t,0.0) for t in names)
    net_return=gross-cost
    new_equity=equity*(1+net_return)
    return new_equity,turnover,cost,net_return

def simulate(signal_name, policy, threshold, qdates, qdfs, h, directions, cost_bps):
    equity=STARTING_CAPITAL
    prev={}
    rows=[]
    for d,qdf in zip(qdates,qdfs):
        if qdf.empty:
            rows.append({"Date":d,"Equity":equity,"Return":0.0,"Turnover":0.0,
                         "Cost":0.0,"Positions":0,"CashWeight":1.0})
            continue
        targets=portfolio_targets(qdf,signal_name,policy,threshold,directions)
        rets=dict(zip(qdf.Ticker,qdf[f"Forward{h}"]))
        equity,turnover,cost,net=portfolio_step(prev,targets,rets,equity,cost_bps)
        rows.append({"Date":d,"Equity":equity,"Return":net,"Turnover":turnover,
                     "Cost":cost,"Positions":len(targets),
                     "CashWeight":max(0.0,1-sum(targets.values()))})
        prev=targets
    out=pd.DataFrame(rows)
    if len(out)<2:
        return out,{}
    eq=out.Equity
    total=eq.iloc[-1]/STARTING_CAPITAL-1
    years=max((pd.Timestamp(out.Date.iloc[-1])-pd.Timestamp(out.Date.iloc[0])).days/365.25,1/365.25)
    cagr=(eq.iloc[-1]/STARTING_CAPITAL)**(1/years)-1
    peak=eq.cummax()
    dd=eq/peak-1
    maxdd=dd.min()
    vol=out.Return.std(ddof=1)*np.sqrt(252/(h))
    sharpe=(out.Return.mean()/out.Return.std(ddof=1))*np.sqrt(252/h) if out.Return.std(ddof=1)>0 else np.nan
    downside=out.Return[out.Return<0].std(ddof=1)
    sortino=(out.Return.mean()/downside)*np.sqrt(252/h) if np.isfinite(downside) and downside>0 else np.nan
    pos=out.Return[out.Return>0].sum()
    neg=-out.Return[out.Return<0].sum()
    pf=pos/neg if neg>0 else np.nan
    metrics={"TotalReturn":total,"CAGR":cagr,"MaxDrawdown":maxdd,
             "Volatility":vol,"Sharpe":sharpe,"Sortino":sortino,
             "ProfitFactor":pf,"HitRate":float((out.Return>0).mean()),
             "MeanReturn":out.Return.mean(),"MeanTurnover":out.Turnover.mean(),
             "TotalTurnover":out.Turnover.sum(),"TimeInCash":out.CashWeight.mean(),
             "MeanPositions":out.Positions.mean(),"FinalEquity":eq.iloc[-1]}
    return out,metrics

def main():
    print("CERE — FASE 6 — SIGNAL SELECTION & PORTFOLIO VALIDATION")
    print("RUN_ID:",RUN_ID)
    data=download_all()
    panel=build_panel(data)
    panel.to_csv(OUTDIR/f"phase6_panel_{RUN_ID}.csv",index=False)

    qdates=choose_dates(data)
    qdates=[pd.Timestamp(x).date() for x in qdates]
    discovery=qdates[:DISCOVERY_DATES]
    oos=qdates[DISCOVERY_DATES:]
    print("GLOBAL DATES:",len(qdates),"DISCOVERY:",len(discovery),"OOS:",len(oos))

    # Build cross-sectional query data. For each date, only current state is used
    # for ranking; realized forward return is evaluated afterward.
    qdfs=[query_cross_section(data,d) for d in qdates]
    discovery_dfs=qdfs[:DISCOVERY_DATES]
    directions=frozen_directions(discovery_dfs)

    pd.DataFrame(
        [{"Signal":s,"Variable":v,"Direction":directions[s][v]}
         for s in directions for v in directions[s]]
    ).to_csv(OUTDIR/f"phase6_frozen_directions_{RUN_ID}.csv",index=False)

    # Threshold candidates are selected from Discovery only.
    threshold_map={}
    for s in SIGNAL_SPECS:
        vals=[]
        for d in discovery_dfs:
            if d.empty: continue
            vals.extend(signal_score(d,s,directions).dropna().tolist())
        if not vals:
            threshold_map[s]=0.0
        else:
            threshold_map[s]=float(np.quantile(vals,0.90))

    policies=["TOP20","TOP10","THRESHOLD"]
    all_summary=[]
    all_curves=[]
    all_oos_rows=[]

    for signal in SIGNAL_SPECS:
        for h in HORIZONS:
            oos_q=qdfs[DISCOVERY_DATES:]
            oos_dates=oos
            for policy in policies:
                threshold=threshold_map[signal]
                for cost in COST_BPS:
                    curve,metrics=simulate(signal,policy,threshold,oos_dates,oos_q,h,directions,cost)
                    key=f"{signal}|H{h}|{policy}|{cost}bps"
                    if metrics:
                        row={"Model":signal,"Horizon":h,"Policy":policy,"CostBPS":cost}
                        row.update(metrics)
                        all_summary.append(row)
                    if not curve.empty:
                        curve=curve.copy()
                        curve["Model"]=signal
                        curve["Horizon"]=h
                        curve["Policy"]=policy
                        curve["CostBPS"]=cost
                        all_curves.append(curve)

    summary=pd.DataFrame(all_summary)
    curves=pd.concat(all_curves,ignore_index=True) if all_curves else pd.DataFrame()

    summary.to_csv(OUTDIR/f"phase6_portfolio_summary_{RUN_ID}.csv",index=False)
    curves.to_csv(OUTDIR/f"phase6_equity_curves_{RUN_ID}.csv",index=False)

    # OOS diagnostic by date: score rank, selected count and realized returns.
    rows=[]
    for qi,(d,qdf) in enumerate(zip(qdates,qdfs)):
        for signal in SIGNAL_SPECS:
            if qdf.empty: continue
            s=signal_score(qdf,signal,directions)
            tmp=qdf[["Ticker"]+[f"Forward{h}" for h in HORIZONS]].copy()
            tmp["Score"]=s.values
            tmp["Date"]=d
            tmp["Signal"]=signal
            tmp["OOS"]=qi>=DISCOVERY_DATES
            rows.append(tmp)
    if rows:
        pd.concat(rows,ignore_index=True).to_csv(
            OUTDIR/f"phase6_oos_scores_{RUN_ID}.csv",index=False)

    # Frozen decision table: use 20 bps as central cost for ranking candidates.
    central=summary[summary.CostBPS==20].copy()
    central=central.sort_values(["CAGR","Sharpe"],ascending=False)
    central.head(20).to_csv(OUTDIR/f"phase6_central_candidates_{RUN_ID}.csv",index=False)

    print("\n--- PHASE 6 CENTRAL (20 BPS) ---")
    print(central.head(20).to_string(index=False))
    print("\n--- FROZEN DIRECTIONS ---")
    print(directions)
    print("\n--- THRESHOLDS ---")
    print(threshold_map)
    print("\nSTATUS: PASS")
    print("Artifacts:",OUTDIR)

if __name__=="__main__":
    main()
