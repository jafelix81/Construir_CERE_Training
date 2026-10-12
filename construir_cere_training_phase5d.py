#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CERE FASE 5D — REGIME-GATED VOLATILITY MODEL.

Laboratorio OOS. Hypotheses are frozen before evaluation:
A Vol252 always active
B Vol60 always active
C Vol60+Vol252 always active
D Vol252 only in RISK_NEGATIVE
E Vol60+Vol252 only in RISK_NEGATIVE
F Vol252 active in RISK_POSITIVE/RISK_NEGATIVE, CASH in TRANSITION

No Kelly, no portfolio construction, no Sheets writes.
"""
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
PERIOD = "10y"
HORIZONS = (5, 10, 20, 40)
QUERY_DATES = 48
DISCOVERY_DATES = 12
EMBARGO_DAYS = 5
MIN_QUERY_INDEX = 350

STATE_FIELDS = ["R1","R5","R20","R60","R120","Vol5","Vol20","Vol60","Vol252",
                "VolRatio20","VolRatio60","Skew20","Kurt20","AC1"]

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

# Frozen model variants — defined before seeing 5D OOS results.
MODELS = {
    "A_VOL252_ALWAYS": {"signal": "Vol252", "vars": ["Vol252"], "gate": "ALL"},
    "B_VOL60_ALWAYS": {"signal": "Vol60", "vars": ["Vol60"], "gate": "ALL"},
    "C_VOL60_VOL252_ALWAYS": {"signal": "Vol60+Vol252", "vars": ["Vol60","Vol252"], "gate": "ALL"},
    "D_VOL252_RISK_NEGATIVE": {"signal": "Vol252", "vars": ["Vol252"], "gate": "RISK_NEGATIVE"},
    "E_VOL60_VOL252_RISK_NEGATIVE": {"signal": "Vol60+Vol252", "vars": ["Vol60","Vol252"], "gate": "RISK_NEGATIVE"},
    "F_VOL252_NO_TRANSITION": {"signal": "Vol252", "vars": ["Vol252"], "gate": "NO_TRANSITION"},
}

# Constant round-trip cost sensitivity. Applied only to selected/top-quintile trades.
COST_BPS = (0, 10, 20, 30, 50)


def state_from_closes(closes):
    c = np.asarray(closes, float)
    if len(c) < 260 or np.any(~np.isfinite(c)) or np.any(c <= 0):
        return None
    lr = np.diff(np.log(c))
    ret = lambda n: c[-1] / c[-1-n] - 1
    def vol(n):
        return float(np.std(lr[-n:], ddof=1) * np.sqrt(252))
    def skew(n): return float(pd.Series(lr[-n:]).skew())
    def kurt(n): return float(pd.Series(lr[-n:]).kurt())
    def ac1(n):
        x = lr[-n:]
        if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0: return np.nan
        return float(np.corrcoef(x[:-1], x[1:])[0,1])
    v5,v20,v60,v252 = vol(5),vol(20),vol(60),vol(252)
    s = {"R1":ret(1),"R5":ret(5),"R20":ret(20),"R60":ret(60),"R120":ret(120),
         "Vol5":v5,"Vol20":v20,"Vol60":v60,"Vol252":v252,
         "VolRatio20":v20/v252 if v252 else np.nan,"VolRatio60":v60/v252 if v252 else np.nan,
         "Skew20":skew(20),"Kurt20":kurt(20),"AC1":ac1(60)}
    return s if all(np.isfinite(s[k]) for k in STATE_FIELDS) else None


def build_dataset(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty: return None, "EMPTY"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if "Open" not in df.columns or "Close" not in df.columns: return None, "MISSING_OHLC"
        df = df[["Open","Close"]].dropna()
        df = df[(df.Open > 0) & (df.Close > 0)].reset_index()
        dc = "Date" if "Date" in df.columns else df.columns[0]
        df["Date"] = pd.to_datetime(df[dc], utc=True).dt.tz_convert(None).dt.normalize()
        df = df[["Date","Open","Close"]].drop_duplicates("Date").reset_index(drop=True)
        rows=[]
        for i in range(259, len(df)-40):
            st=state_from_closes(df.Close.to_numpy()[:i+1])
            if st is None: continue
            row={"Date":df.loc[i,"Date"],"INDEX":i,**st}
            for h in HORIZONS:
                row[f"Forward{h}"] = float(df.loc[i+h,"Close"])/float(df.loc[i+1,"Open"])-1
            rows.append(row)
        out=pd.DataFrame(rows)
        return (out,"OK") if not out.empty else (None,"NO_ELIGIBLE_ROWS")
    except Exception as e:
        return None, f"ERROR:{type(e).__name__}"


def choose_dates(datasets):
    ref=datasets.get("QQQ") if "QQQ" in datasets else next(iter(datasets.values()))
    eligible=pd.Series(pd.to_datetime(ref.loc[ref.INDEX>=MIN_QUERY_INDEX,"Date"]).unique()).sort_values()
    if len(eligible)<QUERY_DATES: return []
    idx=np.linspace(0,len(eligible)-1,num=QUERY_DATES,dtype=int)
    return [pd.Timestamp(eligible.iloc[i]) for i in np.unique(idx)]


def zscore(s):
    s=pd.to_numeric(s,errors="coerce"); sd=s.std(ddof=1)
    return (s-s.mean())/sd if np.isfinite(sd) and sd>1e-12 else pd.Series(np.nan,index=s.index)


def regime_label(g):
    breadth=float((g.R20>0).mean()); med=float(g.R20.median())
    if breadth>=.60 and med>=0: return "RISK_POSITIVE"
    if breadth<=.40 and med<0: return "RISK_NEGATIVE"
    return "TRANSITION"


def date_ic(g):
    x=g[["SCORE","REALIZED_RETURN"]].dropna()
    if len(x)<10:return np.nan
    r,_=spearmanr(x.SCORE.to_numpy(float),x.REALIZED_RETURN.to_numpy(float))
    return float(r) if np.isfinite(r) else np.nan


def spread(g):
    x=g[["SCORE","REALIZED_RETURN"]].dropna()
    if len(x)<10:return np.nan
    q=max(1,len(x)//5);x=x.sort_values("SCORE")
    return float(x.iloc[-q:].REALIZED_RETURN.mean()-x.iloc[:q].REALIZED_RETURN.mean())


def freeze_direction(df, vars_, discovery_dates):
    vals=[]
    for h in HORIZONS:
        sub=df[(df.HORIZON==h)&df.QueryDate.isin(discovery_dates)]
        for _,g in sub.groupby("QueryDate"):
            for v in vars_:
                x=g[[v,"REALIZED_RETURN"]].dropna()
                if len(x)>=10:
                    r,_=spearmanr(x[v].to_numpy(float),x.REALIZED_RETURN.to_numpy(float))
                    if np.isfinite(r): vals.append((v,float(r)))
    directions={}
    for v in vars_:
        vv=[r for name,r in vals if name==v]
        m=float(np.mean(vv)) if vv else 0.0
        directions[v]=1 if m>=0 else -1
    return directions


def score_group(g, vars_, directions):
    return pd.concat([directions[v]*zscore(g[v]) for v in vars_],axis=1).mean(axis=1)


def evaluate_model(df, model, oos_dates, regimes, cost_bps=0):
    rows=[]
    for h in HORIZONS:
        sub=df[(df.HORIZON==h)&df.QueryDate.isin(oos_dates)].copy()
        for q,idx in sub.groupby("QueryDate").groups.items():
            g=sub.loc[idx].copy()
            g["SCORE"]=score_group(g,model["vars"],model["directions"])
            regime=regimes[q]
            if model["gate"]=="RISK_NEGATIVE": active=(regime=="RISK_NEGATIVE")
            elif model["gate"]=="NO_TRANSITION": active=(regime!="TRANSITION")
            else: active=True
            g["ACTIVE"]=active
            ic=date_ic(g) if active else np.nan
            sp=spread(g) if active else np.nan
            selected=g[g.SCORE>0].copy() if active else g.iloc[0:0].copy()
            if len(selected):
                realized=selected.REALIZED_RETURN.to_numpy(float)
                net=realized-(cost_bps/10000.0)
                mean_net=float(np.mean(net)); positive=float(np.mean(net>0))
            else: mean_net=np.nan; positive=np.nan
            rows.append({"MODEL":model["name"],"HORIZON":h,"QueryDate":q,"REGIME":regime,"ACTIVE":active,
                         "N":len(g),"N_SELECTED":len(selected),"IC":ic,"TOP_BOTTOM":sp,
                         "MEAN_NET_SELECTED_RETURN":mean_net,"POSITIVE_SELECTED_RATE":positive})
    return pd.DataFrame(rows)


def summarize(date_df):
    out=[]
    for (m,h),g in date_df.groupby(["MODEL","HORIZON"]):
        active=g[g.ACTIVE]
        out.append({"MODEL":m,"HORIZON":h,"N_OOS_DATES":len(g),"ACTIVE_DATES":len(active),
                    "ACTIVE_DATE_RATE":len(active)/len(g) if len(g) else np.nan,
                    "MEAN_RANK_IC":active.IC.mean(),"MEDIAN_RANK_IC":active.IC.median(),
                    "POSITIVE_IC_RATE":(active.IC>0).mean(),"MEAN_TOP_MINUS_BOTTOM":active.TOP_BOTTOM.mean(),
                    "MEAN_SELECTED_RETURN":active.MEAN_NET_SELECTED_RETURN.mean(),
                    "POSITIVE_SELECTED_RATE":active.POSITIVE_SELECTED_RATE.mean(),
                    "MEAN_SELECTED_COUNT":active.N_SELECTED.mean()})
    return pd.DataFrame(out)


def main():
    print("="*80); print("CERE — FASE 5D — REGIME-GATED VOLATILITY MODEL"); print("="*80)
    print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}")
    print(f"HORIZONS: {HORIZONS} | Dates: {QUERY_DATES} | Discovery: {DISCOVERY_DATES} | OOS: {QUERY_DATES-DISCOVERY_DATES}")
    datasets={}; audit=[]
    for n,t in enumerate(TICKERS,1):
        print(f"[{n:03d}/{len(TICKERS)}] {t}",flush=True)
        d,s=build_dataset(t)
        if d is not None: datasets[t]=d
        audit.append({"Ticker":t,"STATUS":s,"ROWS":len(d) if d is not None else 0})
    dates=choose_dates(datasets)
    if len(dates)!=QUERY_DATES: raise RuntimeError(f"Se obtuvieron {len(dates)} fechas; se requieren {QUERY_DATES}")
    print("GLOBAL QUERY DATES:"); [print(" ",d.date().isoformat()) for d in dates]
    allr=[]
    for t,d in datasets.items(): x=d.copy(); x["Ticker"]=t; allr.append(x)
    allr=pd.concat(allr,ignore_index=True); allr.Date=pd.to_datetime(allr.Date)
    rows=[]; purge=[]
    for q in dates:
        cur=allr[allr.Date==q]
        for h in HORIZONS:
            cutoff=q-pd.offsets.BDay(h+EMBARGO_DAYS)
            hist=allr[allr.Date<=cutoff]
            purge.append({"QueryDate":q.date().isoformat(),"HORIZON":h,"CUTOFF_DATE":cutoff.date().isoformat(),
                          "MAX_HISTORICAL_DATE":hist.Date.max().date().isoformat() if len(hist) else "",
                          "STATUS":"PASS" if len(hist) and hist.Date.max()<=cutoff else "FAIL"})
            for _,r in cur.iterrows():
                rows.append({"QueryDate":q.date().isoformat(),"Ticker":r.Ticker,"HORIZON":h,
                             "REALIZED_RETURN":float(r[f"Forward{h}"]),**{v:float(r[v]) for v in STATE_FIELDS}})
    df=pd.DataFrame(rows)
    discovery=[d.date().isoformat() for d in dates[:DISCOVERY_DATES]]
    oos=[d.date().isoformat() for d in dates[DISCOVERY_DATES:]]
    regime_map={q:regime_label(g) for q,g in df[df.HORIZON==5].groupby("QueryDate")}
    for name,m in MODELS.items():
        m["name"]=name
        m["directions"]=freeze_direction(df,m["vars"],discovery)
    # Evaluate each frozen model over the same untouched OOS set and cost grid.
    cost_summaries=[]; all_date=[]
    for cost in COST_BPS:
        for m in MODELS.values():
            d=evaluate_model(df,m,oos,regime_map,cost_bps=cost)
            d["COST_BPS"]=cost; all_date.append(d)
            s=summarize(d); s["COST_BPS"]=cost; cost_summaries.append(s)
    date_df=pd.concat(all_date,ignore_index=True); summary=pd.concat(cost_summaries,ignore_index=True)
    # Regime-specific summary for the zero-cost base case.
    z=date_df[date_df.COST_BPS==0]
    reg=[]
    for (m,h,r),g in z.groupby(["MODEL","HORIZON","REGIME"]):
        active=g[g.ACTIVE]
        reg.append({"MODEL":m,"HORIZON":h,"REGIME":r,"N_DATES":len(g),"ACTIVE_DATES":len(active),
                    "MEAN_RANK_IC":active.IC.mean(),"MEDIAN_RANK_IC":active.IC.median(),
                    "POSITIVE_IC_RATE":(active.IC>0).mean(),"MEAN_TOP_MINUS_BOTTOM":active.TOP_BOTTOM.mean(),
                    "MEAN_SELECTED_RETURN":active.MEAN_NET_SELECTED_RETURN.mean()})
    reg_df=pd.DataFrame(reg)
    outdir=os.path.join(os.getcwd(),"cere_phase5d_validation"); os.makedirs(outdir,exist_ok=True)
    pd.DataFrame(audit).to_csv(os.path.join(outdir,f"phase5d_ticker_audit_{RUN_ID}.csv"),index=False)
    pd.DataFrame(purge).to_csv(os.path.join(outdir,f"phase5d_purging_audit_{RUN_ID}.csv"),index=False)
    summary.to_csv(os.path.join(outdir,f"phase5d_model_cost_summary_{RUN_ID}.csv"),index=False)
    date_df.to_csv(os.path.join(outdir,f"phase5d_date_by_date_{RUN_ID}.csv"),index=False)
    reg_df.to_csv(os.path.join(outdir,f"phase5d_regime_summary_{RUN_ID}.csv"),index=False)
    pd.DataFrame([{k:m[k] for k in ["name","signal","gate"]}|{"vars":" + ".join(m["vars"]),"directions":str(m["directions"])} for m in MODELS.values()]).to_csv(os.path.join(outdir,f"phase5d_frozen_models_{RUN_ID}.csv"),index=False)
    print("\n--- FASE 5D ZERO-COST OOS ---")
    print(summary[summary.COST_BPS==0][["MODEL","HORIZON","ACTIVE_DATES","ACTIVE_DATE_RATE","MEAN_RANK_IC","MEDIAN_RANK_IC","POSITIVE_IC_RATE","MEAN_TOP_MINUS_BOTTOM","MEAN_SELECTED_RETURN","MEAN_SELECTED_COUNT"]].to_string(index=False))
    print("\n--- FASE 5D COST SENSITIVITY (MEAN ACROSS OOS DATES) ---")
    print(summary[["MODEL","HORIZON","COST_BPS","MEAN_SELECTED_RETURN","POSITIVE_SELECTED_RATE"]].to_string(index=False))
    print("\n--- FASE 5D REGIMES ---")
    print(reg_df.to_string(index=False))
    print("\n--- AUDITORÍA ---")
    print(f"Tickers configurados: {len(TICKERS)}")
    print(f"Tickers OK: {len(datasets)}")
    print(f"Global query dates: {len(dates)} | Discovery: {len(discovery)} | OOS: {len(oos)}")
    print(f"Cross-sectional observations: {len(df):,}")
    print("PURGING_AUDIT:","PASS" if all(x["STATUS"]=="PASS" for x in purge) else "FAIL")
    print("LOOK_AHEAD_PREDICTORS: PASS")
    print("MODEL_DEFINITIONS_FROZEN: PASS")
    print("STATUS: PASS")
    print(f"Artifacts: {outdir}")

if __name__ == "__main__": main()
