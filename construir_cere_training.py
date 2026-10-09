"""CERE_TRAINING Fase 2A — DRY RUN.
t = cierre; estado usa datos <= t; entrada Open(t+1); salida Close(t+H).
No escribe Google Sheets y no calcula EV/vecinos/Student-t/Kelly.
"""
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

PERIOD="10y"
H=(5,10,20,40)
MIN_STATE=260
RETRIES=2
OUT=Path("cere_training_dry_run"); OUT.mkdir(exist_ok=True)
RUN_ID=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
TICKERS=sorted(set(['AAPL', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'INTC', 'QCOM', 'AVGO', 'NFLX', 'CSCO', 'AMAT', 'MU', 'PANW', 'SNPS', 'CDNS', 'PLTR', 'PYPL', 'SHOP', 'NET', 'DDOG', 'CRWD', 'OKTA', 'ZS', 'MDB', 'TEAM', 'WDAY', 'NOW', 'SNOW', 'ZM', 'DOCU', 'ROKU', 'TWLO', 'PINS', 'SNAP', 'MTCH', 'TSM', 'ASML', 'LRCX', 'KLAC', 'NXPI', 'TXN', 'ADI', 'MCHP', 'ON', 'MRVL', 'TER', 'ENPH', 'SEDG', 'FSLR', 'FLEX', 'COIN', 'MARA', 'RIOT', 'SOFI', 'AFRM', 'UPST', 'HOOD', 'DKNG', 'NU', 'MELI', 'SE', 'V', 'MA', 'AXP', 'REGN', 'BIIB', 'GILD', 'AMGN', 'VRTX', 'ILMN', 'ALGN', 'MRNA', 'BNTX', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'SBUX', 'MDLZ', 'CHTR', 'TMUS', 'CMCSA', 'EA', 'TTWO', 'ABNB', 'BKNG', 'EXPE', 'TRIP', 'PDD', 'JD', 'BABA', 'BIDU', 'NIO', 'LI', 'XPEV', 'LCID', 'RIVN', 'QS', 'PLUG', 'RUN', 'CHPT', 'BLNK', 'BE', 'FCEL', 'SPWR', 'CAT', 'DE', 'HON', 'GE', 'MMM', 'LMT', 'BA', 'NOC', 'GD', 'RTX', 'UPS', 'FDX', 'CSX', 'NSC', 'UNP', 'WM', 'RSG', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'BX', 'KKR', 'APO', 'TROW', 'BEN', 'STT', 'NTRS', 'SCHW', 'AMTD', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'EL', 'CL', 'KMB', 'GIS', 'MNST', 'CELH', 'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'OXY', 'DVN', 'APA', 'FCX', 'NEM', 'NUE', 'STLD', 'AA', 'CLF', 'T', 'VZ', 'DIS', 'WBD', 'PARA', 'FOXA', 'NWSA', 'LYV', 'SIFY', 'VOD', 'TELFY', 'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'MOS', 'CF', 'NTR', 'FMC', 'VMC', 'MLM', 'CX', 'LEN', 'DHI', 'PHM', 'PLD', 'AMT', 'CCI', 'EQIX', 'O', 'SPG', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'VICI', 'DLR', 'SBAC', 'WY', 'BXP', 'ISRG', 'SYK', 'ZBH', 'EW', 'BSX', 'MDT', 'ABT', 'BMY', 'PFE', 'JNJ', 'LLY', 'NVO', 'AZN', 'SNY', 'GSK', 'TAK', 'Z', 'ZG', 'OPEN', 'COMP', 'W', 'CVNA', 'CHWY', 'JMIA', 'EBAY', 'ETSY', 'WAY', 'PAGS', 'STNE', 'PAYS', 'FLYW', 'LOT', 'LPRO', 'OPFI', 'CACC', 'OMF', 'FCF', 'GPRO', 'FITB', 'HBAN', 'KEY', 'RF', 'CFG', 'MTB', 'ZION', 'TFC', 'AX', 'CUBI', 'HOMB', 'OZK', 'FNB', 'ASB', 'VLY', 'UMBF', 'BOKF', 'EGBN', 'WBS', 'CATY', 'IWM', 'QQQ', 'VEA', 'VWO', 'IEFA', 'EEM', 'VNQ', 'GLD', 'USO', 'UNG', 'OIH', 'XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'SMH', 'SOXX', 'XBI', 'KRE', 'JETS', 'ARKK', 'ARKW', 'ARKG', 'ARKF', 'ARKQ', 'BITO', 'RUM', 'DJT', 'PSNY', 'MP', 'SMCI', 'DELL', 'ANET', 'VRT', 'LITE', 'CLS', 'MOD', 'CRDO', 'ALAB', 'GLW', 'CEG', 'VST', 'GEV', 'ETN', 'PWR', 'APP', 'AXON', 'RDDT', 'DUOL', 'CAVA', 'TEM', 'TMDX', 'IREN', 'RKLB', 'ASTS', 'HUT', 'WULF', 'CIFR', 'ATI', 'XYZ', 'HAPN']))
STATE=["R1","R5","R20","R60","R120","Vol5","Vol20","Vol60","Vol252","VolRatio20","VolRatio60","Skew20","Kurt20","AC1"]

def col(df,name,ticker):
    if df is None or df.empty: return None
    try:
        if isinstance(df.columns,pd.MultiIndex):
            if name in df.columns.get_level_values(0):
                x=df[name]
                if isinstance(x,pd.DataFrame):
                    return x[ticker] if ticker in x.columns else (x.iloc[:,0] if len(x.columns)==1 else None)
                return x
            if ticker in df.columns.get_level_values(0):
                x=df[ticker]; return x[name] if name in x.columns else None
            return None
        return df[name] if name in df.columns else None
    except Exception: return None

def clean(s):
    s=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna().copy()
    if isinstance(s.index,pd.DatetimeIndex):
        if s.index.tz is not None: s.index=s.index.tz_localize(None)
        s.index=s.index.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()

def state(close,i):
    p=close.iloc[:i+1].to_numpy(float)
    lr=np.diff(np.log(p))
    def ret(n): return np.nan if len(p)<=n or p[-n-1]<=0 else p[-1]/p[-n-1]-1
    def vol(n): return np.nan if len(lr)<n else np.std(lr[-n:],ddof=1)*np.sqrt(252)
    v5,v20,v60,v252=[vol(x) for x in (5,20,60,252)]
    s=pd.Series(lr)
    ac=s.iloc[-60:].autocorr(1) if len(lr)>=61 else np.nan
    return dict(R1=ret(1),R5=ret(5),R20=ret(20),R60=ret(60),R120=ret(120),
                Vol5=v5,Vol20=v20,Vol60=v60,Vol252=v252,
                VolRatio20=v20/v252 if pd.notna(v20) and pd.notna(v252) and v252>0 else np.nan,
                VolRatio60=v60/v252 if pd.notna(v60) and pd.notna(v252) and v252>0 else np.nan,
                Skew20=s.iloc[-20:].skew() if len(lr)>=20 else np.nan,
                Kurt20=s.iloc[-20:].kurt() if len(lr)>=20 else np.nan,
                AC1=ac)

def get_data(ticker,bulk):
    o,c=col(bulk,"Open",ticker),col(bulk,"Close",ticker)
    if o is not None and c is not None: return clean(o),clean(c),"BULK"
    for n in range(RETRIES):
        try:
            d=yf.download(ticker,period=PERIOD,interval="1d",auto_adjust=True,progress=False,threads=False)
            o,c=col(d,"Open",ticker),col(d,"Close",ticker)
            if o is not None and c is not None: return clean(o),clean(c),"RETRY"
        except Exception: pass
        time.sleep(2)
    return None,None,"FAILED"

def build(ticker,bulk):
    o,c,src=get_data(ticker,bulk)
    if o is None or c is None: return pd.DataFrame(),dict(Ticker=ticker,STATUS="NO_DATA",OBSERVATIONS=0)
    d=pd.concat([o.rename("Open"),c.rename("Close")],axis=1,join="inner").dropna().sort_index()
    if len(d)<MIN_STATE: return pd.DataFrame(),dict(Ticker=ticker,STATUS="INSUFFICIENT_DATA",OBSERVATIONS=len(d))
    first=MIN_STATE-1; last=len(d)-41
    rows=[]
    for i in range(first,last+1):
        st=state(d.Close,i)
        if not all(pd.notna(st[x]) and np.isfinite(st[x]) for x in STATE): continue
        r={"Ticker":ticker,"Date":d.index[i].strftime("%Y-%m-%d"),
            "EntryDate":d.index[i+1].strftime("%Y-%m-%d")}
        r.update(st)
        for h in H:
            r[f"ExitDate{h}"]=d.index[i+h].strftime("%Y-%m-%d")
            r[f"Forward{h}"]=d.Close.iloc[i+h]/d.Open.iloc[i+1]-1
        rows.append(r)
    out=pd.DataFrame(rows)
    return out,dict(Ticker=ticker,STATUS="OK" if len(out) else "NO_VALID_OBSERVATIONS",
                    OBSERVATIONS=len(out),FIRST_DATE=out.Date.iloc[0] if len(out) else "",
                    LAST_DATE=out.Date.iloc[-1] if len(out) else "",DATA_SOURCE=src)

def audit(df):
    if df.empty: return dict(LOOK_AHEAD_VIOLATIONS=0,DUPLICATES=0,INVALID_DATES=0,INVALID_STATES=0,INVALID_FORWARDS=0)
    dup=int(df.duplicated(["Ticker","Date"]).sum()); bad_dates=bad_states=bad_fw=viol=0
    t=pd.to_datetime(df.Date,errors="coerce"); e=pd.to_datetime(df.EntryDate,errors="coerce")
    bad_dates+=int(t.isna().sum()+e.isna().sum()+(e<=t).sum())
    for x in STATE: bad_states+=int((~np.isfinite(pd.to_numeric(df[x],errors="coerce"))).sum())
    for h in H:
        x=pd.to_datetime(df[f"ExitDate{h}"],errors="coerce")
        bad_dates+=int(x.isna().sum()+(x<=t).sum()+(x<=e).sum())
        bad_fw+=int((~np.isfinite(pd.to_numeric(df[f"Forward{h}"],errors="coerce"))).sum())
        viol+=int((x<=t).sum())
    for _,g in df.groupby("Ticker"):
        if not pd.to_datetime(g.Date).is_monotonic_increasing: bad_dates+=1
    return dict(LOOK_AHEAD_VIOLATIONS=viol,DUPLICATES=dup,INVALID_DATES=bad_dates,INVALID_STATES=bad_states,INVALID_FORWARDS=bad_fw)

def main():
    print("="*80); print("CERE_TRAINING — FASE 2A — DRY RUN"); print("="*80)
    print("RUN_ID:",RUN_ID,"| Tickers:",len(TICKERS),"| Period:",PERIOD)
    bulk=yf.download(TICKERS,period=PERIOD,interval="1d",auto_adjust=True,progress=False,group_by="column",threads=True)
    frames=[]; statuses=[]
    for n,t in enumerate(TICKERS,1):
        print(f"[{n:03d}/{len(TICKERS):03d}] {t}")
        try:
            x,s=build(t,bulk)
            if not x.empty: frames.append(x)
            statuses.append(s)
        except Exception as ex:
            statuses.append(dict(Ticker=t,STATUS="ERROR",OBSERVATIONS=0,ERROR=f"{type(ex).__name__}: {ex}"))
    if not frames: raise RuntimeError("No se produjo dataset.")
    df=pd.concat(frames,ignore_index=True).sort_values(["Ticker","Date"]).reset_index(drop=True)
    a=audit(df)
    print("\n--- AUDITORÍA ---")
    print("Tickers configurados:",len(TICKERS))
    print("Tickers con dataset:",df.Ticker.nunique())
    print("Observaciones:",f"{len(df):,}")
    for h in H: print(f"Forward{h} válidos:",f"{df[f'Forward{h}'].notna().sum():,}")
    for k,v in a.items(): print(f"{k}:",v)
    sample=df[["Ticker","Date","EntryDate","ExitDate5","ExitDate10","ExitDate20","ExitDate40"]].head(5)
    print("\n--- MUESTRA TEMPORAL ---"); print(sample.to_string(index=False))
    df.to_csv(OUT/f"cere_training_dry_run_{RUN_ID}.csv",index=False)
    pd.DataFrame(statuses).to_csv(OUT/f"cere_training_ticker_audit_{RUN_ID}.csv",index=False)
    summary=dict(RUN_ID=RUN_ID,CAPTURED_AT_UTC=datetime.now(timezone.utc).isoformat(),
                 TICKERS_CONFIGURED=len(TICKERS),TICKERS_WITH_DATASET=df.Ticker.nunique(),
                 OBSERVATIONS=len(df),**a,PASS=all(v==0 for v in a.values()))
    pd.DataFrame([summary]).to_csv(OUT/f"cere_training_summary_{RUN_ID}.csv",index=False)
    if any(a.values()): raise RuntimeError("ABORTADO: auditoría con observaciones.")
    print("\n✅ AUDITORÍA CERE_TRAINING DRY RUN COMPLETADA SIN OBSERVACIONES.")
    print("ℹ️ No se escribió Google Sheets.")

if __name__=="__main__": main()
