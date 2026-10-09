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
OUT=Path("cere_training_phase2b"); OUT.mkdir(exist_ok=True)
SAMPLE_STEP=5  # conservar 1 de cada 5 observaciones por ticker para el dataset operativo compacto

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

def compact_sample(df, step=SAMPLE_STEP):
    """
    Reduce el dataset para almacenamiento/consumo operativo sin cambiar la
    definición de estados ni forwards. La selección es determinista y se hace
    dentro de cada ticker, después de ordenar por fecha.
    """
    x = df.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    x["_ROW_TICKER"] = x.groupby("Ticker").cumcount()
    compact = x[x["_ROW_TICKER"] % step == 0].drop(columns="_ROW_TICKER").copy()
    return compact.reset_index(drop=True)

def distribution_summary(df, label):
    out = {"DATASET": label, "ROWS": len(df), "TICKERS": df.Ticker.nunique()}
    for h in H:
        s = pd.to_numeric(df[f"Forward{h}"], errors="coerce")
        out[f"F{h}_MEAN"] = float(s.mean())
        out[f"F{h}_MEDIAN"] = float(s.median())
        out[f"F{h}_STD"] = float(s.std(ddof=1))
        out[f"F{h}_P10"] = float(s.quantile(0.10))
        out[f"F{h}_P90"] = float(s.quantile(0.90))
        out[f"F{h}_POSITIVE"] = float((s > 0).mean())
    return out

def compare_full_compact(full, compact):
    rows = []
    for h in H:
        for stat, fn in [
            ("MEAN", lambda s: s.mean()),
            ("MEDIAN", lambda s: s.median()),
            ("STD", lambda s: s.std(ddof=1)),
            ("P10", lambda s: s.quantile(0.10)),
            ("P90", lambda s: s.quantile(0.90)),
            ("POSITIVE", lambda s: (s > 0).mean()),
        ]:
            a = fn(pd.to_numeric(full[f"Forward{h}"], errors="coerce"))
            b = fn(pd.to_numeric(compact[f"Forward{h}"], errors="coerce"))
            rows.append({
                "HORIZON": h,
                "STAT": stat,
                "FULL": float(a),
                "COMPACT": float(b),
                "ABS_DIFF": float(abs(a-b)),
            })
    return pd.DataFrame(rows)

def main():
    print("="*80)
    print("CERE_TRAINING — FASE 2B — COMPACT DATASET DRY RUN")
    print("="*80)
    print("RUN_ID:", RUN_ID, "| Tickers:", len(TICKERS), "| Period:", PERIOD)
    print("SAMPLE_STEP:", SAMPLE_STEP, "(1 de cada 5 observaciones por ticker)")

    bulk=yf.download(
        TICKERS, period=PERIOD, interval="1d", auto_adjust=True,
        progress=False, group_by="column", threads=True
    )

    frames=[]; statuses=[]
    for n,t in enumerate(TICKERS,1):
        print(f"[{n:03d}/{len(TICKERS):03d}] {t}")
        try:
            x,s=build(t,bulk)
            if not x.empty: frames.append(x)
            statuses.append(s)
        except Exception as ex:
            statuses.append(dict(
                Ticker=t, STATUS="ERROR", OBSERVATIONS=0,
                ERROR=f"{type(ex).__name__}: {ex}"
            ))

    if not frames:
        raise RuntimeError("No se produjo dataset.")

    df=pd.concat(frames,ignore_index=True).sort_values(
        ["Ticker","Date"]
    ).reset_index(drop=True)

    # Auditoría del dataset completo
    a_full=audit(df)

    # Dataset compacto
    compact=compact_sample(df, SAMPLE_STEP)
    a_compact=audit(compact)

    print("\n--- AUDITORÍA FULL ---")
    print("Tickers:", df.Ticker.nunique())
    print("Observaciones:", f"{len(df):,}")
    for k,v in a_full.items(): print(f"{k}:",v)

    print("\n--- AUDITORÍA COMPACT ---")
    print("Tickers:", compact.Ticker.nunique())
    print("Observaciones:", f"{len(compact):,}")
    for k,v in a_compact.items(): print(f"{k}:",v)

    # Cobertura por ticker
    full_counts=df.groupby("Ticker").size().rename("FULL_ROWS")
    compact_counts=compact.groupby("Ticker").size().rename("COMPACT_ROWS")
    ticker_cov=pd.concat([full_counts,compact_counts],axis=1).fillna(0)
    ticker_cov["COMPRESSION_RATIO"]=ticker_cov["COMPACT_ROWS"]/ticker_cov["FULL_ROWS"]

    # Comparación de distribuciones de forward
    dist=pd.DataFrame([
        distribution_summary(df,"FULL"),
        distribution_summary(compact,"COMPACT")
    ])
    cmp=compare_full_compact(df,compact)

    # Muestra temporal del compacto
    sample=compact[
        ["Ticker","Date","EntryDate","ExitDate5","ExitDate10","ExitDate20","ExitDate40"]
    ].head(10)

    print("\n--- MUESTRA TEMPORAL COMPACT ---")
    print(sample.to_string(index=False))

    print("\n--- COMPARACIÓN DE DISTRIBUCIONES ---")
    print(cmp.to_string(index=False))

    # Archivos de auditoría/resultado.
    # FULL se conserva como artefacto pesado para investigación.
    df.to_csv(OUT/f"cere_training_full_{RUN_ID}.csv", index=False)
    compact.to_csv(OUT/f"cere_training_compact_{RUN_ID}.csv", index=False)
    pd.DataFrame(statuses).to_csv(
        OUT/f"cere_training_ticker_audit_{RUN_ID}.csv", index=False
    )
    ticker_cov.reset_index().to_csv(
        OUT/f"cere_training_ticker_coverage_{RUN_ID}.csv", index=False
    )
    dist.to_csv(OUT/f"cere_training_distribution_{RUN_ID}.csv", index=False)
    cmp.to_csv(OUT/f"cere_training_distribution_compare_{RUN_ID}.csv", index=False)

    summary=dict(
        RUN_ID=RUN_ID,
        CAPTURED_AT_UTC=datetime.now(timezone.utc).isoformat(),
        SAMPLE_STEP=SAMPLE_STEP,
        TICKERS_CONFIGURED=len(TICKERS),
        TICKERS_FULL=df.Ticker.nunique(),
        TICKERS_COMPACT=compact.Ticker.nunique(),
        OBSERVATIONS_FULL=len(df),
        OBSERVATIONS_COMPACT=len(compact),
        COMPRESSION_RATIO=len(compact)/len(df),
        **{f"FULL_{k}":v for k,v in a_full.items()},
        **{f"COMPACT_{k}":v for k,v in a_compact.items()},
    )
    pd.DataFrame([summary]).to_csv(
        OUT/f"cere_training_phase2b_summary_{RUN_ID}.csv", index=False
    )

    if any(a_full.values()) or any(a_compact.values()):
        raise RuntimeError("ABORTADO: auditoría con observaciones.")

    print("\n✅ FASE 2B COMPLETADA SIN OBSERVACIONES.")
    print("ℹ️ Aún NO se escribió Google Sheets.")
    print("ℹ️ El objetivo es decidir si el dataset compacto es suficientemente representativo.")

if __name__=="__main__":
    main()
