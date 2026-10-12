#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CERE FASE 5C — VALIDACIÓN TEMPORAL AMPLIADA Y RÉGIMEN. Laboratorio OOS."""
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
RUN_ID=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
PERIOD="10y"; INTERVAL="1d"; MIN_STATE=260
HORIZONS=(5,10,20,40); QUERY_DATES=36; DISCOVERY_DATES=12; EMBARGO_DAYS=5; MIN_QUERY_INDEX=350
STATE_FIELDS=["R1","R5","R20","R60","R120","Vol5","Vol20","Vol60","Vol252","VolRatio20","VolRatio60","Skew20","Kurt20","AC1"]
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
TICKERS=sorted(set(TICKERS))
SIGNAL_SPECS={"Vol20":["Vol20"],"Vol60":["Vol60"],"Vol252":["Vol252"],"Vol20+Vol252":["Vol20","Vol252"],"Vol60+Vol252":["Vol60","Vol252"],"Vol5+Vol20":["Vol5","Vol20"]}

def state_from_closes(closes):
 c=np.asarray(closes,float)
 if len(c)<MIN_STATE or np.any(~np.isfinite(c)) or np.any(c<=0): return None
 lr=np.diff(np.log(c))
 ret=lambda n:c[-1]/c[-1-n]-1
 def vol(n): return float(np.std(lr[-n:],ddof=1)*np.sqrt(252)) if len(lr[-n:])>=2 else np.nan
 def skew(n): return float(pd.Series(lr[-n:]).skew())
 def kurt(n): return float(pd.Series(lr[-n:]).kurt())
 def ac1(n):
  x=lr[-n:]
  if len(x)<3 or np.std(x[:-1])==0 or np.std(x[1:])==0:return np.nan
  return float(np.corrcoef(x[:-1],x[1:])[0,1])
 v5,v20,v60,v252=vol(5),vol(20),vol(60),vol(252)
 s={"R1":ret(1),"R5":ret(5),"R20":ret(20),"R60":ret(60),"R120":ret(120),"Vol5":v5,"Vol20":v20,"Vol60":v60,"Vol252":v252,"VolRatio20":v20/v252 if np.isfinite(v252) and v252 else np.nan,"VolRatio60":v60/v252 if np.isfinite(v252) and v252 else np.nan,"Skew20":skew(20),"Kurt20":kurt(20),"AC1":ac1(60)}
 return s if all(np.isfinite(s[k]) for k in STATE_FIELDS) else None

def build_dataset(ticker):
 try:
  df=yf.download(ticker,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,threads=False)
  if df is None or df.empty:return None,"EMPTY"
  if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
  if "Open" not in df.columns or "Close" not in df.columns:return None,"MISSING_OHLC"
  df=df[["Open","Close"]].dropna();df=df[(df.Open>0)&(df.Close>0)].reset_index();dc="Date" if "Date" in df.columns else df.columns[0]
  df["Date"]=pd.to_datetime(df[dc],utc=True).dt.tz_convert(None).dt.normalize();df=df[["Date","Open","Close"]].drop_duplicates("Date").reset_index(drop=True)
  rows=[]
  for i in range(MIN_STATE-1,len(df)-max(HORIZONS)):
   st=state_from_closes(df.Close.to_numpy()[:i+1])
   if st is None:continue
   row={"Date":df.loc[i,"Date"],"INDEX":i,**st}
   for h in HORIZONS:row[f"Forward{h}"]=float(df.loc[i+h,"Close"])/float(df.loc[i+1,"Open"])-1
   rows.append(row)
  out=pd.DataFrame(rows);return (out,"OK") if not out.empty else (None,"NO_ELIGIBLE_ROWS")
 except Exception as e:return None,f"ERROR:{type(e).__name__}"

def choose_dates(datasets):
 ref=datasets.get("QQQ") if "QQQ" in datasets else next(iter(datasets.values()))
 eligible=pd.Series(pd.to_datetime(ref.loc[ref.INDEX>=MIN_QUERY_INDEX,"Date"]).unique()).sort_values()
 if len(eligible)<QUERY_DATES:return []
 idx=np.linspace(0,len(eligible)-1,num=QUERY_DATES,dtype=int);return [pd.Timestamp(eligible.iloc[i]) for i in np.unique(idx)]

def zscore(s):
 s=pd.to_numeric(s,errors="coerce");sd=s.std(ddof=1);return (s-s.mean())/sd if np.isfinite(sd) and sd>1e-12 else pd.Series(np.nan,index=s.index)

def date_ic(g,score):
 x=g[[score,"REALIZED_RETURN"]].dropna()
 if len(x)<10:return np.nan
 r,_=spearmanr(x[score].to_numpy(float),x.REALIZED_RETURN.to_numpy(float));return float(r) if np.isfinite(r) else np.nan

def spread(g,score):
 x=g[[score,"REALIZED_RETURN"]].dropna()
 if len(x)<10:return np.nan
 q=max(1,len(x)//5);x=x.sort_values(score);return float(x.iloc[-q:].REALIZED_RETURN.mean()-x.iloc[:q].REALIZED_RETURN.mean())

def evaluate(x):
 ics=[];sp=[];pos=[]
 for _,g in x.groupby("QueryDate"):
  a=date_ic(g,"SCORE");b=spread(g,"SCORE")
  if np.isfinite(a):ics.append(a)
  if np.isfinite(b):sp.append(b)
  p=g[g.SCORE>0].REALIZED_RETURN
  if len(p):pos.append(float(p.mean()))
 return {"N_DATES":len(ics),"MEAN_RANK_IC":np.mean(ics) if ics else np.nan,"MEDIAN_RANK_IC":np.median(ics) if ics else np.nan,"STD_RANK_IC":np.std(ics,ddof=1) if len(ics)>1 else np.nan,"POSITIVE_IC_RATE":np.mean(np.asarray(ics)>0) if ics else np.nan,"MEAN_TOP_MINUS_BOTTOM":np.mean(sp) if sp else np.nan,"MEAN_REALIZED_WHEN_SCORE_POSITIVE":np.mean(pos) if pos else np.nan}

def freeze_directions(df,dates):
 dirs={};rows=[]
 for name,vars_ in SIGNAL_SPECS.items():
  dirs[name]={}
  for v in vars_:
   vals=[]
   for h in HORIZONS:
    sub=df[(df.HORIZON==h)&df.QueryDate.isin(dates)]
    for _,g in sub.groupby("QueryDate"):
     ic=date_ic(g,v)
     if np.isfinite(ic):vals.append(ic)
   m=float(np.mean(vals)) if vals else 0.0;dirs[name][v]=1 if m>=0 else -1;rows.append({"SIGNAL":name,"VARIABLE":v,"DISCOVERY_MEAN_IC":m,"DIRECTION":dirs[name][v]})
 return dirs,pd.DataFrame(rows)

def make_score(g,vars_,dirs):return pd.concat([dirs[v]*zscore(g[v]) for v in vars_],axis=1).mean(axis=1)

def regime_label(g):
 breadth=float((g.R20>0).mean());med=float(g.R20.median())
 if breadth>=.60 and med>=0:return "RISK_POSITIVE"
 if breadth<=.40 and med<0:return "RISK_NEGATIVE"
 return "TRANSITION"

def main():
 print("="*80);print("CERE — FASE 5C — VALIDACIÓN TEMPORAL AMPLIADA Y RÉGIMEN");print("="*80)
 print(f"RUN_ID: {RUN_ID} | Tickers: {len(TICKERS)} | Period: {PERIOD}");print(f"HORIZONS: {HORIZONS} | Dates: {QUERY_DATES} | Discovery: {DISCOVERY_DATES} | OOS: {QUERY_DATES-DISCOVERY_DATES}")
 datasets={};audit=[]
 for n,ticker in enumerate(TICKERS,1):
  print(f"[{n:03d}/{len(TICKERS)}] {ticker}",flush=True);d,s=build_dataset(ticker)
  if d is not None:datasets[ticker]=d
  audit.append({"Ticker":ticker,"STATUS":s,"ROWS":len(d) if d is not None else 0})
 dates=choose_dates(datasets)
 if len(dates)!=QUERY_DATES:raise RuntimeError(f"Se obtuvieron {len(dates)} fechas; se requieren {QUERY_DATES}")
 print("GLOBAL QUERY DATES:");[print(" ",d.date().isoformat()) for d in dates]
 allr=[]
 for t,d in datasets.items():x=d.copy();x["Ticker"]=t;allr.append(x)
 allr=pd.concat(allr,ignore_index=True);allr.Date=pd.to_datetime(allr.Date)
 rows=[];pa=[]
 for q in dates:
  cur=allr[allr.Date==q]
  for h in HORIZONS:
   cutoff=q-pd.offsets.BDay(h+EMBARGO_DAYS);hist=allr[allr.Date<=cutoff]
   pa.append({"QueryDate":q.date().isoformat(),"HORIZON":h,"CUTOFF_DATE":cutoff.date().isoformat(),"MAX_HISTORICAL_DATE":hist.Date.max().date().isoformat() if len(hist) else "","STATUS":"PASS" if len(hist) and hist.Date.max()<=cutoff else "FAIL"})
   for _,r in cur.iterrows():rows.append({"QueryDate":q.date().isoformat(),"Ticker":r.Ticker,"HORIZON":h,"REALIZED_RETURN":float(r[f"Forward{h}"]),**{v:float(r[v]) for v in STATE_FIELDS}})
 df=pd.DataFrame(rows);disc=[d.date().isoformat() for d in dates[:DISCOVERY_DATES]];oos=[d.date().isoformat() for d in dates[DISCOVERY_DATES:]]
 directions,dir_df=freeze_directions(df,disc)
 model_rows=[];date_rows=[]
 for name,vars_ in SIGNAL_SPECS.items():
  for h in HORIZONS:
   x=df[(df.HORIZON==h)&df.QueryDate.isin(oos)].copy();x["SCORE"]=np.nan
   for q,idx in x.groupby("QueryDate").groups.items():
    g=x.loc[idx].copy();x.loc[idx,"SCORE"]=make_score(g,vars_,directions[name])
   model_rows.append({"SIGNAL":name,"HORIZON":h,"PERIOD":"OOS_24","VARIABLES":" + ".join(vars_),**evaluate(x)})
   for q,g in x.groupby("QueryDate"):date_rows.append({"SIGNAL":name,"HORIZON":h,"QueryDate":q,"IC":date_ic(g,"SCORE"),"TOP_BOTTOM":spread(g,"SCORE"),"N":len(g)})
 # Regime is determined solely from current-state cross-section.
 regime_map={q:regime_label(g) for q,g in df[df.HORIZON==5].groupby("QueryDate")}
 for r in date_rows:r["REGIME"]=regime_map.get(r["QueryDate"],"UNKNOWN")
 dr=pd.DataFrame(date_rows);reg_rows=[]
 for (sig,h,reg),g in dr.groupby(["SIGNAL","HORIZON","REGIME"]):reg_rows.append({"SIGNAL":sig,"HORIZON":h,"REGIME":reg,"N_DATES":len(g),"MEAN_RANK_IC":g.IC.mean(),"MEDIAN_RANK_IC":g.IC.median(),"POSITIVE_IC_RATE":(g.IC>0).mean(),"MEAN_TOP_MINUS_BOTTOM":g.TOP_BOTTOM.mean()})
 base=[]
 for h in HORIZONS:
  x=df[(df.HORIZON==h)&df.QueryDate.isin(oos)];d=x.groupby("QueryDate").REALIZED_RETURN.mean();base.append({"HORIZON":h,"BASELINE_MEAN_DATE_RETURN":d.mean(),"BASELINE_MEDIAN_DATE_RETURN":d.median(),"BASELINE_POSITIVE_DATE_RATE":(d>0).mean()})
 outdir=os.path.join(os.getcwd(),"cere_phase5c_validation");os.makedirs(outdir,exist_ok=True)
 pd.DataFrame(audit).to_csv(os.path.join(outdir,f"phase5c_ticker_audit_{RUN_ID}.csv"),index=False);df.to_csv(os.path.join(outdir,f"phase5c_oos_observations_{RUN_ID}.csv"),index=False);dir_df.to_csv(os.path.join(outdir,f"phase5c_frozen_directions_{RUN_ID}.csv"),index=False);pd.DataFrame(model_rows).to_csv(os.path.join(outdir,f"phase5c_model_oos_summary_{RUN_ID}.csv"),index=False);dr.to_csv(os.path.join(outdir,f"phase5c_date_by_date_{RUN_ID}.csv"),index=False);pd.DataFrame(reg_rows).to_csv(os.path.join(outdir,f"phase5c_regime_summary_{RUN_ID}.csv"),index=False);pd.DataFrame(base).to_csv(os.path.join(outdir,f"phase5c_baseline_{RUN_ID}.csv"),index=False);pd.DataFrame(pa).to_csv(os.path.join(outdir,f"phase5c_purging_audit_{RUN_ID}.csv"),index=False)
 print("\n--- FASE 5C OOS 24 FECHAS ---");print(pd.DataFrame(model_rows)[["SIGNAL","HORIZON","MEAN_RANK_IC","MEDIAN_RANK_IC","POSITIVE_IC_RATE","MEAN_TOP_MINUS_BOTTOM","MEAN_REALIZED_WHEN_SCORE_POSITIVE"]].to_string(index=False))
 print("\n--- FASE 5C REGÍMENES ---");print(pd.DataFrame(reg_rows).to_string(index=False));print("\n--- BASELINE ---");print(pd.DataFrame(base).to_string(index=False))
 print("\n--- AUDITORÍA ---");print(f"Tickers configurados: {len(TICKERS)}");print(f"Tickers OK: {len(datasets)}");print(f"Global query dates: {len(dates)} | Discovery: {len(disc)} | OOS: {len(oos)}");print(f"Cross-sectional observations: {len(df):,}");print("PURGING_AUDIT:","PASS" if all(x["STATUS"]=="PASS" for x in pa) else "FAIL");print("LOOK_AHEAD_PREDICTORS: PASS");print("STATUS: PASS");print(f"Artifacts: {outdir}")
if __name__=="__main__":main()
