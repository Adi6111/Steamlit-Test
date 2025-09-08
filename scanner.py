# scanner.py
import io, os, time, warnings
import numpy as np
import pandas as pd
import requests, yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# =======================
# CONFIG (defaults)
# =======================

# Liquidity Screen
MIN_ADV_USD           = 10_000_000
ADV_LOOKBACK_DAYS     = 90
HIST_YEARS            = 3
MAX_TICKERS_DEBUG     = None
PRINT_PROGRESS_EVERY  = 200

# TEST MODE
TEST_MODE        = True
TEST_CUTOFF_STR  = "01/01/25"   # dd/mm/yy
SHOW_BACKTEST_PERF = True

# Risk & trend settings
VOL_LOOKBACK_DAYS  = 60
MOM_3M_DAYS        = 63
MOM_6M_DAYS        = 126
MA_DAYS            = 200
APPLY_TREND_FILTER = True
APPLY_VOL_FILTER   = True
REGIME_CHECK       = True

# Output & cache
READ_CACHE_IF_FRESH = True
CACHE_MAX_AGE_HOURS = 24
SAVE_RESULTS        = False
FULL_CSV   = "us_scanner_full.csv"
LONGS_CSV  = "us_scanner_longs.csv"
SHORTS_CSV = "us_scanner_shorts.csv"

# Symbol directories
URL_NASDAQ = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
URL_OTHER  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

# =======================
# Functions (unchanged)
# =======================
def parse_cutoff(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, dayfirst=True, utc=False).normalize().tz_localize(None)

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd

def cache_is_fresh(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path): return False
    return (time.time() - os.path.getmtime(path)) / 3600.0 <= max_age_hours

def latest(df: pd.DataFrame, row: str):
    try:
        if df is None or df.empty or row not in df.index: return np.nan
        s = df.loc[row].dropna()
        return float(s.iloc[0]) if len(s) else np.nan
    except Exception:
        return np.nan

def latest_asof(df: pd.DataFrame, row: str, cutoff: pd.Timestamp):
    try:
        if df is None or df.empty or row not in df.index: return np.nan
        cols = pd.to_datetime(df.columns, errors="coerce")
        mask = cols.notna() & (cols <= cutoff)
        if not mask.any(): return np.nan
        s = df.loc[row, df.columns[mask]].dropna()
        return float(s.iloc[0]) if len(s) else np.nan
    except Exception:
        return np.nan

def last_close_asof(tk: str, cutoff: pd.Timestamp) -> float:
    start = cutoff - pd.Timedelta(days=365)
    end   = cutoff + pd.Timedelta(days=2)
    try:
        df = yf.Ticker(tk).history(start=start, end=end, interval="1d", auto_adjust=False)
        if df.empty: return np.nan
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        df = df[df.index <= cutoff]
        if df.empty: return np.nan
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan

def series_asof(tk: str, end_cutoff: pd.Timestamp, lookback_days: int) -> pd.Series:
    """Close series up to cutoff with lookback window."""
    start = end_cutoff - pd.Timedelta(days=lookback_days*2 + 10)
    end   = end_cutoff + pd.Timedelta(days=2)
    try:
        df = yf.Ticker(tk).history(start=start, end=end, interval="1d", auto_adjust=False)
        if df.empty: return pd.Series(dtype=float)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        s = df["Close"].copy()
        s = s[s.index <= end_cutoff]
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)

def series_live(tk: str, period: str = "2y") -> pd.Series:
    try:
        df = yf.Ticker(tk).history(period=period, interval="1d", auto_adjust=False)
        if df.empty: return pd.Series(dtype=float)
        return df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)

def compute_volatility(tk: str, days: int, cutoff: pd.Timestamp | None) -> float:
    s = series_asof(tk, cutoff, days+10) if cutoff is not None else series_live(tk, "1y")
    if s is None or s.empty or len(s) < min(20, days//2): return np.nan
    ret = s.pct_change().dropna()
    if ret.empty: return np.nan
    return float(ret.tail(days).std())

def compute_momentum(tk: str, days: int, cutoff: pd.Timestamp | None) -> float:
    s = series_asof(tk, cutoff, days+10) if cutoff is not None else series_live(tk, "1y")
    if s is None or s.empty or len(s) < days+1: return np.nan
    s = s.dropna()
    p0 = s.iloc[-(days+1)]
    p1 = s.iloc[-1]
    if p0 > 0:
        return float(p1/p0 - 1.0)
    return np.nan

def compute_ma(tk: str, ma_days: int, cutoff: pd.Timestamp | None) -> float:
    s = series_asof(tk, cutoff, ma_days+30) if cutoff is not None else series_live(tk, "2y")
    if s is None or s.empty: return np.nan
    if len(s) < ma_days:
        return float(s.mean())
    return float(s.tail(ma_days).mean())

def shares_outstanding_asof(t: yf.Ticker, cutoff: pd.Timestamp) -> float:
    # Try full time series
    try:
        s = t.get_shares_full(start="1970-01-01", end=cutoff + pd.Timedelta(days=1))
        if isinstance(s, pd.Series) and not s.empty:
            idx = s.index
            if getattr(idx, "tz", None) is not None:
                s.index = idx.tz_convert(None)
            s = s[s.index <= cutoff]
            if not s.empty:
                return float(s.iloc[-1])
    except Exception:
        pass
    # Point estimate
    try:
        s2 = t.get_shares()
        if s2 is not None:
            return float(s2)
    except Exception:
        pass
    # Info fallback
    try:
        info = t.info or {}
        so = info.get("sharesOutstanding", np.nan)
        return float(so) if np.isfinite(so) else np.nan
    except Exception:
        return np.nan

def horizon_return(tk: str, start_price_date: pd.Timestamp, horizon_days: int) -> float:
    """Compute return from last close at cutoff to cutoff + horizon_days."""
    s = series_asof(tk, start_price_date + pd.Timedelta(days=horizon_days+3), horizon_days+5)
    if s is None or s.empty: return np.nan
    s = s[s.index >= start_price_date - pd.Timedelta(days=1)]
    if s.empty: return np.nan
    # reference price at or before cutoff
    p0 = s[s.index <= start_price_date]
    p1 = s[s.index >  start_price_date]
    if p0.empty or p1.empty: return np.nan
    p0 = float(p0.iloc[-1]); p1 = float(p1.iloc[-1])
    if p0 <= 0: return np.nan
    return float(p1/p0 - 1.0)

# =======================
# Minimal wrapper for Streamlit
# =======================
def run_scan(config_overrides: dict | None = None):
    """
    Runs the exact same pipeline and returns:
      - res (full snapshot DataFrame)
      - longs_sorted (DataFrame)
      - shorts_sorted (DataFrame)
      - backtest dict: {"LONGS": df, "SHORTS": df} if TEST_MODE & SHOW_BACKTEST_PERF, else {}
      - cols_show (list of columns for display)
      - logs (list[str]) the same prints, captured for UI if you want to show them
    """
    global MIN_ADV_USD, ADV_LOOKBACK_DAYS, HIST_YEARS, MAX_TICKERS_DEBUG, PRINT_PROGRESS_EVERY
    global TEST_MODE, TEST_CUTOFF_STR, SHOW_BACKTEST_PERF
    global VOL_LOOKBACK_DAYS, MOM_3M_DAYS, MOM_6M_DAYS, MA_DAYS, APPLY_TREND_FILTER, APPLY_VOL_FILTER, REGIME_CHECK
    global READ_CACHE_IF_FRESH, CACHE_MAX_AGE_HOURS, SAVE_RESULTS, FULL_CSV, LONGS_CSV, SHORTS_CSV

    if config_overrides:
        for k, v in config_overrides.items():
            if k in globals():
                globals()[k] = v

    logs = []
    def logprint(x):
        logs.append(x)

    if (not TEST_MODE) and READ_CACHE_IF_FRESH and cache_is_fresh(FULL_CSV, CACHE_MAX_AGE_HOURS):
        logprint(f"Using cached snapshot: {FULL_CSV} (≤ {CACHE_MAX_AGE_HOURS}h old)")
        res = pd.read_csv(FULL_CSV)
    else:
        # 1) Build universe (US equities)
        logprint("Loading symbol directories...")
        r1 = requests.get(URL_NASDAQ, timeout=30); r1.raise_for_status()
        txt1 = "\n".join([ln for ln in r1.text.splitlines() if "File Creation Time" not in ln])
        nasdaq = pd.read_csv(io.StringIO(txt1), sep="|"); nasdaq.columns = [c.strip().lower().replace(" ", "_") for c in nasdaq.columns]
        r2 = requests.get(URL_OTHER, timeout=30); r2.raise_for_status()
        txt2 = "\n".join([ln for ln in r2.text.splitlines() if "File Creation Time" not in ln])
        other  = pd.read_csv(io.StringIO(txt2), sep="|"); other.columns  = [c.strip().lower().replace(" ", "_") for c in other.columns]

        nas_sym_col = "symbol" if "symbol" in nasdaq.columns else "ticker"
        oth_sym_col = "act_symbol" if "act_symbol" in other.columns else ("symbol" if "symbol" in other.columns else "ticker")
        nasdaq["ticker_raw"] = nasdaq[nas_sym_col].astype(str)
        other["ticker_raw"]  = other[oth_sym_col].astype(str)
        if "security_name" not in nasdaq.columns and "company_name" not in nasdaq.columns: nasdaq["security_name"] = ""
        if "security_name" not in other.columns  and "company_name"  not in other.columns: other["security_name"]  = ""
        nasdaq["ticker"] = nasdaq["ticker_raw"]; other["ticker"] = other["ticker_raw"]

        def _name_col(df): return df["security_name"] if "security_name" in df.columns else df["company_name"]

        for df in (nasdaq, other):
            if "etf" not in df.columns: df["etf"] = "N"
            if "test_issue" not in df.columns: df["test_issue"] = "N"
        bad_words = ["ETF","ETN","PREFERRED","PFD","RIGHT","RIGHTS","UNIT","WARRANT","SPAC","DEPOSITARY"]
        nas_mask = (nasdaq["etf"].astype(str).str.upper()!="Y") & (nasdaq["test_issue"].astype(str).str.upper()!="Y")
        oth_mask = (other["etf"].astype(str).str.upper()!="Y") & (other["test_issue"].astype(str).str.upper()!="Y")
        nas_names = _name_col(nasdaq).fillna("").str.upper(); oth_names = _name_col(other).fillna("").str.upper()
        for w in bad_words:
            nas_mask &= ~nas_names.str.contains(w, regex=False)
            oth_mask &= ~oth_names.str.contains(w, regex=False)
        nas_mask &= ~nasdaq["ticker"].astype(str).str.contains(r"[\^\$/]", regex=True, na=False)
        oth_mask &= ~other["ticker"].astype(str).str.contains(r"[\^\$/]", regex=True, na=False)

        nasdaq_eq = nasdaq[nas_mask].copy(); other_eq = other[oth_mask].copy()
        universe = pd.concat([nasdaq_eq[["ticker_raw","ticker"]], other_eq[["ticker_raw","ticker"]]], ignore_index=True).drop_duplicates()
        universe["yf_ticker"] = (universe["ticker"].astype(str).str.upper()
                                 .str.replace(" ", "", regex=False).str.replace(".", "-", regex=False))
        universe = universe[~universe["yf_ticker"].str.contains(r"=|--|[^A-Z0-9\-\.]", regex=True, na=False)]
        universe = universe.drop_duplicates(subset=["yf_ticker"]).reset_index(drop=True)
        if MAX_TICKERS_DEBUG: universe = universe.head(MAX_TICKERS_DEBUG)
        logprint(f"Universe (post-clean): {len(universe)}")

        # 2) Liquidity (ADV)
        logprint("Computing ADV (this may take a while)...")
        if TEST_MODE:
            CUTOFF = parse_cutoff(TEST_CUTOFF_STR)
            adv_start = CUTOFF - pd.Timedelta(days=ADV_LOOKBACK_DAYS*2)
            adv_end   = CUTOFF + pd.Timedelta(days=2)
        else:
            CUTOFF = None
            adv_start = adv_end = None

        advs = []
        for i, tk in enumerate(universe["yf_ticker"], 1):
            adv = np.nan
            try:
                if TEST_MODE:
                    df_px = yf.Ticker(tk).history(start=adv_start, end=adv_end, interval="1d", auto_adjust=False)
                    if not df_px.empty and getattr(df_px.index, "tz", None) is not None:
                        df_px.index = df_px.index.tz_convert(None)
                    df_px = df_px[df_px.index <= CUTOFF]
                else:
                    df_px = yf.Ticker(tk).history(period=f"{ADV_LOOKBACK_DAYS}d", interval="1d", auto_adjust=False)
                if df_px is not None and not df_px.empty:
                    sub = df_px[["Close","Volume"]].dropna()
                    sub = sub[(sub["Close"]>0) & (sub["Volume"]>0)]
                    if len(sub) >= 10:
                        adv = float((sub["Close"]*sub["Volume"]).mean())
            except Exception:
                adv = np.nan
            advs.append(adv)
            if i % PRINT_PROGRESS_EVERY == 0:
                logprint(f"  processed {i} / {len(universe)}")

        universe["avg_dollar_vol"] = advs
        res = universe.dropna(subset=["avg_dollar_vol"])
        res = res[res["avg_dollar_vol"] >= MIN_ADV_USD].reset_index(drop=True)
        logprint(f"Pass liquidity (${MIN_ADV_USD:,} ADV): {len(res)}")
        if res.empty:
            return {
                "res": pd.DataFrame(), "longs_sorted": pd.DataFrame(), "shorts_sorted": pd.DataFrame(),
                "backtest": {}, "cols_show": [], "logs": logs
            }

        # 3) Fundamentals snapshot
        logprint("Fetching fundamentals...")
        rows = []
        for i, tk in enumerate(res["yf_ticker"], 1):
            rec = {"yf_ticker": tk,
                   "price": np.nan, "market_cap": np.nan, "ev": np.nan,
                   "revenue": np.nan, "ebit": np.nan, "net_income": np.nan, "fcf": np.nan,
                   "gross_margin": np.nan, "op_margin": np.nan, "net_margin": np.nan, "fcf_margin": np.nan,
                   "ROA": np.nan, "ROE": np.nan, "PE": np.nan, "EV_EBIT": np.nan, "P_FCF": np.nan}
            try:
                t   = yf.Ticker(tk)
                fin = t.financials if isinstance(t.financials, pd.DataFrame) else pd.DataFrame()
                bs  = t.balance_sheet if isinstance(t.balance_sheet, pd.DataFrame) else pd.DataFrame()
                cf  = t.cashflow if isinstance(t.cashflow, pd.DataFrame) else pd.DataFrame()

                if TEST_MODE:
                    if not fin.empty:
                        fcols = pd.to_datetime(fin.columns, errors="coerce"); fin = fin.loc[:, fcols.notna() & (fcols <= CUTOFF)]
                    if not bs.empty:
                        bcols = pd.to_datetime(bs.columns, errors="coerce");  bs  = bs.loc[:,  bcols.notna() & (bcols  <= CUTOFF)]
                    if not cf.empty:
                        ccols = pd.to_datetime(cf.columns, errors="coerce");  cf  = cf.loc[:,  ccols.notna() & (ccols  <= CUTOFF)]

                if TEST_MODE:
                    px_cut = last_close_asof(tk, CUTOFF)
                    sh_out = shares_outstanding_asof(t, CUTOFF)
                    price  = px_cut if np.isfinite(px_cut) else np.nan
                    mcap   = (px_cut * sh_out) if np.isfinite(px_cut) and np.isfinite(sh_out) and sh_out>0 else np.nan
                else:
                    fi = getattr(t, "fast_info", None)
                    price = (fi.get("last_price") if isinstance(fi, dict) else getattr(fi, "last_price", np.nan)) if fi is not None else np.nan
                    mcap  = (fi.get("market_cap") if isinstance(fi, dict) else getattr(fi, "market_cap", np.nan))  if fi is not None else np.nan

                _latest = (lambda df, row: latest_asof(df, row, CUTOFF)) if TEST_MODE else latest
                rev     = _latest(fin, "Total Revenue")
                gp      = _latest(fin, "Gross Profit")
                op_inc  = _latest(fin, "Operating Income")
                net_inc = _latest(fin, "Net Income")
                ebit    = _latest(fin, "Ebit");  ebit = ebit if np.isfinite(ebit) else _latest(fin, "EBIT")
                cash    = _latest(bs,  "Cash And Cash Equivalents"); cash = cash if np.isfinite(cash) else _latest(bs, "Cash")
                debt    = _latest(bs,  "Total Debt")
                equity  = _latest(bs,  "Total Stockholder Equity")
                assets  = _latest(bs,  "Total Assets")

                fcf = _latest(cf, "Free Cash Flow")
                if not np.isfinite(fcf):
                    cfo   = _latest(cf, "Total Cash From Operating Activities")
                    capex = _latest(cf, "Capital Expenditures")
                    if np.isfinite(cfo) and np.isfinite(capex): fcf = cfo - capex

                ev = mcap if np.isfinite(mcap) else np.nan
                if np.isfinite(debt): ev = (ev if np.isfinite(ev) else 0) + debt
                if np.isfinite(cash): ev = (ev if np.isfinite(ev) else 0) - cash

                gm   = (gp / rev) if np.isfinite(gp) and np.isfinite(rev) and rev != 0 else np.nan
                opm  = (op_inc / rev) if np.isfinite(op_inc) and np.isfinite(rev) and rev != 0 else np.nan
                npm  = (net_inc / rev) if np.isfinite(net_inc) and np.isfinite(rev) and rev != 0 else np.nan
                fcfm = (fcf / rev) if np.isfinite(fcf) and np.isfinite(rev) and rev != 0 else np.nan
                roa  = (net_inc / assets) if np.isfinite(net_inc) and np.isfinite(assets) and assets != 0 else np.nan
                roe  = (net_inc / equity) if np.isfinite(net_inc) and np.isfinite(equity) and equity != 0 else np.nan

                pe      = (mcap / net_inc) if np.isfinite(mcap) and np.isfinite(net_inc) and net_inc > 0 else np.nan
                ev_ebit = (ev / ebit)      if np.isfinite(ev)   and np.isfinite(ebit)   and ebit    > 0 else np.nan
                p_fcf   = (mcap / fcf)     if np.isfinite(mcap) and np.isfinite(fcf)    and fcf     > 0 else np.nan

                rec.update({"price": price, "market_cap": mcap, "ev": ev,
                            "revenue": rev, "ebit": ebit, "net_income": net_inc, "fcf": fcf,
                            "gross_margin": gm, "op_margin": opm, "net_margin": npm, "fcf_margin": fcfm,
                            "ROA": roa, "ROE": roe, "PE": pe, "EV_EBIT": ev_ebit, "P_FCF": p_fcf})
            except Exception:
                pass
            rows.append(rec)
            if i % PRINT_PROGRESS_EVERY == 0:
                logprint(f"  processed {i} / {len(res)}")

        res = res.merge(pd.DataFrame(rows), on="yf_ticker", how="right")
        res = res.replace([np.inf, -np.inf], np.nan)

        # 4) Scores
        q_parts = ["gross_margin","op_margin","fcf_margin","ROE"]
        v_parts = ["PE","EV_EBIT","P_FCF"]
        res["QualityScore"] = 0
        for c in q_parts:
            if c in res.columns:
                fill = np.nanmedian(res[c]) if np.isfinite(res[c]).any() else 0.0
                res["QualityScore"] = res["QualityScore"] + zscore(res[c].fillna(fill))
        res["ValuationScore"] = 0
        for c in v_parts:
            if c in res.columns:
                fill = np.nanmedian(res[c]) if np.isfinite(res[c]).any() else 0.0
                res["ValuationScore"] = res["ValuationScore"] - zscore(res[c].fillna(fill))

        # 5) Valuation percentiles vs own history
        logprint("Computing valuation percentiles vs own history...")
        pct_rows = []
        for i, r in res.iterrows():
            tk = r["yf_ticker"]
            cur_ev_ebit = r.get("EV_EBIT", np.nan)
            cur_p_fcf   = r.get("P_FCF",   np.nan)
            out = {"EV_EBIT_pctile": np.nan, "P_FCF_pctile": np.nan}
            try:
                t = yf.Ticker(tk)
                fin = t.financials if isinstance(t.financials, pd.DataFrame) else pd.DataFrame()
                bs  = t.balance_sheet if isinstance(t.balance_sheet, pd.DataFrame) else pd.DataFrame()
                cf  = t.cashflow if isinstance(t.cashflow, pd.DataFrame) else pd.DataFrame()

                if TEST_MODE:
                    mcap_ref = r.get("market_cap", np.nan)
                    if not fin.empty:
                        fcols = pd.to_datetime(fin.columns, errors="coerce"); fin = fin.loc[:, fcols.notna() & (fcols <= CUTOFF)]
                    if not bs.empty:
                        bcols = pd.to_datetime(bs.columns, errors="coerce");  bs  = bs.loc[:,  bcols.notna() & (bcols  <= CUTOFF)]
                    if not cf.empty:
                        ccols = pd.to_datetime(cf.columns, errors="coerce");  cf  = cf.loc[:,  ccols.notna() & (ccols  <= CUTOFF)]
                else:
                    fi  = getattr(t, "fast_info", None)
                    mcap_ref = (fi.get("market_cap") if isinstance(fi, dict) else getattr(fi, "market_cap", np.nan)) if fi is not None else np.nan

                if not fin.empty and not bs.empty:
                    cols = fin.columns.tolist()[:HIST_YEARS]
                    ev_ebit_hist, p_fcf_hist = [], []
                    for c in cols:
                        ebit_h = fin.at["Ebit", c] if "Ebit" in fin.index else (fin.at["EBIT", c] if "EBIT" in fin.index else np.nan)
                        try:
                            if "Free Cash Flow" in cf.index:
                                fcf_h = cf.at["Free Cash Flow", c]
                            else:
                                cfo_h   = cf.at["Total Cash From Operating Activities", c]
                                capex_h = cf.at["Capital Expenditures", c]
                                fcf_h   = cfo_h - capex_h
                        except Exception:
                            fcf_h = np.nan
                        debt_h = bs.at["Total Debt", c] if "Total Debt" in bs.index else np.nan
                        cash_h = bs.at["Cash And Cash Equivalents", c] if "Cash And Cash Equivalents" in bs.index else np.nan

                        ev_h = mcap_ref if np.isfinite(mcap_ref) else np.nan
                        if np.isfinite(debt_h): ev_h = (ev_h if np.isfinite(ev_h) else 0) + debt_h
                        if np.isfinite(cash_h): ev_h = (ev_h if np.isfinite(ev_h) else 0) - cash_h

                        if np.isfinite(ebit_h) and ebit_h > 0 and np.isfinite(ev_h) and ev_h > 0: ev_ebit_hist.append(ev_h/ebit_h)
                        if np.isfinite(fcf_h)  and fcf_h  > 0 and np.isfinite(mcap_ref) and mcap_ref>0: p_fcf_hist.append(mcap_ref/fcf_h)

                    def pctile(cur, hist):
                        if not np.isfinite(cur) or not hist: return np.nan
                        arr = np.array([x for x in hist if np.isfinite(x)])
                        if len(arr) < 3: return np.nan
                        return float((arr < cur).mean())  # 0=cheap, 1=expensive

                    out["EV_EBIT_pctile"] = pctile(cur_ev_ebit, ev_ebit_hist)
                    out["P_FCF_pctile"]   = pctile(cur_p_fcf,   p_fcf_hist)
            except Exception:
                pass
            pct_rows.append({"yf_ticker": tk, **out})
            if (i+1) % PRINT_PROGRESS_EVERY == 0:
                logprint(f"  hist %iles processed {i+1} / {len(res)}")

        res = res.merge(pd.DataFrame(pct_rows), on="yf_ticker", how="left")

        # 6) Risk & Trend Features
        logprint("Computing risk & trend features...")
        vols, mom3, mom6, ma200, px = [], [], [], [], []
        for i, tk in enumerate(res["yf_ticker"], 1):
            v  = compute_volatility(tk, VOL_LOOKBACK_DAYS, CUTOFF if TEST_MODE else None)
            m3 = compute_momentum(tk, MOM_3M_DAYS, CUTOFF if TEST_MODE else None)
            m6 = compute_momentum(tk, MOM_6M_DAYS, CUTOFF if TEST_MODE else None)
            ma = compute_ma(tk, MA_DAYS, CUTOFF if TEST_MODE else None)
            if TEST_MODE:
                p  = last_close_asof(tk, CUTOFF)
            else:
                s = series_live(tk, "1y"); p = float(s.iloc[-1]) if not s.empty else np.nan
            vols.append(v); mom3.append(m3); mom6.append(m6); ma200.append(ma); px.append(p)
            if i % PRINT_PROGRESS_EVERY == 0:
                logprint(f"  risk/trend processed {i} / {len(res)}")

        res["volatility"] = vols
        res["mom_3m"]     = mom3
        res["mom_6m"]     = mom6
        res["MA200"]      = ma200
        res["price_pti"]  = px   # price at cutoff or latest

        # Factor additions
        res["MomentumScore"] = zscore(res["mom_6m"].fillna(0))
        res["LowVolScore"]   = -zscore(res["volatility"].fillna(res["volatility"].median()))
        res["TotalScore"]    = (res["QualityScore"].fillna(0)
                                + res["ValuationScore"].fillna(0)
                                + 0.5*res["MomentumScore"].fillna(0)
                                + 0.5*res["LowVolScore"].fillna(0))

        # 7) Fundamental long/short buckets
        longs = res[(res["QualityScore"]  >= res["QualityScore"].quantile(0.70)) &
                    (res["ValuationScore"] >= res["ValuationScore"].quantile(0.60)) &
                    (res["EV_EBIT_pctile"] <= 0.35)].copy()

        shorts = res[(res["QualityScore"]  <= res["QualityScore"].quantile(0.30)) &
                     (res["ValuationScore"] <= res["ValuationScore"].quantile(0.40)) &
                     (res["EV_EBIT_pctile"] >= 0.65)].copy()

        if APPLY_TREND_FILTER:
            longs["TrendOK"]  = (longs["price_pti"] > longs["MA200"]) & (longs["mom_3m"] > 0)
            shorts["TrendOK"] = (shorts["price_pti"] < shorts["MA200"]) & (shorts["mom_3m"] < 0)
            longs  = longs[longs["TrendOK"] == True].copy()
            shorts = shorts[shorts["TrendOK"] == True].copy()

        if APPLY_VOL_FILTER:
            vq = res["volatility"].quantile(0.90)
            longs  = longs[longs["volatility"] < vq].copy()
            shorts = shorts[shorts["volatility"] < vq].copy()

        if REGIME_CHECK:
            spy = series_asof("SPY", CUTOFF, MA_DAYS+30) if TEST_MODE else series_live("SPY", "2y")
            if not spy.empty:
                m = spy.tail(MA_DAYS).mean() if len(spy) >= MA_DAYS else spy.mean()
                bear = spy.iloc[-1] < m
                if bear:
                    q80 = res["QualityScore"].quantile(0.80)
                    longs = longs[longs["QualityScore"] >= q80].copy()

        # 8) Ranking + Suggested holding period
        def suggest_hold(row):
            evp = row.get("EV_EBIT_pctile", np.nan)
            m3  = row.get("mom_3m", 0.0)
            m6  = row.get("mom_6m", 0.0)
            if np.isfinite(evp) and evp <= 0.20 and m3 < 0.10:
                return "~9-12 months (deep value, gradual rerating)"
            if m6 >= 0.20 and m3 >= 0.05:
                return "~3-6 months (momentum-led rerating)"
            return "~6 months (reassess quarterly)"

        sort_cols_long  = ["EV_EBIT_pctile","TotalScore","ValuationScore","QualityScore"]
        sort_cols_short = ["EV_EBIT_pctile","TotalScore","ValuationScore","QualityScore"]

        longs_sorted  = longs.sort_values(sort_cols_long, ascending=[True, False, False, False]).copy()
        shorts_sorted = shorts.sort_values(sort_cols_short, ascending=[False, True, True, True]).copy()

        if not longs_sorted.empty:
            longs_sorted.insert(0, "Rank", range(1, len(longs_sorted)+1))
            longs_sorted["Suggested_Hold"] = longs_sorted.apply(suggest_hold, axis=1)
        if not shorts_sorted.empty:
            shorts_sorted.insert(0, "Rank", range(1, len(shorts_sorted)+1))
            shorts_sorted["Suggested_Hold"] = "~3-6 months or until momentum turns"

        if SAVE_RESULTS and (not TEST_MODE):
            res.to_csv(FULL_CSV, index=False)

    cols_show = ["Rank","yf_ticker","avg_dollar_vol","QualityScore","ValuationScore",
                 "MomentumScore","LowVolScore","TotalScore",
                 "PE","EV_EBIT","P_FCF","EV_EBIT_pctile","P_FCF_pctile",
                 "volatility","mom_3m","mom_6m","price_pti","MA200","Suggested_Hold"]

    # 10) Backtest since cutoff (TEST_MODE)
    backtest = {}
    if TEST_MODE and SHOW_BACKTEST_PERF:
        ct = parse_cutoff(TEST_CUTOFF_STR)
        horizons = [21, 63, 126]  # ~1m, ~3m, ~6m
        for label, df in [("LONGS", longs_sorted if 'longs_sorted' in locals() else pd.DataFrame()),
                          ("SHORTS", shorts_sorted if 'shorts_sorted' in locals() else pd.DataFrame())]:
            if df is None or df.empty:
                backtest[label] = pd.DataFrame()
                continue
            rows = []
            for tk in df["yf_ticker"]:
                p0 = last_close_asof(tk, ct)
                pN = series_live(tk, "1y")
                lat = float(pN.iloc[-1]) if pN is not None and not pN.empty else np.nan
                since = (lat/p0 - 1.0) if np.isfinite(p0) and p0>0 and np.isfinite(lat) else np.nan
                hr = {f"{h//21}m": horizon_return(tk, ct, h) for h in horizons}
                out = {"yf_ticker": tk, "cutoff_close": p0, "latest_close": lat, "since_cutoff": since, **hr}
                rows.append(out)
            perf = pd.DataFrame(rows)
            if label == "LONGS":
                perf["profitable"] = perf["since_cutoff"] > 0
            else:
                perf["profitable"] = perf["since_cutoff"] < 0
            backtest[label] = perf[["yf_ticker","cutoff_close","latest_close","since_cutoff","1m","3m","6m","profitable"]]

    return {
        "res": res if 'res' in locals() else pd.DataFrame(),
        "longs_sorted": longs_sorted if 'longs_sorted' in locals() else pd.DataFrame(),
        "shorts_sorted": shorts_sorted if 'shorts_sorted' in locals() else pd.DataFrame(),
        "backtest": backtest,
        "cols_show": cols_show,
        "logs": logs,
    }

if __name__ == "__main__":
    out = run_scan()
    cols_show = out["cols_show"]

    print("\n=== Long candidates (quality + cheap vs own history + trend OK) ===")
    if not out["longs_sorted"].empty:
        print(out["longs_sorted"][cols_show].head(25).to_string(index=False))
    else:
        print("None found (try relaxing thresholds, disabling trend filter, or expanding universe).")

    print("\n=== Short candidates (low quality + rich vs own history + trend OK) ===")
    if not out["shorts_sorted"].empty:
        print(out["shorts_sorted"][cols_show].head(25).to_string(index=False))
    else:
        print("None found (try relaxing thresholds).")

    if TEST_MODE and SHOW_BACKTEST_PERF and out["backtest"]:
        ct = parse_cutoff(TEST_CUTOFF_STR)
        print(f"\nBacktest (realized) since cutoff = {ct.date()}")
        for label, df in out["backtest"].items():
            if df is not None and not df.empty:
                print(f"\n{label} performance:")
                print(df.to_string(index=False))
            else:
                print(f"\n{label}: none")
