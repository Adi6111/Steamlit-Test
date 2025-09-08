# app.py
import streamlit as st
import pandas as pd
from datetime import date
import scanner 

st.set_page_config(page_title="US Stock Scanner Demo", layout="wide")
st.title("US Stock Scanner — Quality × Valuation (with Trend/Vol overlays)")


with st.sidebar:
    st.header("Configuration")

    TEST_MODE = st.toggle("TEST_MODE (point-in-time)", value=True)
    TEST_CUTOFF = st.date_input("Cutoff date (dd/mm/yy)", value=date(2025,1,1))
    SHOW_BACKTEST_PERF = st.toggle("Show realized backtest since cutoff", value=True)

    st.divider()
    MIN_ADV_USD = st.number_input("MIN_ADV_USD", value=10_000_000, step=1_000_000, format="%d")
    ADV_LOOKBACK_DAYS = st.number_input("ADV_LOOKBACK_DAYS", value=90, step=5)
    HIST_YEARS = st.number_input("HIST_YEARS (valuation history years)", value=3, step=1)
    MAX_TICKERS_DEBUG = st.number_input("MAX_TICKERS_DEBUG (0 = all)", value=0, step=100)

    st.divider()
    VOL_LOOKBACK_DAYS = st.number_input("VOL_LOOKBACK_DAYS", value=60, step=5)
    MOM_3M_DAYS = st.number_input("MOM_3M_DAYS", value=63, step=1)
    MOM_6M_DAYS = st.number_input("MOM_6M_DAYS", value=126, step=1)
    MA_DAYS = st.number_input("MA_DAYS", value=200, step=5)

    APPLY_TREND_FILTER = st.toggle("APPLY_TREND_FILTER", value=True)
    APPLY_VOL_FILTER = st.toggle("APPLY_VOL_FILTER", value=True)
    REGIME_CHECK = st.toggle("REGIME_CHECK (SPY MA200 tighten)", value=True)

    st.divider()
    READ_CACHE_IF_FRESH = st.toggle("READ_CACHE_IF_FRESH (live only)", value=True)
    CACHE_MAX_AGE_HOURS = st.number_input("CACHE_MAX_AGE_HOURS", value=24, step=1)
    SAVE_RESULTS = st.toggle("SAVE_RESULTS (live only)", value=False)

    run_btn = st.button("Run scan", type="primary")

if run_btn:
    overrides = {
        "TEST_MODE": TEST_MODE,
        "TEST_CUTOFF_STR": TEST_CUTOFF.strftime("%d/%m/%y"),
        "SHOW_BACKTEST_PERF": SHOW_BACKTEST_PERF,
        "MIN_ADV_USD": int(MIN_ADV_USD),
        "ADV_LOOKBACK_DAYS": int(ADV_LOOKBACK_DAYS),
        "HIST_YEARS": int(HIST_YEARS),
        "MAX_TICKERS_DEBUG": None if int(MAX_TICKERS_DEBUG)==0 else int(MAX_TICKERS_DEBUG),
        "VOL_LOOKBACK_DAYS": int(VOL_LOOKBACK_DAYS),
        "MOM_3M_DAYS": int(MOM_3M_DAYS),
        "MOM_6M_DAYS": int(MOM_6M_DAYS),
        "MA_DAYS": int(MA_DAYS),
        "APPLY_TREND_FILTER": APPLY_TREND_FILTER,
        "APPLY_VOL_FILTER": APPLY_VOL_FILTER,
        "REGIME_CHECK": REGIME_CHECK,
        "READ_CACHE_IF_FRESH": READ_CACHE_IF_FRESH,
        "CACHE_MAX_AGE_HOURS": int(CACHE_MAX_AGE_HOURS),
        "SAVE_RESULTS": SAVE_RESULTS,
    }

    with st.spinner("Running pipeline (this can take a while on first run)…"):
        out = scanner.run_scan(overrides)

    # show logs (your original print lines)
    if out["logs"]:
        with st.expander("Run log"):
            for line in out["logs"]:
                st.text(line)

    cols_show = out["cols_show"]
    longs = out["longs_sorted"]
    shorts = out["shorts_sorted"]

    st.subheader("Long candidates (quality + cheap vs own history + trend OK)")
    if not longs.empty:
        st.dataframe(longs[cols_show].head(50), use_container_width=True)
        st.download_button("Download LONGS CSV", longs[cols_show].to_csv(index=False).encode(),
                           file_name="longs.csv", mime="text/csv")
    else:
        st.info("None found (try relaxing thresholds, disabling trend filter, or expanding universe).")

    st.subheader("Short candidates (low quality + rich vs own history + trend OK)")
    if not shorts.empty:
        st.dataframe(shorts[cols_show].head(50), use_container_width=True)
        st.download_button("Download SHORTS CSV", shorts[cols_show].to_csv(index=False).encode(),
                           file_name="shorts.csv", mime="text/csv")
    else:
        st.info("None found (try relaxing thresholds).")

    # Backtest
    if TEST_MODE and SHOW_BACKTEST_PERF and out["backtest"]:
        st.subheader(f"Backtest (realized) since cutoff = {TEST_CUTOFF.isoformat()}")
        tabs = st.tabs(["LONGS", "SHORTS"])
        for tab, label in zip(tabs, ["LONGS", "SHORTS"]):
            with tab:
                df = out["backtest"].get(label, pd.DataFrame())
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No {label.lower()} to show.")

    with st.expander("Full snapshot (all columns)"):
        st.dataframe(out["res"], use_container_width=True)


