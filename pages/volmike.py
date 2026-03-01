# pages/components/jsonExport.py

import io
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st



SECTOR_MAP = {
    "ETFs": ["spy", "qqq"],
    "finance": ["wfc", "c", "jpm", "bac", "hood", "coin", "pypl"],
    "Semiconductors": ["nvda", "avgo", "amd", "mu", "mrvl", "qcom", "smci"],
    "Software": ["msft", "pltr", "aapl", "googl", "meta", "uber", "tsla", "amzn"],
    "Futures": ["nq", "es", "gc", "ym", "cl"],
}

def detect_sector(ticker: str) -> str:
    t = ticker.lower()
    for sector, listing in SECTOR_MAP.items():
        if t in listing:
            return sector
    return "Other"

def human_volume(n):
    try:
        n = float(n)
    except:
        return n
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return str(int(n))


def extract_entries(intraday: pd.DataFrame) -> dict:
    call_entries = []
    put_entries  = []

    def add_entry(target, label, idx):
        target.append({
            "type":   label,
            "time":   pd.to_datetime(intraday.at[idx, "Time"]).strftime("%H:%M"),
            "price":  float(intraday.at[idx, "Close"]),
            "fLevel": float(intraday.at[idx, "F_numeric"]),
        })

    def col(name):
        return intraday[name] if name in intraday.columns else pd.Series("", index=intraday.index)

    # PUT
    for i in intraday.index[col("Put_FirstEntry_Emoji") == "🎯"]:
        add_entry(put_entries, "Put E1 🎯", i)
    for i in intraday.index[col("Put_FirstEntry_Emoji") == "⏳"]:
        add_entry(put_entries, "Put E1 ⏳ Blocked", i)
    for i in intraday.index[col("Put_DeferredEntry_Emoji") == "🧿"]:
        put_entries.append({
            "type":   "Put Reclaim 🧿",
            "time":   pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
            "price":  float(intraday.at[i, "Close"]),
            "fLevel": float(intraday.at[i, "F_numeric"]),
            "horse":  col("Put_DeferredReinforce_Emoji").at[i] == "❗️",
        })
    for i in intraday.index[col("Put_SecondEntry_Emoji") == "🎯2"]:
        add_entry(put_entries, "Put E2 🎯2", i)
    for i in intraday.index[col("Put_ThirdEntry_Emoji") == "🎯3"]:
        add_entry(put_entries, "Put E3 🎯3", i)

    # CALL
    for i in intraday.index[col("Call_FirstEntry_Emoji") == "🎯"]:
        add_entry(call_entries, "Call E1 🎯", i)
    for i in intraday.index[col("Call_FirstEntry_Emoji") == "⏳"]:
        add_entry(call_entries, "Call E1 ⏳ Blocked", i)
    for i in intraday.index[col("Call_DeferredEntry_Emoji") == "🧿"]:
        call_entries.append({
            "type":   "Call Reclaim 🧿",
            "time":   pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
            "price":  float(intraday.at[i, "Close"]),
            "fLevel": float(intraday.at[i, "F_numeric"]),
            "horse":  col("Call_DeferredReinforce_Emoji").at[i] == "❗️",
        })
    for i in intraday.index[col("Call_SecondEntry_Emoji") == "🎯2"]:
        add_entry(call_entries, "Call E2 🎯2", i)
    for i in intraday.index[col("Call_ThirdEntry_Emoji") == "🎯3"]:
        add_entry(call_entries, "Call E3 🎯3", i)

    return {"call": call_entries, "put": put_entries}


def detect_expansion_near_e1(intraday: pd.DataFrame, perimeter: int = 10) -> dict:
    out = {
        "bbw": {"present": False, "time": None, "count": 0},
        "std": {"present": False, "time": None, "count": 0},
    }
    if intraday is None or intraday.empty:
        return out

    call_e1_idx = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1_idx  = intraday.index[intraday.get("Put_FirstEntry_Emoji",  "") == "🎯"]
    if len(call_e1_idx) == 0 and len(put_e1_idx) == 0:
        return out

    all_e1_idx = list(call_e1_idx) + list(put_e1_idx)
    e1_pos = min(intraday.index.get_loc(idx) for idx in all_e1_idx)
    n = len(intraday)
    start = max(0, e1_pos - perimeter)
    end   = min(n - 1, e1_pos + perimeter)

    bbw_series = intraday.get("BBW Alert")
    std_series = intraday.get("STD_Alert")

    if bbw_series is not None:
        bbw_before = bbw_after = 0
        for pos in range(start, end + 1):
            if bbw_series.iloc[pos] == "🔥":
                if pos <= e1_pos: bbw_before += 1
                else:             bbw_after  += 1
        total = bbw_before + bbw_after
        if total > 0:
            out["bbw"]["present"] = True
            out["bbw"]["count"]   = total
            out["bbw"]["time"]    = ("both" if bbw_before and bbw_after
                                     else "before" if bbw_before else "after")

    if std_series is not None:
        std_before = std_after = 0
        for pos in range(start, end + 1):
            if std_series.iloc[pos] == "🐦‍🔥":
                if pos <= e1_pos: std_before += 1
                else:             std_after  += 1
        total = std_before + std_after
        if total > 0:
            out["std"]["present"] = True
            out["std"]["count"]   = total
            out["std"]["time"]    = ("both" if std_before and std_after
                                     else "before" if std_before else "after")

    return out


def extract_milestones(intraday: pd.DataFrame) -> dict:
    def first_hit(emoji_col, emoji_val):
        rows = intraday[intraday.get(emoji_col, "") == emoji_val]
        if len(rows) == 0:
            return {}
        row = rows.iloc[0]
        return {
            "Time":  pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%":    float(row["F_numeric"]),
        }

    gm_hits = intraday[intraday.get("Goldmine_E1_Emoji", "") == "💰"]
    goldmine = [
        {"Time": pd.to_datetime(r["Time"]).strftime("%H:%M"),
         "Price": float(r["Close"]), "F%": float(r["F_numeric"])}
        for _, r in gm_hits.iterrows()
    ]

    return {
        "T0":      first_hit("T0_Emoji",  "🚪"),
        "T1":      first_hit("T1_Emoji",  "🏇🏼"),
        "T2":      first_hit("T2_Emoji",  "⚡"),
        "goldmine": goldmine,
    }


def extract_market_profile(mp_df: pd.DataFrame | None) -> dict:
    """Returns only the IB-related market profile data — no nose, no ear."""
    if mp_df is None or mp_df.empty:
        return {}
    if "F% Level" not in mp_df.columns:
        return {}

    df = mp_df.copy()
    for c, default in [("TPO_Count", 0), ("%Vol", 0.0), ("🦻🏼", ""), ("👃🏽", "")]:
        if c not in df.columns:
            df[c] = default

    # Return only VAH / VAL / POC if present — skip nose/ear entirely
    out = {}
    for label, col in [("vah", "VAH"), ("val", "VAL"), ("poc", "POC")]:
        if col in df.columns:
            val = df[col].dropna()
            if not val.empty:
                out[label] = float(val.iloc[0])

    return out


def extract_range_extension(
    intraday: pd.DataFrame,
    ib_high_f: float | None,
    ib_low_f:  float | None,
    perimeter: int = 4,
) -> dict:
    """
    First bar breaking outside IB High or IB Low.
    Window ±perimeter bars → bishops (yellow/purple/green/red) + horses (RVOL) + z3On.
    """
    if intraday is None or intraday.empty or ib_high_f is None or ib_low_f is None:
        return {}

    f   = pd.to_numeric(intraday["F_numeric"], errors="coerce")
    n   = len(intraday)
    IB_END = 12

    # resolve column names once
    rv_col  = next((c for c in ("RVOL_5", "RVOL", "rvol") if c in intraday.columns), None)
    z3_col  = next((c for c in ("Z3_Score", "z3", "Z3", "Z3_score") if c in intraday.columns), None)
    has_bbw_tight = "BBW_Tight_Emoji" in intraday.columns
    has_std       = "STD_Alert"        in intraday.columns
    has_bbw_exp   = "BBW Alert"        in intraday.columns
    has_kijun     = "Kijun_F"          in intraday.columns

    def window_analysis(start: int, end: int) -> dict:
        bishops = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
        horses  = []

        for pos in range(max(0, start), min(n - 1, end) + 1):
            # Yellow bishop — BBW Tight
            if has_bbw_tight:
                val = intraday["BBW_Tight_Emoji"].iat[pos]
                if isinstance(val, str) and val.strip() == "🐝":
                    bishops["yellow"] += 1

            # Purple bishop — STD Expansion
            if has_std:
                val = intraday["STD_Alert"].iat[pos]
                if isinstance(val, str) and val.strip() not in ("", "nan"):
                    bishops["purple"] += 1

            # Green / Red bishop — BBW Expansion (color by Kijun)
            if has_bbw_exp:
                val = intraday["BBW Alert"].iat[pos]
                if isinstance(val, str) and val.strip() not in ("", "nan"):
                    fv = f.iat[pos]
                    kv = pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce") if has_kijun else float("nan")
                    if pd.notna(fv) and pd.notna(kv):
                        if fv >= kv: bishops["green"] += 1
                        else:        bishops["red"]   += 1
                    else:
                        bishops["green"] += 1  # fallback

            # Horse — RVOL
            if rv_col is not None:
                rv = pd.to_numeric(intraday[rv_col].iat[pos], errors="coerce")
                if pd.notna(rv) and rv > 1.2:
                    horses.append(round(float(rv), 2))

        return {
            "bishops": {k: v for k, v in bishops.items() if v > 0},
            "horses":  {"count": len(horses), "rvolValues": horses},
        }

    def scan_extension(direction: str) -> dict:
        ext_loc = None
        for i in range(IB_END, n):
            fv = f.iat[i]
            if pd.notna(fv):
                if direction == "high" and fv > ib_high_f:
                    ext_loc = i; break
                elif direction == "low" and fv < ib_low_f:
                    ext_loc = i; break

        if ext_loc is None:
            return {}

        pre_start = max(IB_END, ext_loc - perimeter)
        post_end  = min(n - 1,  ext_loc + perimeter)

        z3_val = None
        if z3_col is not None:
            z3_val = pd.to_numeric(intraday[z3_col].iat[ext_loc], errors="coerce")
        z3_on = bool(pd.notna(z3_val) and abs(float(z3_val)) >= 1.5)

        time_str = None
        try:
            time_str = pd.to_datetime(intraday["Time"].iat[ext_loc]).strftime("%H:%M")
        except:
            time_str = str(intraday["Time"].iat[ext_loc])

        return {
            "time":    time_str,
            "fLevel":  round(float(f.iat[ext_loc]), 2),
            "z3On":    z3_on,
            "z3Value": round(float(z3_val), 2) if pd.notna(z3_val) else None,
            "pre":     window_analysis(pre_start, ext_loc - 1),
            "post":    window_analysis(ext_loc + 1, post_end),
        }

    result = {}
    high_ext = scan_extension("high")
    low_ext  = scan_extension("low")
    if high_ext: result["aboveIBHigh"] = high_ext
    if low_ext:  result["belowIBLow"]  = low_ext
    return result


def build_basic_json(
    intraday: pd.DataFrame,
    ticker: str,
    mp_df: pd.DataFrame | None = None,
) -> dict:

    if intraday is None or intraday.empty:
        total_vol = 0
        last_date = date.today()
        total_vol_readable = "0"
    else:
        total_vol = int(intraday["Volume"].sum()) if "Volume" in intraday.columns else 0
        last_date = intraday["Date"].iloc[-1] if "Date" in intraday.columns else date.today()
        total_vol_readable = human_volume(total_vol)

    try:
        open_price = float(intraday["Open"].iloc[0])
    except:
        open_price = None

    try:
        close_price = float(intraday["Close"].iloc[-1])
    except:
        close_price = None

    # MIDAS BEAR
    try:
        bear_idx = intraday["MIDAS_Bear"].first_valid_index()
        if bear_idx is not None:
            midas_bear_time  = intraday.loc[bear_idx, "Time"]
            midas_bear_f     = float(intraday.loc[bear_idx, "F_numeric"])
            midas_bear_price = float(intraday.loc[bear_idx, "Close"])
        else:
            midas_bear_time = midas_bear_f = midas_bear_price = None
    except:
        midas_bear_time = midas_bear_f = midas_bear_price = None

    # MIDAS BULL
    try:
        bull_idx = intraday["MIDAS_Bull"].first_valid_index()
        if bull_idx is not None:
            midas_bull_time  = intraday.loc[bull_idx, "Time"]
            midas_bull_f     = float(intraday.loc[bull_idx, "F_numeric"])
            midas_bull_price = float(intraday.loc[bull_idx, "Close"])
        else:
            midas_bull_time = midas_bull_f = midas_bull_price = None
    except:
        midas_bull_time = midas_bull_f = midas_bull_price = None

    # INITIAL BALANCE
    try:
        ib_slice  = intraday.iloc[:12]
        ib_high_f = float(ib_slice["F_numeric"].max())
        ib_low_f  = float(ib_slice["F_numeric"].min())

        ib_high_row   = intraday.loc[intraday["F_numeric"] == ib_high_f].iloc[0]
        ib_high_time  = ib_high_row["Time"]
        ib_high_price = float(ib_high_row["Close"])

        ib_low_row    = intraday.loc[intraday["F_numeric"] == ib_low_f].iloc[0]
        ib_low_time   = ib_low_row["Time"]
        ib_low_price  = float(ib_low_row["Close"])
    except:
        ib_high_f = ib_low_f = None
        ib_high_time = ib_low_time = None
        ib_high_price = ib_low_price = None

    # IB zones (cellar / core / loft)
    ib_zones = {}
    if ib_high_f is not None and ib_low_f is not None:
        ib_range  = ib_high_f - ib_low_f
        ib_zones  = {
            "cellarTop": round(ib_low_f  + ib_range / 3,       2),
            "loftBottom": round(ib_low_f + 2 * ib_range / 3,   2),
        }

    sector = detect_sector(ticker)

    return round_all_numeric({
        "name":    str(ticker).lower(),
        "date":    str(last_date),
        "sector":  sector,
        "slug":    f"{ticker.lower()}-{last_date}-{sector}",
        "totalVolume": total_vol_readable,
        "open":  open_price,
        "close": close_price,

        "expansionInsight": detect_expansion_near_e1(intraday, perimeter=10),
        "entries":          extract_entries(intraday),
        "milestones":       extract_milestones(intraday),
        "marketProfile":    extract_market_profile(mp_df),
        "rangeExtension":   extract_range_extension(intraday, ib_high_f, ib_low_f, perimeter=4),

        "initialBalance": {
            "high":   {"time": ib_high_time,  "fLevel": ib_high_f,  "price": ib_high_price},
            "low":    {"time": ib_low_time,   "fLevel": ib_low_f,   "price": ib_low_price},
            "zones":  ib_zones,
        },

        "midas": {
            "bear": {"anchorTime": midas_bear_time,  "price": midas_bear_price,  "fLevel": midas_bear_f},
            "bull": {"anchorTime": midas_bull_time,  "price": midas_bull_price,  "fLevel": midas_bull_f},
        },
    })


def round_all_numeric(obj):
    if isinstance(obj, bool):   # must be before float check — bool is subclass of int
        return obj
    if isinstance(obj, dict):
        return {k: round_all_numeric(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_all_numeric(x) for x in obj]
    try:
        return round(float(obj), 2)
    except:
        return obj


def render_json_batch_download(json_map: dict):
    if not json_map:
        return

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for tkr, payload in json_map.items():
            safe_name = payload.get("name", str(tkr)).lower()
            date_str  = payload.get("date", "")
            fname     = f"{safe_name}-{date_str}.json" if date_str else f"{safe_name}.json"
            zf.writestr(fname, json.dumps(payload, indent=4, ensure_ascii=False))

    buffer.seek(0)
    st.download_button(
        label="⬇️ Download JSON batch",
        data=buffer,
        file_name="mike_json_batch.zip",
        mime="application/zip",
    )
