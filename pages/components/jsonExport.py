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
def extract_entries(intraday: pd.DataFrame, perimeter: int = 4) -> dict:
    call_entries = []
    put_entries  = []
    n = len(intraday)

    def col(name):
        return intraday[name] if name in intraday.columns else pd.Series("", index=intraday.index)

    # resolve columns once
    rv_col      = next((c for c in ("RVOL_5", "RVOL", "rvol") if c in intraday.columns), None)
    z3_col      = next((c for c in ("Z3_Score", "z3", "Z3", "Z3_score") if c in intraday.columns), None)
    has_tight   = "BBW_Tight_Emoji" in intraday.columns
    has_std     = "STD_Alert"        in intraday.columns
    has_bbw_exp = "BBW Alert"        in intraday.columns
    has_kijun   = "Kijun_F"          in intraday.columns
    f           = pd.to_numeric(intraday["F_numeric"], errors="coerce")

    def window_analysis(center_pos: int) -> dict:
        start = max(0, center_pos - perimeter)
        end   = min(n - 1, center_pos + perimeter)
        pre_bishops  = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
        post_bishops = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
        pre_horses, post_horses = [], []

        for pos in range(start, end + 1):
            if pos == center_pos:
                continue
            target_b = pre_bishops  if pos < center_pos else post_bishops
            target_h = pre_horses   if pos < center_pos else post_horses

            if has_tight:
                val = intraday["BBW_Tight_Emoji"].iat[pos]
                if isinstance(val, str) and val.strip() == "🐝":
                    target_b["yellow"] += 1

            if has_std:
                val = intraday["STD_Alert"].iat[pos]
                if isinstance(val, str) and val.strip() not in ("", "nan"):
                    target_b["purple"] += 1

            if has_bbw_exp:
                val = intraday["BBW Alert"].iat[pos]
                if isinstance(val, str) and val.strip() not in ("", "nan"):
                    fv = f.iat[pos]
                    kv = pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce") if has_kijun else float("nan")
                    if pd.notna(fv) and pd.notna(kv):
                        if fv >= kv: target_b["green"] += 1
                        else:        target_b["red"]   += 1
                    else:
                        target_b["green"] += 1

            if rv_col is not None:
                rv = pd.to_numeric(intraday[rv_col].iat[pos], errors="coerce")
                if pd.notna(rv) and rv > 1.2:
                    target_h.append(round(float(rv), 2))

        z3_on = False
        if z3_col is not None:
            for pos in range(max(0, center_pos - perimeter), min(n - 1, center_pos + perimeter) + 1):
                try:
                    zv = float(intraday[z3_col].iat[pos])
                    if zv >= 1.5 or zv <= -1.5:
                        z3_on = True
                        break
                except:
                    pass
        z3_col = next((c for c in ("Z3_Score", "z3", "Z3", "Z3_score") if c in intraday.columns), None)
        print("Z3 col found:", z3_col, "| columns:", [c for c in intraday.columns if "z3" in c.lower()])
        return {
            "pre":  {"bishops": {k: v for k, v in pre_bishops.items()  if v > 0}, "horses": {"count": len(pre_horses),  "rvolValues": pre_horses}},
            "post": {"bishops": {k: v for k, v in post_bishops.items() if v > 0}, "horses": {"count": len(post_horses), "rvolValues": post_horses}},
            "z3On": z3_on,
        }

    def add_entry(target, label, idx, with_perimeter=False):
        pos = intraday.index.get_loc(idx)
        row = {
            "type":   label,
            "time":   pd.to_datetime(intraday.at[idx, "Time"]).strftime("%H:%M"),
            "price":  float(intraday.at[idx, "Close"]),
            "fLevel": float(intraday.at[idx, "F_numeric"]),
        }
        if with_perimeter:
            row["perimeter"] = window_analysis(pos)
        target.append(row)

    # ── PUT ──────────────────────────────────────
    for i in intraday.index[col("Put_FirstEntry_Emoji") == "🎯"]:
        add_entry(put_entries, "Put E1 🎯", i, with_perimeter=True)

    for i in intraday.index[col("Put_FirstEntry_Emoji") == "⏳"]:
        add_entry(put_entries, "Put E1 ⏳ Blocked", i, with_perimeter=True)

    for i in intraday.index[col("Put_DeferredEntry_Emoji") == "🧿"]:
        pos = intraday.index.get_loc(i)
        put_entries.append({
            "type":      "Put Reclaim 🧿",
            "time":      pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
            "price":     float(intraday.at[i, "Close"]),
            "fLevel":    float(intraday.at[i, "F_numeric"]),
            "horse":     col("Put_DeferredReinforce_Emoji").at[i] == "❗️",
            "perimeter": window_analysis(pos),
        })

    for i in intraday.index[col("Put_SecondEntry_Emoji") == "🎯2"]:
        add_entry(put_entries, "Put E2 🎯2", i, with_perimeter=True)

    for i in intraday.index[col("Put_ThirdEntry_Emoji") == "🎯3"]:
        add_entry(put_entries, "Put E3 🎯3", i)

    # ── CALL ─────────────────────────────────────
    for i in intraday.index[col("Call_FirstEntry_Emoji") == "🎯"]:
        add_entry(call_entries, "Call E1 🎯", i, with_perimeter=True)

    for i in intraday.index[col("Call_FirstEntry_Emoji") == "⏳"]:
        add_entry(call_entries, "Call E1 ⏳ Blocked", i, with_perimeter=True)

    for i in intraday.index[col("Call_DeferredEntry_Emoji") == "🧿"]:
        pos = intraday.index.get_loc(i)
        call_entries.append({
            "type":      "Call Reclaim 🧿",
            "time":      pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
            "price":     float(intraday.at[i, "Close"]),
            "fLevel":    float(intraday.at[i, "F_numeric"]),
            "horse":     col("Call_DeferredReinforce_Emoji").at[i] == "❗️",
            "perimeter": window_analysis(pos),
        })

    for i in intraday.index[col("Call_SecondEntry_Emoji") == "🎯2"]:
        add_entry(call_entries, "Call E2 🎯2", i, with_perimeter=True)

    for i in intraday.index[col("Call_ThirdEntry_Emoji") == "🎯3"]:
        add_entry(call_entries, "Call E3 🎯3", i)

    return {"call": call_entries, "put": put_entries}

def detect_expansion_near_e1(
    intraday: pd.DataFrame,
    perimeter: int = 10
) -> dict:
    """
    Detect if BBW Alert (🔥) or STD Alert (🐦‍🔥) happened
    within +/- perimeter bars around Entry 1.

    Returns:
      {
        "bbw": {
          "present": True/False,
          "time": "before" / "after" / "both" / None,
          "count": int
        },
        "std": {
          "present": True/False,
          "time": "before" / "after" / "both" / None,
          "count": int
        }
      }

    NOTE:
      - "before" = at least one alert at or before E1, none after
      - "after"  = at least one alert after E1, none before
      - "both"   = alerts on both sides of E1
      - Bar == E1 is counted as "before" (change <= to < if you want strict).
    """

    out = {
        "bbw": {"present": False, "time": None, "count": 0},
        "std": {"present": False, "time": None, "count": 0},
    }

    if intraday is None or intraday.empty:
        return out

    # ---- Find Entry 1 (earliest of call/put) ----
    call_e1_idx = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1_idx  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    if len(call_e1_idx) == 0 and len(put_e1_idx) == 0:
        return out

    all_e1_idx = list(call_e1_idx) + list(put_e1_idx)
    # earliest by positional location
    e1_pos = min(intraday.index.get_loc(idx) for idx in all_e1_idx)

    # ---- Define perimeter window (positional) ----
    n = len(intraday)
    start = max(0, e1_pos - perimeter)
    end   = min(n - 1, e1_pos + perimeter)

    # Pre-grab alert series (if missing, there is no alert at all)
    bbw_series = intraday.get("BBW Alert")
    std_series = intraday.get("STD_Alert")

    # ---------- BBW 🔥 ----------
    if bbw_series is not None:
        bbw_before = 0
        bbw_after = 0

        for pos in range(start, end + 1):
            val = bbw_series.iloc[pos]
            if val == "🔥":
                # count bar on E1 as "before" (<=). Change to < if you prefer.
                if pos <= e1_pos:
                    bbw_before += 1
                else:
                    bbw_after += 1

        total = bbw_before + bbw_after
        if total > 0:
            out["bbw"]["present"] = True
            out["bbw"]["count"] = total

            if bbw_before > 0 and bbw_after == 0:
                out["bbw"]["time"] = "before"
            elif bbw_after > 0 and bbw_before == 0:
                out["bbw"]["time"] = "after"
            elif bbw_before > 0 and bbw_after > 0:
                out["bbw"]["time"] = "both"

    # ---------- STD 🐦‍🔥 ----------
    if std_series is not None:
        std_before = 0
        std_after = 0

        for pos in range(start, end + 1):
            val = std_series.iloc[pos]
            if val == "🐦‍🔥":
                if pos <= e1_pos:
                    std_before += 1
                else:
                    std_after += 1

        total = std_before + std_after
        if total > 0:
            out["std"]["present"] = True
            out["std"]["count"] = total

            if std_before > 0 and std_after == 0:
                out["std"]["time"] = "before"
            elif std_after > 0 and std_before == 0:
                out["std"]["time"] = "after"
            elif std_before > 0 and std_after > 0:
                out["std"]["time"] = "both"

    return out


def extract_milestones(intraday: pd.DataFrame) -> dict:
    """
    Extracts T0, T1, T2 and Goldmine_E1 hits into clean JSON-friendly dicts.
    - Missing values return {} (Mongo-safe)
    - Goldmine returns a list of hits
    """

    # --------------- T0 ----------------
    t0 = intraday[intraday.get("T0_Emoji", "") == "🚪"]
    if len(t0) > 0:
        row = t0.iloc[0]
        T0 = {
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        }
    else:
        T0 = {}

    # --------------- T1 ----------------
    t1 = intraday[intraday.get("T1_Emoji", "") == "🏇🏼"]
    if len(t1) > 0:
        row = t1.iloc[0]
        T1 = {
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        }
    else:
        T1 = {}

    # --------------- T2 ----------------
    t2 = intraday[intraday.get("T2_Emoji", "") == "⚡"]
    if len(t2) > 0:
        row = t2.iloc[0]
        T2 = {
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        }
    else:
        T2 = {}

    # --------------- GOLDMINE (E1 ladder) ----------------
    gm_hits = intraday[intraday.get("Goldmine_E1_Emoji", "") == "💰"]
    goldmine = []
    for _, row in gm_hits.iterrows():
        goldmine.append({
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        })

    return {
        "T0": T0,
        "T1": T1,
        "T2": T2,
        "goldmine": goldmine
    }



def extract_vector_capacitance(intraday: pd.DataFrame, perimeter: int = 5) -> dict:
    """
    Returns highest Vector_Capacitance BEFORE/AFTER Entry-1,
    separately for Call and Put.
    """

    if intraday.empty or "Vector_Capacitance" not in intraday.columns:
        return {"call": {}, "put": {}}

    def side_block(side: str, e1_idx):
        """Compute before/after for a specific entry index."""
        if e1_idx is None:
            return {}

        e1_loc = intraday.index.get_loc(e1_idx)

        start_before = max(0, e1_loc - perimeter)
        end_before   = e1_loc - 1

        start_after  = e1_loc + 1
        end_after    = min(len(intraday) - 1, e1_loc + perimeter)

        before_slice = intraday.iloc[start_before:end_before+1]
        after_slice  = intraday.iloc[start_after:end_after+1]

        # --- highest BEFORE ---
        before = {}
        if not before_slice.empty:
            temp = before_slice.dropna(subset=["Vector_Capacitance"])
            if not temp.empty:
                row = temp.loc[temp["Vector_Capacitance"].idxmax()]
                before = {
                    "time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
                    "value": float(row["Vector_Capacitance"])
                }

        # --- highest AFTER ---
        after = {}
        if not after_slice.empty:
            temp = after_slice.dropna(subset=["Vector_Capacitance"])
            if not temp.empty:
                row = temp.loc[temp["Vector_Capacitance"].idxmax()]
                after = {
                    "time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
                    "value": float(row["Vector_Capacitance"])
                }

        return {"before": before, "after": after}

    # Find entries for each side
    call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    call_idx = call_e1[0] if len(call_e1) > 0 else None
    put_idx  = put_e1[0]  if len(put_e1)  > 0 else None

    return {
        "call": side_block("call", call_idx),
        "put":  side_block("put",  put_idx)
    }
def extract_market_profile(mp_df: pd.DataFrame | None) -> dict:
    """
    Compact Nose/Ear block from Market Profile df.

    Returns:
      {
        "nose": {"fLevel": int, "tpoCount": int},
        "ear":  {"fLevel": int, "percentVol": float}
      }

    If mp_df is None/empty → {}.
    """
    if mp_df is None or mp_df.empty:
        return {}

    df = mp_df.copy()

    # Make sure columns exist
    if "F% Level" not in df.columns:
        return {}

    if "TPO_Count" not in df.columns:
        df["TPO_Count"] = 0
    if "%Vol" not in df.columns:
        df["%Vol"] = 0.0
    if "🦻🏼" not in df.columns:
        df["🦻🏼"] = ""
    if "👃🏽" not in df.columns:
        df["👃🏽"] = ""

    out: dict = {}

    # 👉 Nose (time POC)
    nose_row = df[df["👃🏽"] == "👃🏽"]
    if nose_row.empty:
        # fallback: max TPO_Count
        nose_row = df.sort_values(by="TPO_Count", ascending=False).head(1)

    if not nose_row.empty:
        out["nose"] = {
            "fLevel": int(nose_row["F% Level"].iloc[0]),
            "tpoCount": int(nose_row["TPO_Count"].iloc[0]),
        }

    # 👉 Ear (volume POC)
    ear_row = df[df["🦻🏼"] == "🦻🏼"]
    if ear_row.empty:
        # fallback: max %Vol
        ear_row = df.sort_values(by="%Vol", ascending=False).head(1)

    if not ear_row.empty:
        out["ear"] = {
            "fLevel": int(ear_row["F% Level"].iloc[0]),
            "percentVol": float(ear_row["%Vol"].iloc[0]),
        }

    return out
def extract_profile_cross_insight(
    intraday: pd.DataFrame,
    mp_block: dict | None,
    goldmine_dist: float = 64.0,
) -> dict:
    """
    Insight: did crossing the Market Profile level (Nose/Ear) in favor
    of the trade actually pay?

    For each level (Nose / Ear) we compute:

      call:
        - first Call 🎯1
        - if entry F < levelF → look for first bar where F >= levelF
        - after that cross, measure best F up-move from entry
        - check if move >= +goldmine_dist

      put:
        - first Put 🎯1
        - if entry F > levelF → look for first bar where F <= levelF
        - after that cross, measure best F down-move from entry
        - check if move >= goldmine_dist in favor of put

    Returns:
      {
        "nose": { "call": {...}, "put": {...} },
        "ear":  { "call": {...}, "put": {...} }
      }
    """
    if intraday is None or intraday.empty or not mp_block:
        return {}

    nose_info = mp_block.get("nose") or {}
    ear_info  = mp_block.get("ear")  or {}

    nose_f = nose_info.get("fLevel")
    ear_f  = ear_info.get("fLevel")

    if "F_numeric" not in intraday.columns or "Time" not in intraday.columns:
        return {}

    f = pd.to_numeric(intraday["F_numeric"], errors="coerce")
    t = pd.to_datetime(intraday["Time"], format="%I:%M %p", errors="coerce")

    def _compute_for_level(level_f: float | None) -> dict:
        """Return {'call': {...}, 'put': {...}} for a single F level."""
        if level_f is None:
            return {"call": {}, "put": {}}

        # ---------- CALL SIDE ----------
        def _side_call():
            call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
            if len(call_e1) == 0:
                return {}

            idx = call_e1[0]
            loc = intraday.index.get_loc(idx)
            entry_f = f.iloc[loc]
            if pd.isna(entry_f):
                return {}

            entry_time = (
                t.iloc[loc].strftime("%H:%M")
                if pd.notna(t.iloc[loc])
                else intraday["Time"].iloc[loc]
            )

            if entry_f < level_f:
                pos = "below"
            elif entry_f > level_f:
                pos = "above"
            else:
                pos = "on"

            crossed = "no"
            cross_time = None
            cross_f = None
            best_after = None
            best_move = None
            goldmine_flag = "no"

            # We care about bullish cross-up if entry is below level
            if entry_f < level_f:
                after = f.iloc[loc + 1 :]
                hit_mask = after >= level_f
                if hit_mask.any():
                    hit_idx = hit_mask[hit_mask].index[0]
                    hit_loc = intraday.index.get_loc(hit_idx)

                    cross_f_val = f.iloc[hit_loc]
                    cross_ts = t.iloc[hit_loc]
                    cross_time = (
                        cross_ts.strftime("%H:%M")
                        if pd.notna(cross_ts)
                        else intraday["Time"].iloc[hit_loc]
                    )
                    cross_f = float(cross_f_val) if pd.notna(cross_f_val) else None
                    crossed = "yes"

                    future = f.iloc[hit_loc:]
                    best_after_val = future.max()
                    if pd.notna(best_after_val):
                        best_after = float(best_after_val)
                        best_move = float(best_after - entry_f)
                        if best_move >= goldmine_dist:
                            goldmine_flag = "yes"

            return {
                "entryF": float(entry_f),
                "entryTime": entry_time,
                "levelF": float(level_f),
                "entryPositionVsLevel": pos,
                "crossedInFavor": crossed,          # "yes"/"no"
                "crossTime": cross_time,
                "crossF": cross_f,
                "bestFAfterCross": best_after,
                "bestMoveFromEntry": best_move,
                "goldmineLike64F": goldmine_flag,   # "yes"/"no"
            }

        # ---------- PUT SIDE ----------
        def _side_put():
            put_e1 = intraday.index[intraday.get("Put_FirstEntry_Emoji", "") == "🎯"]
            if len(put_e1) == 0:
                return {}

            idx = put_e1[0]
            loc = intraday.index.get_loc(idx)
            entry_f = f.iloc[loc]
            if pd.isna(entry_f):
                return {}

            entry_time = (
                t.iloc[loc].strftime("%H:%M")
                if pd.notna(t.iloc[loc])
                else intraday["Time"].iloc[loc]
            )

            if entry_f > level_f:
                pos = "above"
            elif entry_f < level_f:
                pos = "below"
            else:
                pos = "on"

            crossed = "no"
            cross_time = None
            cross_f = None
            best_after = None
            best_move = None
            goldmine_flag = "no"

            # Bearish cross-down if entry is above level
            if entry_f > level_f:
                after = f.iloc[loc + 1 :]
                hit_mask = after <= level_f
                if hit_mask.any():
                    hit_idx = hit_mask[hit_mask].index[0]
                    hit_loc = intraday.index.get_loc(hit_idx)

                    cross_f_val = f.iloc[hit_loc]
                    cross_ts = t.iloc[hit_loc]
                    cross_time = (
                        cross_ts.strftime("%H:%M")
                        if pd.notna(cross_ts)
                        else intraday["Time"].iloc[hit_loc]
                    )
                    cross_f = float(cross_f_val) if pd.notna(cross_f_val) else None
                    crossed = "yes"

                    future = f.iloc[hit_loc:]
                    worst_after = future.min()
                    if pd.notna(worst_after):
                        best_after = float(worst_after)
                        best_move = float(entry_f - worst_after)
                        if best_move >= goldmine_dist:
                            goldmine_flag = "yes"

            return {
                "entryF": float(entry_f),
                "entryTime": entry_time,
                "levelF": float(level_f),
                "entryPositionVsLevel": pos,
                "crossedInFavor": crossed,
                "crossTime": cross_time,
                "crossF": cross_f,
                "bestFAfterCross": best_after,
                "bestMoveFromEntry": best_move,
                "goldmineLike64F": goldmine_flag,
            }

        return {
            "call": _side_call(),
            "put": _side_put(),
        }

    return {
        "nose": _compute_for_level(nose_f),
        "ear":  _compute_for_level(ear_f),
    }


def extract_range_extension(
    intraday: pd.DataFrame,
    ib_high_f: float | None,
    ib_low_f: float | None,
    perimeter: int = 4,
) -> dict:
    """
    Finds the first bar where Mike breaks above IB High or below IB Low.
    For each direction, returns a ±perimeter window analysis:
      - bishops: yellow (BBW Tight), purple (STD Expansion), green/red (BBW Expansion)
      - horses: count + RVOL values of ♘ bars
      - z3On: was Z3 >= 1.5 or <= -1.5 at the extension bar
    """

    if intraday is None or intraday.empty or ib_high_f is None or ib_low_f is None:
        return {}

    f = pd.to_numeric(intraday["F_numeric"], errors="coerce")
    n = len(intraday)

    # Skip IB bars (first 12)
    IB_END = 12

    def scan_extension(direction: str) -> dict:
        """direction = 'high' or 'low'"""
        ext_loc = None
        for i in range(IB_END, n):
            fv = f.iloc[i]
            if not pd.isna(fv):
                if direction == "high" and fv > ib_high_f:
                    ext_loc = i
                    break
                elif direction == "low" and fv < ib_low_f:
                    ext_loc = i
                    break

        if ext_loc is None:
            return {}

        ext_bar = intraday.iloc[ext_loc]
        pre_start  = max(IB_END, ext_loc - perimeter)
        post_end   = min(n - 1, ext_loc + perimeter)

        def window_analysis(start: int, end: int) -> dict:
            bishops = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
            horses  = []

            for pos in range(start, end + 1):
                row = intraday.iloc[pos]

                # ── Bishops ──
                # Yellow = BBW Tight ♗
                if intraday.columns.__contains__("BBW_Tight_Emoji"):
                    if row.get("BBW_Tight_Emoji") == "🐝":
                        bishops["yellow"] += 1

                # Purple = STD Expansion ♗
                if "STD_Alert" in intraday.columns:
                    if str(row.get("STD_Alert", "")) not in ("", "nan"):
                        bishops["purple"] += 1

                # Green/Red = BBW Expansion ♗ (Kijun-based color)
                if "BBW Alert" in intraday.columns:
                    if str(row.get("BBW Alert", "")) not in ("", "nan"):
                        kijun = pd.to_numeric(row.get("Kijun_F"), errors="coerce")
                        fv    = pd.to_numeric(row.get("F_numeric"), errors="coerce")
                        if pd.notna(kijun) and pd.notna(fv):
                            if fv >= kijun:
                                bishops["green"] += 1
                            else:
                                bishops["red"] += 1
                        else:
                            bishops["green"] += 1  # fallback

                # ── Horses (RVOL ♘) ──
                for rv_col in ("RVOL_5", "RVOL", "rvol"):
                    if rv_col in intraday.columns:
                        rv = pd.to_numeric(row.get(rv_col), errors="coerce")
                        if pd.notna(rv) and rv > 1.2:
                            horses.append(round(float(rv), 2))
                        break

            return {
                "bishops": {k: v for k, v in bishops.items() if v > 0},
                "horses":  {"count": len(horses), "rvolValues": horses},
            }

        # Z3 at extension bar
        z3_val = None
        for z3_col in ("Z3_Score", "z3", "Z3", "Z3_score"):
            if z3_col in intraday.columns:
                z3_val = pd.to_numeric(ext_bar.get(z3_col), errors="coerce")
                break
        z3_on = bool(pd.notna(z3_val) and abs(z3_val) >= 1.5)

        return {
            "time":      pd.to_datetime(ext_bar["Time"]).strftime("%H:%M") if "Time" in ext_bar else None,
            "fLevel":    round(float(f.iloc[ext_loc]), 2),
            "z3On":      z3_on,
            "z3Value":   round(float(z3_val), 2) if pd.notna(z3_val) else None,
            "pre":       window_analysis(pre_start, ext_loc - 1),
            "post":      window_analysis(ext_loc + 1, post_end),
        }

    result = {}
    high_ext = scan_extension("high")
    low_ext  = scan_extension("low")
    if high_ext:
        result["aboveIBHigh"] = high_ext
    if low_ext:
        result["belowIBLow"] = low_ext

    return result


def build_basic_json(
    intraday: pd.DataFrame,
    ticker: str,
    mp_df: pd.DataFrame | None = None,
) -> dict:
    """
    Minimal JSON + MIDAS (anchor time, price, and F level)
    """

    # --- Base (same as original) ---
    if intraday is None or intraday.empty:
        total_vol = 0
        last_date = date.today()
    else:
        total_vol = int(intraday["Volume"].sum()) if "Volume" in intraday.columns else 0
        last_date = intraday["Date"].iloc[-1] if "Date" in intraday.columns else date.today()
        total_vol_readable = human_volume(total_vol)


    # ==========================
    # OPEN & CLOSE (real prices)
    # ==========================
    try:
        open_price = float(intraday["Open"].iloc[0])
    except:
        open_price = None

    try:
        close_price = float(intraday["Close"].iloc[-1])
    except:
        close_price = None

    # ==========================
    # MIDAS BEAR
    # ==========================
    try:
        bear_idx = intraday["MIDAS_Bear"].first_valid_index()
        if bear_idx is not None:
            midas_bear_time = intraday.loc[bear_idx, "Time"]
            midas_bear_f = float(intraday.loc[bear_idx, "F_numeric"])    # F level
            midas_bear_price = float(intraday.loc[bear_idx, "Close"])    # real price
        else:
            midas_bear_time = None
            midas_bear_f = None
            midas_bear_price = None
    except:
        midas_bear_time = midas_bear_f = midas_bear_price = None

    # ==========================
    # MIDAS BULL
    # ==========================
    try:
        bull_idx = intraday["MIDAS_Bull"].first_valid_index()
        if bull_idx is not None:
            midas_bull_time = intraday.loc[bull_idx, "Time"]
            midas_bull_f = float(intraday.loc[bull_idx, "F_numeric"])
            midas_bull_price = float(intraday.loc[bull_idx, "Close"])
        else:
            midas_bull_time = None
            midas_bull_f = None
            midas_bull_price = None
    except:
        midas_bull_time = midas_bull_f = midas_bull_price = None


    # ==========================
    # INITIAL BALANCE (IB)
    # ==========================
    try:
        # IB is always first 12 bars of intraday, matching your compute_initial_balance()
        ib_slice = intraday.iloc[:12]

        ib_high_f = float(ib_slice["F_numeric"].max())
        ib_low_f  = float(ib_slice["F_numeric"].min())

        # Locate time & real price for IB high
        ib_high_row = intraday.loc[intraday["F_numeric"] == ib_high_f].iloc[0]
        ib_high_time  = ib_high_row["Time"]
        ib_high_price = float(ib_high_row["Close"])

        # Locate time & real price for IB low
        ib_low_row  = intraday.loc[intraday["F_numeric"] == ib_low_f].iloc[0]
        ib_low_time   = ib_low_row["Time"]
        ib_low_price  = float(ib_low_row["Close"])

    except:
        ib_high_f = ib_low_f = None
        ib_high_time = ib_low_time = None
        ib_high_price = ib_low_price = None





    # ==========================
    # FINAL JSON
    # ==========================
    # ==========================
    # FINAL JSON
    # ==========================


    entries_block = extract_entries(intraday)
    milestones_block = extract_milestones(intraday)
    vector_cap_block = extract_vector_capacitance(intraday)
    mp_block = extract_market_profile(mp_df)
    sector = detect_sector(ticker)
    slug = f"{ticker.lower()}-{last_date}-{sector}"
    expansion_block = detect_expansion_near_e1(intraday, perimeter=10)
    range_ext_block = extract_range_extension(intraday, ib_high_f, ib_low_f, perimeter=4)

    return round_all_numeric({
        "name": str(ticker).lower(),
        "date": str(last_date),
        "sector": sector,

        "slug": slug,

        "totalVolume": total_vol_readable,

        "open": open_price,
        "close": close_price,
        "expansionInsight": expansion_block,

        "entries": entries_block,
        "milestones": milestones_block,
        "rangeExtension": range_ext_block,
        "initialBalance": {
            "high": {
                "time": ib_high_time,
                "fLevel": ib_high_f,
                "price": ib_high_price
            },
            "low": {
                "time": ib_low_time,
                "fLevel": ib_low_f,
                "price": ib_low_price
            }
        },

        "midas": {
            "bear": {
                "anchorTime": midas_bear_time,
                "price": midas_bear_price,
                "fLevel": midas_bear_f
            },
            "bull": {
                "anchorTime": midas_bull_time,
                "price": midas_bull_price,
                "fLevel": midas_bull_f
            }
        }
    })




def round_all_numeric(obj):
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {k: round_all_numeric(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_all_numeric(x) for x in obj]
    else:
        try:
            return round(float(obj), 2)
        except:
            return obj

def render_json_batch_download(json_map: dict):
    """
    json_map = {
      "NVDA": {...},
      "SPY":  {...},
      ...
    }
    Renders one ZIP download button with all JSON files inside.
    """
    if not json_map:
        return

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for tkr, payload in json_map.items():
            # filename pattern: nvda-2025-11-21.json
            safe_name = payload.get("name", str(tkr)).lower()
            date_str = payload.get("date", "")
            if date_str:
                fname = f"{safe_name}-{date_str}.json"
            else:
                fname = f"{safe_name}.json"

            zf.writestr(fname, json.dumps(payload, indent=4, ensure_ascii=False)
)

    buffer.seek(0)

    st.download_button(
        label="⬇️ Download JSON batch",
        data=buffer,
        file_name="mike_json_batch.zip",
        mime="application/zip",
    )