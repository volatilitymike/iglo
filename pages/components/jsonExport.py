# # pages/components/jsonExport.py

# import io
# import json
# import zipfile
# from datetime import date

# import pandas as pd
# import streamlit as st



# SECTOR_MAP = {
#     "ETFs": ["spy", "qqq"],
#     "finance": ["wfc", "c", "jpm", "bac", "hood", "coin", "pypl"],
#     "Semiconductors": ["nvda", "avgo", "amd", "mu", "mrvl", "qcom", "smci"],
#     "Software": ["msft", "pltr", "aapl", "googl", "meta", "uber", "tsla", "amzn"],
#     "Futures": ["nq", "es", "gc", "ym", "cl"],
# }
# def detect_sector(ticker: str) -> str:
#     t = ticker.lower()
#     for sector, listing in SECTOR_MAP.items():
#         if t in listing:
#             return sector
#     return "Other"

# def human_volume(n):
#     try:
#         n = float(n)
#     except:
#         return n

#     if n >= 1_000_000_000:
#         return f"{n/1_000_000_000:.2f}B"
#     elif n >= 1_000_000:
#         return f"{n/1_000_000:.2f}M"
#     elif n >= 1_000:
#         return f"{n/1_000:.2f}K"
#     else:
#         return str(int(n))
# def extract_entries(intraday: pd.DataFrame, perimeter: int = 4) -> dict:
#     call_entries = []
#     put_entries  = []
#     n = len(intraday)

#     def col(name):
#         return intraday[name] if name in intraday.columns else pd.Series("", index=intraday.index)

#     # resolve columns once
#     rv_col      = next((c for c in ("RVOL_5", "RVOL", "rvol") if c in intraday.columns), None)
#     z3_col      = next((c for c in ("Z3_Score", "z3", "Z3", "Z3_score") if c in intraday.columns), None)
#     has_tight   = "BBW_Tight_Emoji" in intraday.columns
#     has_std     = "STD_Alert"        in intraday.columns
#     has_bbw_exp = "BBW Alert"        in intraday.columns
#     has_kijun   = "Kijun_F"          in intraday.columns
#     f           = pd.to_numeric(intraday["F_numeric"], errors="coerce")

#     def window_analysis(center_pos: int) -> dict:
#         start = max(0, center_pos - perimeter)
#         end   = min(n - 1, center_pos + perimeter)
#         pre_bishops  = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
#         post_bishops = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
#         pre_horses, post_horses = [], []

#         for pos in range(start, end + 1):
#             if pos == center_pos:
#                 continue
#             target_b = pre_bishops  if pos < center_pos else post_bishops
#             target_h = pre_horses   if pos < center_pos else post_horses

#             if has_tight:
#                 val = intraday["BBW_Tight_Emoji"].iat[pos]
#                 if isinstance(val, str) and val.strip() == "🐝":
#                     target_b["yellow"] += 1

#             if has_std:
#                 val = intraday["STD_Alert"].iat[pos]
#                 if isinstance(val, str) and val.strip() not in ("", "nan"):
#                     target_b["purple"] += 1

#             if has_bbw_exp:
#                 val = intraday["BBW Alert"].iat[pos]
#                 if isinstance(val, str) and val.strip() not in ("", "nan"):
#                     fv = f.iat[pos]
#                     kv = pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce") if has_kijun else float("nan")
#                     if pd.notna(fv) and pd.notna(kv):
#                         if fv >= kv: target_b["green"] += 1
#                         else:        target_b["red"]   += 1
#                     else:
#                         target_b["green"] += 1

#             if rv_col is not None:
#                 rv = pd.to_numeric(intraday[rv_col].iat[pos], errors="coerce")
#                 if pd.notna(rv) and rv > 1.2:
#                     target_h.append(round(float(rv), 2))

#             def pick_time_col(df):
#                 for c in ["time", "Time", "datetime", "Datetime", "timestamp", "Timestamp", "DateTime"]:
#                     if c in df.columns:
#                         return c
#                 return None


#             z3_on = False
#             z3_value = None
#             z3_last3_value = None
#             z3_max_time = None
#             z3_max_bars_from_entry = None

#             time_col = pick_time_col(intraday)

#             if z3_col is not None:
#                 lo = max(0, center_pos - perimeter)
#                 hi = min(n - 1, center_pos + perimeter)

#                 # window (same window used to decide z3On)
#                 s = pd.to_numeric(intraday[z3_col].iloc[lo : hi + 1], errors="coerce")

#                 if s.notna().any():
#                     idx = s.abs().idxmax()                  # <-- index label in df
#                     z3_value = float(s.loc[idx])
#                     z3_on = abs(z3_value) >= 1.5

#                     # timing fields
#                     z3_max_bars_from_entry = int(idx) - int(center_pos)
#                     if time_col is not None:
#                         z3_max_time = str(intraday[time_col].iat[int(idx)])

#                 # "last 3" bars after entry (entry bar + next 2 bars)
#                 s3 = pd.to_numeric(
#                     intraday[z3_col].iloc[center_pos : min(n - 1, center_pos + 2) + 1],
#                     errors="coerce",
#                 )
#                 if s3.notna().any():
#                     idx3 = s3.abs().idxmax()
#                     z3_last3_value = float(s3.loc[idx3])

#             return {
#                 "pre":  {"bishops": {k: v for k, v in pre_bishops.items()  if v > 0}, "horses": {"count": len(pre_horses),  "rvolValues": pre_horses}},
#                 "post": {"bishops": {k: v for k, v in post_bishops.items() if v > 0}, "horses": {"count": len(post_horses), "rvolValues": post_horses}},
#                 "z3On": z3_on,
#                 "z3Value": z3_value,
#                 "z3ValueLast3": z3_last3_value,
#                 "z3MaxTime": z3_max_time,
#                 "z3MaxBarsFromEntry": z3_max_bars_from_entry,
#             }

#     def add_entry(target, label, idx, with_perimeter=False):
#         pos = intraday.index.get_loc(idx)
#         row = {
#             "type":   label,
#             "time":   pd.to_datetime(intraday.at[idx, "Time"]).strftime("%H:%M"),
#             "price":  float(intraday.at[idx, "Close"]),
#             "fLevel": float(intraday.at[idx, "F_numeric"]),
#         }
#         if with_perimeter:
#             row["perimeter"] = window_analysis(pos)
#         target.append(row)

#     # ── PUT ──────────────────────────────────────
#     for i in intraday.index[col("Put_FirstEntry_Emoji") == "🎯"]:
#         add_entry(put_entries, "Put E1 🎯", i, with_perimeter=True)

#     for i in intraday.index[col("Put_FirstEntry_Emoji") == "⏳"]:
#         add_entry(put_entries, "Put E1 ⏳ Blocked", i, with_perimeter=True)

#     for i in intraday.index[col("Put_DeferredEntry_Emoji") == "🧿"]:
#         pos = intraday.index.get_loc(i)
#         put_entries.append({
#             "type":      "Put Reclaim 🧿",
#             "time":      pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
#             "price":     float(intraday.at[i, "Close"]),
#             "fLevel":    float(intraday.at[i, "F_numeric"]),
#             "horse":     col("Put_DeferredReinforce_Emoji").at[i] == "❗️",
#             "perimeter": window_analysis(pos),
#         })

#     for i in intraday.index[col("Put_SecondEntry_Emoji") == "🎯2"]:
#         add_entry(put_entries, "Put E2 🎯2", i, with_perimeter=True)

#     for i in intraday.index[col("Put_ThirdEntry_Emoji") == "🎯3"]:
#         add_entry(put_entries, "Put E3 🎯3", i)

#     # ── CALL ─────────────────────────────────────
#     for i in intraday.index[col("Call_FirstEntry_Emoji") == "🎯"]:
#         add_entry(call_entries, "Call E1 🎯", i, with_perimeter=True)

#     for i in intraday.index[col("Call_FirstEntry_Emoji") == "⏳"]:
#         add_entry(call_entries, "Call E1 ⏳ Blocked", i, with_perimeter=True)

#     for i in intraday.index[col("Call_DeferredEntry_Emoji") == "🧿"]:
#         pos = intraday.index.get_loc(i)
#         call_entries.append({
#             "type":      "Call Reclaim 🧿",
#             "time":      pd.to_datetime(intraday.at[i, "Time"]).strftime("%H:%M"),
#             "price":     float(intraday.at[i, "Close"]),
#             "fLevel":    float(intraday.at[i, "F_numeric"]),
#             "horse":     col("Call_DeferredReinforce_Emoji").at[i] == "❗️",
#             "perimeter": window_analysis(pos),
#         })

#     for i in intraday.index[col("Call_SecondEntry_Emoji") == "🎯2"]:
#         add_entry(call_entries, "Call E2 🎯2", i, with_perimeter=True)

#     for i in intraday.index[col("Call_ThirdEntry_Emoji") == "🎯3"]:
#         add_entry(call_entries, "Call E3 🎯3", i)

#     return {"call": call_entries, "put": put_entries}

# def detect_expansion_near_e1(
#     intraday: pd.DataFrame,
#     perimeter: int = 10
# ) -> dict:
#     """
#     Detect if BBW Alert (🔥) or STD Alert (🐦‍🔥) happened
#     within +/- perimeter bars around Entry 1.

#     Returns:
#       {
#         "bbw": {
#           "present": True/False,
#           "time": "before" / "after" / "both" / None,
#           "count": int
#         },
#         "std": {
#           "present": True/False,
#           "time": "before" / "after" / "both" / None,
#           "count": int
#         }
#       }

#     NOTE:
#       - "before" = at least one alert at or before E1, none after
#       - "after"  = at least one alert after E1, none before
#       - "both"   = alerts on both sides of E1
#       - Bar == E1 is counted as "before" (change <= to < if you want strict).
#     """

#     out = {
#         "bbw": {"present": False, "time": None, "count": 0},
#         "std": {"present": False, "time": None, "count": 0},
#     }

#     if intraday is None or intraday.empty:
#         return out

#     # ---- Find Entry 1 (earliest of call/put) ----
#     call_e1_idx = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
#     put_e1_idx  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

#     if len(call_e1_idx) == 0 and len(put_e1_idx) == 0:
#         return out

#     all_e1_idx = list(call_e1_idx) + list(put_e1_idx)
#     # earliest by positional location
#     e1_pos = min(intraday.index.get_loc(idx) for idx in all_e1_idx)

#     # ---- Define perimeter window (positional) ----
#     n = len(intraday)
#     start = max(0, e1_pos - perimeter)
#     end   = min(n - 1, e1_pos + perimeter)

#     # Pre-grab alert series (if missing, there is no alert at all)
#     bbw_series = intraday.get("BBW Alert")
#     std_series = intraday.get("STD_Alert")

#     # ---------- BBW 🔥 ----------
#     if bbw_series is not None:
#         bbw_before = 0
#         bbw_after = 0

#         for pos in range(start, end + 1):
#             val = bbw_series.iloc[pos]
#             if val == "🔥":
#                 # count bar on E1 as "before" (<=). Change to < if you prefer.
#                 if pos <= e1_pos:
#                     bbw_before += 1
#                 else:
#                     bbw_after += 1

#         total = bbw_before + bbw_after
#         if total > 0:
#             out["bbw"]["present"] = True
#             out["bbw"]["count"] = total

#             if bbw_before > 0 and bbw_after == 0:
#                 out["bbw"]["time"] = "before"
#             elif bbw_after > 0 and bbw_before == 0:
#                 out["bbw"]["time"] = "after"
#             elif bbw_before > 0 and bbw_after > 0:
#                 out["bbw"]["time"] = "both"

#     # ---------- STD 🐦‍🔥 ----------
#     if std_series is not None:
#         std_before = 0
#         std_after = 0

#         for pos in range(start, end + 1):
#             val = std_series.iloc[pos]
#             if val == "🐦‍🔥":
#                 if pos <= e1_pos:
#                     std_before += 1
#                 else:
#                     std_after += 1

#         total = std_before + std_after
#         if total > 0:
#             out["std"]["present"] = True
#             out["std"]["count"] = total

#             if std_before > 0 and std_after == 0:
#                 out["std"]["time"] = "before"
#             elif std_after > 0 and std_before == 0:
#                 out["std"]["time"] = "after"
#             elif std_before > 0 and std_after > 0:
#                 out["std"]["time"] = "both"

#     return out


# def extract_milestones(intraday: pd.DataFrame) -> dict:
#     """
#     Extracts T0, T1, T2 and Goldmine_E1 hits into clean JSON-friendly dicts.
#     - Missing values return {} (Mongo-safe)
#     - Goldmine returns a list of hits
#     """

#     # --------------- T0 ----------------
#     t0 = intraday[intraday.get("T0_Emoji", "") == "🚪"]
#     if len(t0) > 0:
#         row = t0.iloc[0]
#         T0 = {
#             "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
#             "Price": float(row["Close"]),
#             "F%": float(row["F_numeric"]),
#         }
#     else:
#         T0 = {}

#     # --------------- T1 ----------------
#     t1 = intraday[intraday.get("T1_Emoji", "") == "🏇🏼"]
#     if len(t1) > 0:
#         row = t1.iloc[0]
#         T1 = {
#             "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
#             "Price": float(row["Close"]),
#             "F%": float(row["F_numeric"]),
#         }
#     else:
#         T1 = {}

#     # --------------- T2 ----------------
#     t2 = intraday[intraday.get("T2_Emoji", "") == "⚡"]
#     if len(t2) > 0:
#         row = t2.iloc[0]
#         T2 = {
#             "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
#             "Price": float(row["Close"]),
#             "F%": float(row["F_numeric"]),
#         }
#     else:
#         T2 = {}

#     # --------------- GOLDMINE (E1 ladder) ----------------
#     gm_hits = intraday[intraday.get("Goldmine_E1_Emoji", "") == "💰"]
#     goldmine = []
#     for _, row in gm_hits.iterrows():
#         goldmine.append({
#             "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
#             "Price": float(row["Close"]),
#             "F%": float(row["F_numeric"]),
#         })

#     return {
#         "T0": T0,
#         "T1": T1,
#         "T2": T2,
#         "goldmine": goldmine
#     }



# def extract_vector_capacitance(intraday: pd.DataFrame, perimeter: int = 5) -> dict:
#     """
#     Returns highest Vector_Capacitance BEFORE/AFTER Entry-1,
#     separately for Call and Put.
#     """

#     if intraday.empty or "Vector_Capacitance" not in intraday.columns:
#         return {"call": {}, "put": {}}

#     def side_block(side: str, e1_idx):
#         """Compute before/after for a specific entry index."""
#         if e1_idx is None:
#             return {}

#         e1_loc = intraday.index.get_loc(e1_idx)

#         start_before = max(0, e1_loc - perimeter)
#         end_before   = e1_loc - 1

#         start_after  = e1_loc + 1
#         end_after    = min(len(intraday) - 1, e1_loc + perimeter)

#         before_slice = intraday.iloc[start_before:end_before+1]
#         after_slice  = intraday.iloc[start_after:end_after+1]

#         # --- highest BEFORE ---
#         before = {}
#         if not before_slice.empty:
#             temp = before_slice.dropna(subset=["Vector_Capacitance"])
#             if not temp.empty:
#                 row = temp.loc[temp["Vector_Capacitance"].idxmax()]
#                 before = {
#                     "time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
#                     "value": float(row["Vector_Capacitance"])
#                 }

#         # --- highest AFTER ---
#         after = {}
#         if not after_slice.empty:
#             temp = after_slice.dropna(subset=["Vector_Capacitance"])
#             if not temp.empty:
#                 row = temp.loc[temp["Vector_Capacitance"].idxmax()]
#                 after = {
#                     "time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
#                     "value": float(row["Vector_Capacitance"])
#                 }

#         return {"before": before, "after": after}

#     # Find entries for each side
#     call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
#     put_e1  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

#     call_idx = call_e1[0] if len(call_e1) > 0 else None
#     put_idx  = put_e1[0]  if len(put_e1)  > 0 else None

#     return {
#         "call": side_block("call", call_idx),
#         "put":  side_block("put",  put_idx)
#     }
# def extract_market_profile(mp_df: pd.DataFrame | None) -> dict:
#     """
#     Compact Nose/Ear block from Market Profile df.

#     Returns:
#       {
#         "nose": {"fLevel": int, "tpoCount": int},
#         "ear":  {"fLevel": int, "percentVol": float}
#       }

#     If mp_df is None/empty → {}.
#     """
#     if mp_df is None or mp_df.empty:
#         return {}

#     df = mp_df.copy()

#     # Make sure columns exist
#     if "F% Level" not in df.columns:
#         return {}

#     if "TPO_Count" not in df.columns:
#         df["TPO_Count"] = 0
#     if "%Vol" not in df.columns:
#         df["%Vol"] = 0.0
#     if "🦻🏼" not in df.columns:
#         df["🦻🏼"] = ""
#     if "👃🏽" not in df.columns:
#         df["👃🏽"] = ""

#     out: dict = {}

#     # 👉 Nose (time POC)
#     nose_row = df[df["👃🏽"] == "👃🏽"]
#     if nose_row.empty:
#         # fallback: max TPO_Count
#         nose_row = df.sort_values(by="TPO_Count", ascending=False).head(1)

#     if not nose_row.empty:
#         out["nose"] = {
#             "fLevel": int(nose_row["F% Level"].iloc[0]),
#             "tpoCount": int(nose_row["TPO_Count"].iloc[0]),
#         }

#     # 👉 Ear (volume POC)
#     ear_row = df[df["🦻🏼"] == "🦻🏼"]
#     if ear_row.empty:
#         # fallback: max %Vol
#         ear_row = df.sort_values(by="%Vol", ascending=False).head(1)

#     if not ear_row.empty:
#         out["ear"] = {
#             "fLevel": int(ear_row["F% Level"].iloc[0]),
#             "percentVol": float(ear_row["%Vol"].iloc[0]),
#         }

#     return out
# def extract_profile_cross_insight(
#     intraday: pd.DataFrame,
#     mp_block: dict | None,
#     goldmine_dist: float = 64.0,
# ) -> dict:
#     """
#     Insight: did crossing the Market Profile level (Nose/Ear) in favor
#     of the trade actually pay?

#     For each level (Nose / Ear) we compute:

#       call:
#         - first Call 🎯1
#         - if entry F < levelF → look for first bar where F >= levelF
#         - after that cross, measure best F up-move from entry
#         - check if move >= +goldmine_dist

#       put:
#         - first Put 🎯1
#         - if entry F > levelF → look for first bar where F <= levelF
#         - after that cross, measure best F down-move from entry
#         - check if move >= goldmine_dist in favor of put

#     Returns:
#       {
#         "nose": { "call": {...}, "put": {...} },
#         "ear":  { "call": {...}, "put": {...} }
#       }
#     """
#     if intraday is None or intraday.empty or not mp_block:
#         return {}

#     nose_info = mp_block.get("nose") or {}
#     ear_info  = mp_block.get("ear")  or {}

#     nose_f = nose_info.get("fLevel")
#     ear_f  = ear_info.get("fLevel")

#     if "F_numeric" not in intraday.columns or "Time" not in intraday.columns:
#         return {}

#     f = pd.to_numeric(intraday["F_numeric"], errors="coerce")
#     t = pd.to_datetime(intraday["Time"], format="%I:%M %p", errors="coerce")

#     def _compute_for_level(level_f: float | None) -> dict:
#         """Return {'call': {...}, 'put': {...}} for a single F level."""
#         if level_f is None:
#             return {"call": {}, "put": {}}

#         # ---------- CALL SIDE ----------
#         def _side_call():
#             call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
#             if len(call_e1) == 0:
#                 return {}

#             idx = call_e1[0]
#             loc = intraday.index.get_loc(idx)
#             entry_f = f.iloc[loc]
#             if pd.isna(entry_f):
#                 return {}

#             entry_time = (
#                 t.iloc[loc].strftime("%H:%M")
#                 if pd.notna(t.iloc[loc])
#                 else intraday["Time"].iloc[loc]
#             )

#             if entry_f < level_f:
#                 pos = "below"
#             elif entry_f > level_f:
#                 pos = "above"
#             else:
#                 pos = "on"

#             crossed = "no"
#             cross_time = None
#             cross_f = None
#             best_after = None
#             best_move = None
#             goldmine_flag = "no"

#             # We care about bullish cross-up if entry is below level
#             if entry_f < level_f:
#                 after = f.iloc[loc + 1 :]
#                 hit_mask = after >= level_f
#                 if hit_mask.any():
#                     hit_idx = hit_mask[hit_mask].index[0]
#                     hit_loc = intraday.index.get_loc(hit_idx)

#                     cross_f_val = f.iloc[hit_loc]
#                     cross_ts = t.iloc[hit_loc]
#                     cross_time = (
#                         cross_ts.strftime("%H:%M")
#                         if pd.notna(cross_ts)
#                         else intraday["Time"].iloc[hit_loc]
#                     )
#                     cross_f = float(cross_f_val) if pd.notna(cross_f_val) else None
#                     crossed = "yes"

#                     future = f.iloc[hit_loc:]
#                     best_after_val = future.max()
#                     if pd.notna(best_after_val):
#                         best_after = float(best_after_val)
#                         best_move = float(best_after - entry_f)
#                         if best_move >= goldmine_dist:
#                             goldmine_flag = "yes"

#             return {
#                 "entryF": float(entry_f),
#                 "entryTime": entry_time,
#                 "levelF": float(level_f),
#                 "entryPositionVsLevel": pos,
#                 "crossedInFavor": crossed,          # "yes"/"no"
#                 "crossTime": cross_time,
#                 "crossF": cross_f,
#                 "bestFAfterCross": best_after,
#                 "bestMoveFromEntry": best_move,
#                 "goldmineLike64F": goldmine_flag,   # "yes"/"no"
#             }

#         # ---------- PUT SIDE ----------
#         def _side_put():
#             put_e1 = intraday.index[intraday.get("Put_FirstEntry_Emoji", "") == "🎯"]
#             if len(put_e1) == 0:
#                 return {}

#             idx = put_e1[0]
#             loc = intraday.index.get_loc(idx)
#             entry_f = f.iloc[loc]
#             if pd.isna(entry_f):
#                 return {}

#             entry_time = (
#                 t.iloc[loc].strftime("%H:%M")
#                 if pd.notna(t.iloc[loc])
#                 else intraday["Time"].iloc[loc]
#             )

#             if entry_f > level_f:
#                 pos = "above"
#             elif entry_f < level_f:
#                 pos = "below"
#             else:
#                 pos = "on"

#             crossed = "no"
#             cross_time = None
#             cross_f = None
#             best_after = None
#             best_move = None
#             goldmine_flag = "no"

#             # Bearish cross-down if entry is above level
#             if entry_f > level_f:
#                 after = f.iloc[loc + 1 :]
#                 hit_mask = after <= level_f
#                 if hit_mask.any():
#                     hit_idx = hit_mask[hit_mask].index[0]
#                     hit_loc = intraday.index.get_loc(hit_idx)

#                     cross_f_val = f.iloc[hit_loc]
#                     cross_ts = t.iloc[hit_loc]
#                     cross_time = (
#                         cross_ts.strftime("%H:%M")
#                         if pd.notna(cross_ts)
#                         else intraday["Time"].iloc[hit_loc]
#                     )
#                     cross_f = float(cross_f_val) if pd.notna(cross_f_val) else None
#                     crossed = "yes"

#                     future = f.iloc[hit_loc:]
#                     worst_after = future.min()
#                     if pd.notna(worst_after):
#                         best_after = float(worst_after)
#                         best_move = float(entry_f - worst_after)
#                         if best_move >= goldmine_dist:
#                             goldmine_flag = "yes"

#             return {
#                 "entryF": float(entry_f),
#                 "entryTime": entry_time,
#                 "levelF": float(level_f),
#                 "entryPositionVsLevel": pos,
#                 "crossedInFavor": crossed,
#                 "crossTime": cross_time,
#                 "crossF": cross_f,
#                 "bestFAfterCross": best_after,
#                 "bestMoveFromEntry": best_move,
#                 "goldmineLike64F": goldmine_flag,
#             }

#         return {
#             "call": _side_call(),
#             "put": _side_put(),
#         }

#     return {
#         "nose": _compute_for_level(nose_f),
#         "ear":  _compute_for_level(ear_f),
#     }


# def extract_range_extension(
#     intraday: pd.DataFrame,
#     ib_high_f: float | None,
#     ib_low_f: float | None,
#     perimeter: int = 4,
# ) -> dict:
#     """
#     Finds the first bar where Mike breaks above IB High or below IB Low.
#     For each direction, returns a ±perimeter window analysis:
#       - bishops: yellow (BBW Tight), purple (STD Expansion), green/red (BBW Expansion)
#       - horses: count + RVOL values of ♘ bars
#       - z3On: was Z3 >= 1.5 or <= -1.5 at the extension bar
#     """

#     if intraday is None or intraday.empty or ib_high_f is None or ib_low_f is None:
#         return {}

#     f = pd.to_numeric(intraday["F_numeric"], errors="coerce")
#     n = len(intraday)

#     # Skip IB bars (first 12)
#     IB_END = 12

#     def scan_extension(direction: str) -> dict:
#         """direction = 'high' or 'low'"""
#         ext_loc = None
#         for i in range(IB_END, n):
#             fv = f.iloc[i]
#             if not pd.isna(fv):
#                 if direction == "high" and fv > ib_high_f:
#                     ext_loc = i
#                     break
#                 elif direction == "low" and fv < ib_low_f:
#                     ext_loc = i
#                     break

#         if ext_loc is None:
#             return {}

#         ext_bar = intraday.iloc[ext_loc]
#         pre_start  = max(IB_END, ext_loc - perimeter)
#         post_end   = min(n - 1, ext_loc + perimeter)

#         def window_analysis(start: int, end: int) -> dict:
#             bishops = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
#             horses  = []

#             for pos in range(start, end + 1):
#                 row = intraday.iloc[pos]

#                 # ── Bishops ──
#                 # Yellow = BBW Tight ♗
#                 if intraday.columns.__contains__("BBW_Tight_Emoji"):
#                     if row.get("BBW_Tight_Emoji") == "🐝":
#                         bishops["yellow"] += 1

#                 # Purple = STD Expansion ♗
#                 if "STD_Alert" in intraday.columns:
#                     if str(row.get("STD_Alert", "")) not in ("", "nan"):
#                         bishops["purple"] += 1

#                 # Green/Red = BBW Expansion ♗ (Kijun-based color)
#                 if "BBW Alert" in intraday.columns:
#                     if str(row.get("BBW Alert", "")) not in ("", "nan"):
#                         kijun = pd.to_numeric(row.get("Kijun_F"), errors="coerce")
#                         fv    = pd.to_numeric(row.get("F_numeric"), errors="coerce")
#                         if pd.notna(kijun) and pd.notna(fv):
#                             if fv >= kijun:
#                                 bishops["green"] += 1
#                             else:
#                                 bishops["red"] += 1
#                         else:
#                             bishops["green"] += 1  # fallback

#                 # ── Horses (RVOL ♘) ──
#                 for rv_col in ("RVOL_5", "RVOL", "rvol"):
#                     if rv_col in intraday.columns:
#                         rv = pd.to_numeric(row.get(rv_col), errors="coerce")
#                         if pd.notna(rv) and rv > 1.2:
#                             horses.append(round(float(rv), 2))
#                         break

#             return {
#                 "bishops": {k: v for k, v in bishops.items() if v > 0},
#                 "horses":  {"count": len(horses), "rvolValues": horses},
#             }

#         # Z3 at extension bar
#         z3_val = None
#         for z3_col in ("Z3_Score", "z3", "Z3", "Z3_score"):
#             if z3_col in intraday.columns:
#                 z3_val = pd.to_numeric(ext_bar.get(z3_col), errors="coerce")
#                 break
#         z3_on = bool(pd.notna(z3_val) and abs(z3_val) >= 1.5)

#         return {
#             "time":      pd.to_datetime(ext_bar["Time"]).strftime("%H:%M") if "Time" in ext_bar else None,
#             "fLevel":    round(float(f.iloc[ext_loc]), 2),
#             "z3On":      z3_on,
#             "z3Value":   round(float(z3_val), 2) if pd.notna(z3_val) else None,
#             "pre":       window_analysis(pre_start, ext_loc - 1),
#             "post":      window_analysis(ext_loc + 1, post_end),
#         }

#     result = {}
#     high_ext = scan_extension("high")
#     low_ext  = scan_extension("low")
#     if high_ext:
#         result["aboveIBHigh"] = high_ext
#     if low_ext:
#         result["belowIBLow"] = low_ext

#     return result


# def build_basic_json(
#     intraday: pd.DataFrame,
#     ticker: str,
#     mp_df: pd.DataFrame | None = None,
# ) -> dict:
#     """
#     Minimal JSON + MIDAS (anchor time, price, and F level)
#     """

#     # --- Base (same as original) ---
#     if intraday is None or intraday.empty:
#         total_vol = 0
#         last_date = date.today()
#     else:
#         total_vol = int(intraday["Volume"].sum()) if "Volume" in intraday.columns else 0
#         last_date = intraday["Date"].iloc[-1] if "Date" in intraday.columns else date.today()
#         total_vol_readable = human_volume(total_vol)


#     # ==========================
#     # OPEN & CLOSE (real prices)
#     # ==========================
#     try:
#         open_price = float(intraday["Open"].iloc[0])
#     except:
#         open_price = None

#     try:
#         close_price = float(intraday["Close"].iloc[-1])
#     except:
#         close_price = None

#     # ==========================
#     # MIDAS BEAR
#     # ==========================
#     try:
#         bear_idx = intraday["MIDAS_Bear"].first_valid_index()
#         if bear_idx is not None:
#             midas_bear_time = intraday.loc[bear_idx, "Time"]
#             midas_bear_f = float(intraday.loc[bear_idx, "F_numeric"])    # F level
#             midas_bear_price = float(intraday.loc[bear_idx, "Close"])    # real price
#         else:
#             midas_bear_time = None
#             midas_bear_f = None
#             midas_bear_price = None
#     except:
#         midas_bear_time = midas_bear_f = midas_bear_price = None

#     # ==========================
#     # MIDAS BULL
#     # ==========================
#     try:
#         bull_idx = intraday["MIDAS_Bull"].first_valid_index()
#         if bull_idx is not None:
#             midas_bull_time = intraday.loc[bull_idx, "Time"]
#             midas_bull_f = float(intraday.loc[bull_idx, "F_numeric"])
#             midas_bull_price = float(intraday.loc[bull_idx, "Close"])
#         else:
#             midas_bull_time = None
#             midas_bull_f = None
#             midas_bull_price = None
#     except:
#         midas_bull_time = midas_bull_f = midas_bull_price = None


#     # ==========================
#     # INITIAL BALANCE (IB)
#     # ==========================
#     try:
#         # IB is always first 12 bars of intraday, matching your compute_initial_balance()
#         ib_slice = intraday.iloc[:12]

#         ib_high_f = float(ib_slice["F_numeric"].max())
#         ib_low_f  = float(ib_slice["F_numeric"].min())

#         # Locate time & real price for IB high
#         ib_high_row = intraday.loc[intraday["F_numeric"] == ib_high_f].iloc[0]
#         ib_high_time  = ib_high_row["Time"]
#         ib_high_price = float(ib_high_row["Close"])

#         # Locate time & real price for IB low
#         ib_low_row  = intraday.loc[intraday["F_numeric"] == ib_low_f].iloc[0]
#         ib_low_time   = ib_low_row["Time"]
#         ib_low_price  = float(ib_low_row["Close"])

#     except:
#         ib_high_f = ib_low_f = None
#         ib_high_time = ib_low_time = None
#         ib_high_price = ib_low_price = None





#     # ==========================
#     # FINAL JSON
#     # ==========================
#     # ==========================
#     # FINAL JSON
#     # ==========================


#     entries_block = extract_entries(intraday)
#     milestones_block = extract_milestones(intraday)

#     sector = detect_sector(ticker)
#     slug = f"{ticker.lower()}-{last_date}-{sector}"
#     expansion_block = detect_expansion_near_e1(intraday, perimeter=10)
#     range_ext_block = extract_range_extension(intraday, ib_high_f, ib_low_f, perimeter=4)

#     return round_all_numeric({
#         "name": str(ticker).lower(),
#         "date": str(last_date),
#         "sector": sector,

#         "slug": slug,

#         "totalVolume": total_vol_readable,

#         "open": open_price,
#         "close": close_price,
#         "expansionInsight": expansion_block,

#         "entries": entries_block,
#         "milestones": milestones_block,
#         "rangeExtension": range_ext_block,
#         "initialBalance": {
#             "high": {
#                 "time": ib_high_time,
#                 "fLevel": ib_high_f,
#                 "price": ib_high_price
#             },
#             "low": {
#                 "time": ib_low_time,
#                 "fLevel": ib_low_f,
#                 "price": ib_low_price
#             }
#         },

#         "midas": {
#             "bear": {
#                 "anchorTime": midas_bear_time,
#                 "price": midas_bear_price,
#                 "fLevel": midas_bear_f
#             },
#             "bull": {
#                 "anchorTime": midas_bull_time,
#                 "price": midas_bull_price,
#                 "fLevel": midas_bull_f
#             }
#         }
#     })




# def round_all_numeric(obj):
#     if isinstance(obj, bool):
#         return obj
#     if isinstance(obj, dict):
#         return {k: round_all_numeric(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [round_all_numeric(x) for x in obj]
#     else:
#         try:
#             return round(float(obj), 2)
#         except:
#             return obj

# def render_json_batch_download(json_map: dict):
#     """
#     json_map = {
#       "NVDA": {...},
#       "SPY":  {...},
#       ...
#     }
#     Renders one ZIP download button with all JSON files inside.
#     """
#     if not json_map:
#         return

#     buffer = io.BytesIO()
#     with zipfile.ZipFile(buffer, "w") as zf:
#         for tkr, payload in json_map.items():
#             # filename pattern: nvda-2025-11-21.json
#             safe_name = payload.get("name", str(tkr)).lower()
#             date_str = payload.get("date", "")
#             if date_str:
#                 fname = f"{safe_name}-{date_str}.json"
#             else:
#                 fname = f"{safe_name}.json"

#             zf.writestr(fname, json.dumps(payload, indent=4, ensure_ascii=False)
# )

#     buffer.seek(0)

#     st.download_button(
#         label="⬇️ Download JSON batch",
#         data=buffer,
#         file_name="mike_json_batch.zip",
#         mime="application/zip",
#     )



# pages/components/jsonExport.py

from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SECTOR_MAP: dict[str, list[str]] = {
    "ETFs":           ["spy", "qqq"],
    "finance":        ["wfc", "c", "jpm", "bac", "hood", "coin", "pypl"],
    "Semiconductors": ["nvda", "avgo", "amd", "mu", "mrvl", "qcom", "smci"],
    "Software":       ["msft", "pltr", "aapl", "googl", "meta", "uber", "tsla", "amzn"],
    "Futures":        ["nq", "es", "gc", "ym", "cl"],
}

_Z3_COLS   = ("Z3_Score", "z3", "Z3", "Z3_score")
_RV_COLS   = ("RVOL_5", "RVOL", "rvol")
_TIME_COLS = ("Time", "time", "Datetime", "datetime", "Timestamp", "timestamp", "DateTime")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def detect_sector(ticker: str) -> str:
    t = ticker.lower()
    for sector, tickers in SECTOR_MAP.items():
        if t in tickers:
            return sector
    return "Other"


def human_volume(n) -> str:
    try:
        n = float(n)
    except Exception:
        return str(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(int(n))


def round_all_numeric(obj):
    """Recursively round floats to 2dp. Leaves bools untouched."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {k: round_all_numeric(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_all_numeric(v) for v in obj]
    try:
        return round(float(obj), 2)
    except Exception:
        return obj


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series("", index=df.index)


def _fmt_time(val) -> str:
    try:
        return pd.to_datetime(val).strftime("%H:%M")
    except Exception:
        return str(val)


def _resolve_col(df: pd.DataFrame, candidates: tuple) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


# ─────────────────────────────────────────────────────────────
# PERIMETER WINDOW  (bishops + horses + z3)
# ─────────────────────────────────────────────────────────────

def _window_analysis(
    intraday: pd.DataFrame,
    center_pos: int,
    perimeter: int,
    f_series: pd.Series,
    rv_col: str | None,
    z3_col: str | None,
    has_tight: bool,
    has_std: bool,
    has_bbw_exp: bool,
    has_kijun: bool,
) -> dict:
    n     = len(intraday)
    start = max(0, center_pos - perimeter)
    end   = min(n - 1, center_pos + perimeter)

    pre_b  = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
    post_b = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
    pre_h:  list[float] = []
    post_h: list[float] = []

    for pos in range(start, end + 1):
        if pos == center_pos:
            continue
        tb = pre_b  if pos < center_pos else post_b
        th = pre_h  if pos < center_pos else post_h

        if has_tight:
            val = intraday["BBW_Tight_Emoji"].iat[pos]
            if isinstance(val, str) and val.strip() == "\U0001f41d":
                tb["yellow"] += 1

        if has_std:
            val = intraday["STD_Alert"].iat[pos]
            if isinstance(val, str) and val.strip() not in ("", "nan"):
                tb["purple"] += 1

        if has_bbw_exp:
            val = intraday["BBW Alert"].iat[pos]
            if isinstance(val, str) and val.strip() not in ("", "nan"):
                fv = f_series.iat[pos]
                kv = (
                    pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce")
                    if has_kijun else float("nan")
                )
                if pd.notna(fv) and pd.notna(kv):
                    if fv >= kv:
                        tb["green"] += 1
                    else:
                        tb["red"] += 1
                else:
                    tb["green"] += 1

        if rv_col is not None:
            rv = pd.to_numeric(intraday[rv_col].iat[pos], errors="coerce")
            if pd.notna(rv) and rv > 1.2:
                th.append(round(float(rv), 2))

    # Z3 window stats
    z3_on          = False
    z3_value       = None
    z3_last3       = None
    z3_max_time    = None
    z3_max_bars    = None
    time_col       = _resolve_col(intraday, _TIME_COLS)

    if z3_col is not None:
        lo = max(0, center_pos - perimeter)
        hi = min(n - 1, center_pos + perimeter)
        s  = pd.to_numeric(intraday[z3_col].iloc[lo: hi + 1], errors="coerce")

        if s.notna().any():
            peak_idx   = int(s.abs().idxmax())
            z3_value   = float(s.loc[peak_idx])
            z3_on      = abs(z3_value) >= 1.5
            z3_max_bars = peak_idx - center_pos
            if time_col is not None:
                z3_max_time = str(intraday[time_col].iat[peak_idx])

        s3 = pd.to_numeric(
            intraday[z3_col].iloc[center_pos: min(n - 1, center_pos + 2) + 1],
            errors="coerce",
        )
        if s3.notna().any():
            z3_last3 = float(s3.loc[int(s3.abs().idxmax())])

    return {
        "pre":  {
            "bishops": {k: v for k, v in pre_b.items()  if v > 0},
            "horses":  {"count": len(pre_h),  "rvolValues": pre_h},
        },
        "post": {
            "bishops": {k: v for k, v in post_b.items() if v > 0},
            "horses":  {"count": len(post_h), "rvolValues": post_h},
        },
        "z3On":               z3_on,
        "z3Value":            z3_value,
        "z3ValueLast3":       z3_last3,
        "z3MaxTime":          z3_max_time,
        "z3MaxBarsFromEntry": z3_max_bars,
    }


# ─────────────────────────────────────────────────────────────
# ENTRIES  (E1 / E2 / E3 / Reclaim / Blocked + exit)
# ─────────────────────────────────────────────────────────────

def extract_entries(intraday: pd.DataFrame, perimeter: int = 4) -> dict:
    if intraday is None or intraday.empty:
        return {"call": [], "put": []}

    call_entries: list[dict] = []
    put_entries:  list[dict] = []
    n = len(intraday)
    f = pd.to_numeric(intraday["F_numeric"], errors="coerce")

    rv_col      = _resolve_col(intraday, _RV_COLS)
    z3_col      = _resolve_col(intraday, _Z3_COLS)
    has_tight   = "BBW_Tight_Emoji" in intraday.columns
    has_std     = "STD_Alert"        in intraday.columns
    has_bbw_exp = "BBW Alert"        in intraday.columns
    has_kijun   = "Kijun_F"          in intraday.columns

    # Exit signal index sets
    put_signals = (
        set(intraday.index[_col(intraday, "Put_FirstEntry_Emoji").isin(["🎯", "⏳"])])
        | set(intraday.index[_col(intraday, "Put_DeferredEntry_Emoji") == "🧿"])
    )
    call_signals = (
        set(intraday.index[_col(intraday, "Call_FirstEntry_Emoji").isin(["🎯", "⏳"])])
        | set(intraday.index[_col(intraday, "Call_DeferredEntry_Emoji") == "🧿"])
    )

    # Last bar fallback (3:55)
    last_idx   = intraday.index[-1]
    last_price = float(intraday.at[last_idx, "Close"])
    last_f     = float(intraday.at[last_idx, "F_numeric"])
    last_time  = _fmt_time(intraday.at[last_idx, "Time"])

    def _exit_reason(exit_idx) -> str:
        checks = [
            ("Put_FirstEntry_Emoji",    "🎯", "Put E1 🎯"),
            ("Put_FirstEntry_Emoji",    "⏳", "Put E1 ⏳"),
            ("Put_DeferredEntry_Emoji", "🧿", "Put 🧿"),
            ("Call_FirstEntry_Emoji",   "🎯", "Call E1 🎯"),
            ("Call_FirstEntry_Emoji",   "⏳", "Call E1 ⏳"),
            ("Call_DeferredEntry_Emoji","🧿", "Call 🧿"),
        ]
        for col_name, emoji, label in checks:
            if col_name in intraday.columns and intraday.at[exit_idx, col_name] == emoji:
                return label
        return "signal"

    def _find_exit(entry_idx, opposite: set) -> dict:
        entry_pos  = intraday.index.get_loc(entry_idx)
        candidates = [ix for ix in opposite if intraday.index.get_loc(ix) > entry_pos]
        if candidates:
            exit_idx = min(candidates, key=lambda ix: intraday.index.get_loc(ix))
            return {
                "time":   _fmt_time(intraday.at[exit_idx, "Time"]),
                "price":  float(intraday.at[exit_idx, "Close"]),
                "fLevel": float(intraday.at[exit_idx, "F_numeric"]),
                "reason": _exit_reason(exit_idx),
            }
        return {"time": last_time, "price": last_price, "fLevel": last_f, "reason": "close"}

    def _attach_exit(ex: dict, entry_price: float, entry_f: float) -> dict:
        ex["priceMoveUSD"] = round(ex["price"] - entry_price, 2)
        ex["fMove"]        = round(ex["fLevel"] - entry_f, 2)
        return ex

    def _perimeter(pos: int) -> dict:
        return _window_analysis(
            intraday, pos, perimeter, f,
            rv_col, z3_col,
            has_tight, has_std, has_bbw_exp, has_kijun,
        )

    def _add(target: list, label: str, idx,
             opposite: set | None = None,
             with_perimeter: bool = False,
             extra: dict | None = None):
        pos         = intraday.index.get_loc(idx)
        entry_price = float(intraday.at[idx, "Close"])
        entry_f     = float(intraday.at[idx, "F_numeric"])
        row: dict   = {
            "type":   label,
            "time":   _fmt_time(intraday.at[idx, "Time"]),
            "price":  entry_price,
            "fLevel": entry_f,
        }
        if extra:
            row.update(extra)
        if with_perimeter:
            row["perimeter"] = _perimeter(pos)
        if opposite is not None:
            ex = _find_exit(idx, opposite)
            row["exit"] = _attach_exit(ex, entry_price, entry_f)
        target.append(row)

    # PUT
    for i in intraday.index[_col(intraday, "Put_FirstEntry_Emoji") == "🎯"]:
        _add(put_entries, "Put E1 🎯", i, opposite=call_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Put_FirstEntry_Emoji") == "⏳"]:
        _add(put_entries, "Put E1 ⏳ Blocked", i, opposite=call_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Put_DeferredEntry_Emoji") == "🧿"]:
        horse = _col(intraday, "Put_DeferredReinforce_Emoji").at[i] == "❗️"
        _add(put_entries, "Put Reclaim 🧿", i,
             opposite=call_signals, with_perimeter=True, extra={"horse": horse})

    for i in intraday.index[_col(intraday, "Put_SecondEntry_Emoji") == "🎯2"]:
        _add(put_entries, "Put E2 🎯2", i, opposite=call_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Put_ThirdEntry_Emoji") == "🎯3"]:
        _add(put_entries, "Put E3 🎯3", i)

    # CALL
    for i in intraday.index[_col(intraday, "Call_FirstEntry_Emoji") == "🎯"]:
        _add(call_entries, "Call E1 🎯", i, opposite=put_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Call_FirstEntry_Emoji") == "⏳"]:
        _add(call_entries, "Call E1 ⏳ Blocked", i, opposite=put_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Call_DeferredEntry_Emoji") == "🧿"]:
        horse = _col(intraday, "Call_DeferredReinforce_Emoji").at[i] == "❗️"
        _add(call_entries, "Call Reclaim 🧿", i,
             opposite=put_signals, with_perimeter=True, extra={"horse": horse})

    for i in intraday.index[_col(intraday, "Call_SecondEntry_Emoji") == "🎯2"]:
        _add(call_entries, "Call E2 🎯2", i, opposite=put_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Call_ThirdEntry_Emoji") == "🎯3"]:
        _add(call_entries, "Call E3 🎯3", i)

    return {"call": call_entries, "put": put_entries}


# ─────────────────────────────────────────────────────────────
# EXPANSION INSIGHT
# ─────────────────────────────────────────────────────────────

def detect_expansion_near_e1(intraday: pd.DataFrame, perimeter: int = 10) -> dict:
    out = {
        "bbw": {"present": False, "time": None, "count": 0},
        "std": {"present": False, "time": None, "count": 0},
    }
    if intraday is None or intraday.empty:
        return out

    call_e1 = intraday.index[_col(intraday, "Call_FirstEntry_Emoji") == "🎯"]
    put_e1  = intraday.index[_col(intraday, "Put_FirstEntry_Emoji")  == "🎯"]
    if len(call_e1) == 0 and len(put_e1) == 0:
        return out

    e1_pos = min(intraday.index.get_loc(ix) for ix in list(call_e1) + list(put_e1))
    n      = len(intraday)
    start  = max(0, e1_pos - perimeter)
    end    = min(n - 1, e1_pos + perimeter)

    def _count(series: pd.Series, emoji: str) -> tuple[int, int]:
        before = after = 0
        for pos in range(start, end + 1):
            if series.iloc[pos] == emoji:
                if pos <= e1_pos:
                    before += 1
                else:
                    after += 1
        return before, after

    def _timing(b: int, a: int) -> str | None:
        if b and a:  return "both"
        if b:        return "before"
        if a:        return "after"
        return None

    bbw_s = intraday.get("BBW Alert")
    std_s = intraday.get("STD_Alert")

    if bbw_s is not None:
        b, a = _count(bbw_s, "🔥")
        if b + a:
            out["bbw"] = {"present": True, "time": _timing(b, a), "count": b + a}

    if std_s is not None:
        b, a = _count(std_s, "🐦\u200d🔥")
        if b + a:
            out["std"] = {"present": True, "time": _timing(b, a), "count": b + a}

    return out


# ─────────────────────────────────────────────────────────────
# MILESTONES
# ─────────────────────────────────────────────────────────────

def extract_milestones(intraday: pd.DataFrame) -> dict:
    def _first(emoji_col: str, emoji: str) -> dict:
        rows = intraday[_col(intraday, emoji_col) == emoji]
        if rows.empty:
            return {}
        r = rows.iloc[0]
        return {"Time": _fmt_time(r["Time"]), "Price": float(r["Close"]), "F%": float(r["F_numeric"])}

    goldmine = [
        {"Time": _fmt_time(r["Time"]), "Price": float(r["Close"]), "F%": float(r["F_numeric"])}
        for _, r in intraday[_col(intraday, "Goldmine_E1_Emoji") == "💰"].iterrows()
    ]
    return {
        "T0":       _first("T0_Emoji", "🚪"),
        "T1":       _first("T1_Emoji", "🏇🏼"),
        "T2":       _first("T2_Emoji", "⚡"),
        "goldmine": goldmine,
    }


# ─────────────────────────────────────────────────────────────
# MARKET PROFILE  (kept for compat; nose/ear excluded from JSON)
# ─────────────────────────────────────────────────────────────

def extract_market_profile(mp_df: pd.DataFrame | None) -> dict:
    if mp_df is None or mp_df.empty or "F% Level" not in mp_df.columns:
        return {}
    df = mp_df.copy()
    for c, default in [("TPO_Count", 0), ("%Vol", 0.0), ("🦻🏼", ""), ("👃🏽", "")]:
        if c not in df.columns:
            df[c] = default

    out: dict = {}

    nose_row = df[df["👃🏽"] == "👃🏽"]
    if nose_row.empty:
        nose_row = df.sort_values("TPO_Count", ascending=False).head(1)
    if not nose_row.empty:
        out["nose"] = {"fLevel": int(nose_row["F% Level"].iloc[0]),
                       "tpoCount": int(nose_row["TPO_Count"].iloc[0])}

    ear_row = df[df["🦻🏼"] == "🦻🏼"]
    if ear_row.empty:
        ear_row = df.sort_values("%Vol", ascending=False).head(1)
    if not ear_row.empty:
        out["ear"] = {"fLevel": int(ear_row["F% Level"].iloc[0]),
                      "percentVol": float(ear_row["%Vol"].iloc[0])}

    return out


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPAT STUBS  (imported by other modules)
# ─────────────────────────────────────────────────────────────

def extract_vector_capacitance(intraday: pd.DataFrame, perimeter: int = 5) -> dict:
    return {"call": {}, "put": {}}


def extract_profile_cross_insight(
    intraday: pd.DataFrame,
    mp_block: dict | None,
    goldmine_dist: float = 64.0,
) -> dict:
    return {}


# ─────────────────────────────────────────────────────────────
# RANGE EXTENSION
# ─────────────────────────────────────────────────────────────

def extract_range_extension(
    intraday: pd.DataFrame,
    ib_high_f: float | None,
    ib_low_f:  float | None,
    perimeter: int = 4,
) -> dict:
    if intraday is None or intraday.empty or ib_high_f is None or ib_low_f is None:
        return {}

    f      = pd.to_numeric(intraday["F_numeric"], errors="coerce")
    n      = len(intraday)
    IB_END = 12

    rv_col      = _resolve_col(intraday, _RV_COLS)
    z3_col      = _resolve_col(intraday, _Z3_COLS)
    has_tight   = "BBW_Tight_Emoji" in intraday.columns
    has_std     = "STD_Alert"        in intraday.columns
    has_bbw_exp = "BBW Alert"        in intraday.columns
    has_kijun   = "Kijun_F"          in intraday.columns

    def _win(start: int, end: int) -> dict:
        bishops: dict[str, int] = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
        horses:  list[float]    = []

        for pos in range(max(0, start), min(n - 1, end) + 1):
            if has_tight:
                val = intraday["BBW_Tight_Emoji"].iat[pos]
                if isinstance(val, str) and val.strip() == "\U0001f41d":
                    bishops["yellow"] += 1

            if has_std:
                val = intraday["STD_Alert"].iat[pos]
                if isinstance(val, str) and val.strip() not in ("", "nan"):
                    bishops["purple"] += 1

            if has_bbw_exp:
                val = intraday["BBW Alert"].iat[pos]
                if isinstance(val, str) and val.strip() not in ("", "nan"):
                    fv = f.iat[pos]
                    kv = (
                        pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce")
                        if has_kijun else float("nan")
                    )
                    if pd.notna(fv) and pd.notna(kv):
                        if fv >= kv:
                            bishops["green"] += 1
                        else:
                            bishops["red"] += 1
                    else:
                        bishops["green"] += 1

            if rv_col is not None:
                rv = pd.to_numeric(intraday[rv_col].iat[pos], errors="coerce")
                if pd.notna(rv) and rv > 1.2:
                    horses.append(round(float(rv), 2))

        return {
            "bishops": {k: v for k, v in bishops.items() if v > 0},
            "horses":  {"count": len(horses), "rvolValues": horses},
        }

    def _scan(direction: str) -> dict:
        ext_loc = None
        for i in range(IB_END, n):
            fv = f.iat[i]
            if pd.notna(fv):
                if direction == "high" and fv > ib_high_f:
                    ext_loc = i
                    break
                elif direction == "low" and fv < ib_low_f:
                    ext_loc = i
                    break
        if ext_loc is None:
            return {}

        pre_start = max(IB_END, ext_loc - perimeter)
        post_end  = min(n - 1,  ext_loc + perimeter)

        z3_val: float | None = None
        if z3_col is not None:
            raw = pd.to_numeric(intraday[z3_col].iat[ext_loc], errors="coerce")
            if pd.notna(raw):
                z3_val = float(raw)
        z3_on = z3_val is not None and abs(z3_val) >= 1.5

        time_str = None
        if "Time" in intraday.columns:
            time_str = _fmt_time(intraday["Time"].iat[ext_loc])

        return {
            "time":    time_str,
            "fLevel":  round(float(f.iat[ext_loc]), 2),
            "z3On":    bool(z3_on),
            "z3Value": round(z3_val, 2) if z3_val is not None else None,
            "pre":     _win(pre_start, ext_loc - 1),
            "post":    _win(ext_loc + 1, post_end),
        }

    result: dict = {}
    h = _scan("high")
    l = _scan("low")
    if h:
        result["aboveIBHigh"] = h
    if l:
        result["belowIBLow"] = l
    return result


# ─────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────

def build_basic_json(
    intraday: pd.DataFrame,
    ticker: str,
    mp_df: pd.DataFrame | None = None,
) -> dict:
    if intraday is None or intraday.empty:
        return {}

    total_vol     = int(intraday["Volume"].sum()) if "Volume" in intraday.columns else 0
    last_date     = intraday["Date"].iloc[-1]     if "Date"   in intraday.columns else date.today()
    sector        = detect_sector(ticker)
    slug          = f"{ticker.lower()}-{last_date}-{sector}"
    open_price    = float(intraday["Open"].iloc[0])   if "Open"  in intraday.columns else None
    close_price   = float(intraday["Close"].iloc[-1]) if "Close" in intraday.columns else None

    # MIDAS Bear
    try:
        bear_idx         = intraday["MIDAS_Bear"].first_valid_index()
        midas_bear_time  = intraday.loc[bear_idx, "Time"]            if bear_idx else None
        midas_bear_f     = float(intraday.loc[bear_idx, "F_numeric"]) if bear_idx else None
        midas_bear_price = float(intraday.loc[bear_idx, "Close"])     if bear_idx else None
    except Exception:
        midas_bear_time = midas_bear_f = midas_bear_price = None

    # MIDAS Bull
    try:
        bull_idx         = intraday["MIDAS_Bull"].first_valid_index()
        midas_bull_time  = intraday.loc[bull_idx, "Time"]            if bull_idx else None
        midas_bull_f     = float(intraday.loc[bull_idx, "F_numeric"]) if bull_idx else None
        midas_bull_price = float(intraday.loc[bull_idx, "Close"])     if bull_idx else None
    except Exception:
        midas_bull_time = midas_bull_f = midas_bull_price = None

    # Initial Balance (first 12 bars)
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
    except Exception:
        ib_high_f = ib_low_f = None
        ib_high_time = ib_low_time = None
        ib_high_price = ib_low_price = None

    mp_block = extract_market_profile(mp_df)

    payload = {
        "name":             str(ticker).lower(),
        "date":             str(last_date),
        "sector":           sector,
        "slug":             slug,
        "totalVolume":      human_volume(total_vol),
        "open":             open_price,
        "close":            close_price,
        "expansionInsight": detect_expansion_near_e1(intraday, perimeter=10),
        "entries":          extract_entries(intraday, perimeter=4),
        "milestones":       extract_milestones(intraday),
        "marketProfile":    {k: v for k, v in mp_block.items() if k not in ("nose", "ear")},
        "rangeExtension":   extract_range_extension(intraday, ib_high_f, ib_low_f, perimeter=4),
        "initialBalance": {
            "high": {"time": ib_high_time, "fLevel": ib_high_f, "price": ib_high_price},
            "low":  {"time": ib_low_time,  "fLevel": ib_low_f,  "price": ib_low_price},
        },
        "midas": {
            "bear": {"anchorTime": midas_bear_time,  "price": midas_bear_price,  "fLevel": midas_bear_f},
            "bull": {"anchorTime": midas_bull_time,  "price": midas_bull_price,  "fLevel": midas_bull_f},
        },
    }

    return round_all_numeric(payload)


# ─────────────────────────────────────────────────────────────
# DOWNLOAD WIDGET
# ─────────────────────────────────────────────────────────────

def render_json_batch_download(json_map: dict) -> None:
    if not json_map:
        return

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for tkr, payload in json_map.items():
            safe  = payload.get("name", str(tkr)).lower()
            d     = payload.get("date", "")
            fname = f"{safe}-{d}.json" if d else f"{safe}.json"
            zf.writestr(fname, json.dumps(payload, indent=4, ensure_ascii=False))

    buffer.seek(0)
    st.download_button(
        label="⬇️ Download JSON batch",
        data=buffer,
        file_name="mike_json_batch.zip",
        mime="application/zip",
    )
