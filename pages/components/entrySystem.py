# # pages/components/entrySystem.py

# import numpy as np
# import pandas as pd


# def apply_entry_system(
#     intraday: pd.DataFrame,
#     ib_info: dict | None = None,
#     use_physics: bool = True,
# ) -> pd.DataFrame:
#     """
#     Attach Mike's Entry 1 / Entry 2 / Entry 3 markers:

#       🎯   = FirstEntry (after MIDAS anchor + Heaven/Drizzle)
#       🎯2  = SecondEntry (Kijun cross continuation after 🎯)
#       🎯3  = ThirdEntry (IB break after 🎯 + 🎯2)

#     Optional physics filters on 🎯2 / 🎯3 use:
#       - Vector_pct (direction)
#       - Volatility_Composite (for E3 regime check)
#     """

#     if intraday is None or intraday.empty:
#         return intraday

#     df = intraday.copy()

#     # ---------------------------------
#     # 0) Ensure core numeric columns
#     # ---------------------------------
#     df["F_numeric"] = pd.to_numeric(df.get("F_numeric"), errors="coerce")
#     if "Kijun_F" in df.columns:
#         df["Kijun_F"] = pd.to_numeric(df["Kijun_F"], errors="coerce")

#     # ---------------------------------
#     # 1) Attach IB_High / IB_Low columns if we got ib_info
#     # ---------------------------------
#     ib_high = ib_low = None
#     if ib_info is not None:
#         # support both styles: IB_High / ib_high
#         ib_high = ib_info.get("IB_High", ib_info.get("ib_high"))
#         ib_low = ib_info.get("IB_Low", ib_info.get("ib_low"))

#     if "IB_High" not in df.columns:
#         df["IB_High"] = np.nan
#     if "IB_Low" not in df.columns:
#         df["IB_Low"] = np.nan

#     if ib_high is not None:
#         df["IB_High"] = float(ib_high)
#     if ib_low is not None:
#         df["IB_Low"] = float(ib_low)

#     # ---------------------------------
#     # 2) Prepare entry emoji columns
#     # ---------------------------------
#     for col in [
#         "Put_FirstEntry_Emoji",
#         "Call_FirstEntry_Emoji",
#         "Put_SecondEntry_Emoji",
#         "Call_SecondEntry_Emoji",
#         "Put_ThirdEntry_Emoji",
#         "Call_ThirdEntry_Emoji",
#     ]:
#         if col not in df.columns:
#             df[col] = ""

#     # ---------------------------------
#     # 3) Small helpers for physics filters
#     # ---------------------------------
#     def last_vector_sign(idx: int) -> int | None:
#         """
#         Look backwards up to idx and find the last non-null Vector_pct.
#         Returns:
#             +1 → last vector was upward
#             -1 → last vector was downward
#              0 → flat
#             None → no vector info / physics disabled
#         """
#         if not use_physics:
#             return None
#         if "Vector_pct" not in df.columns:
#             return None

#         # idx is positional here (iloc index)
#         series = df["Vector_pct"]
#         if series.isna().all():
#             return None

#         # up to this bar
#         sub = series.iloc[: idx + 1].dropna()
#         if sub.empty:
#             return None

#         val = sub.iloc[-1]
#         if val > 0:
#             return 1
#         if val < 0:
#             return -1
#         return 0

#     def vol_composite_ok(idx: int) -> bool:
#         """
#         For Entry 3: require Volatility_Composite to be at least
#         around its rolling median, so we only trade IB breaks in
#         an active regime, not in dead tape.
#         """
#         if not use_physics:
#             return True

#         col = "Volatility_Composite"
#         if col not in df.columns:
#             return True

#         series = pd.to_numeric(df[col], errors="coerce")
#         if series.isna().all():
#             return True

#         window = 20
#         med = series.rolling(window, min_periods=max(5, window // 2)).median()
#         if np.isnan(med.iloc[idx]):
#             return True

#         return bool(series.iloc[idx] >= med.iloc[idx])

#     # ---------------------------------
#     # 4) ENTRY 1 – MIDAS anchor + Heaven/Drizzle
#     # ---------------------------------
#     # Puts: MIDAS_Bear → first Drizzle_Emoji 🌧️
#     if "MIDAS_Bear" in df.columns and "Drizzle_Emoji" in df.columns:
#         anchor_idx = df["MIDAS_Bear"].first_valid_index()
#         if anchor_idx is not None:
#             start_pos = df.index.get_loc(anchor_idx)
#             for i in range(start_pos, len(df)):
#                 if df.iloc[i]["Drizzle_Emoji"] == "🌧️":
#                     df.at[df.index[i], "Put_FirstEntry_Emoji"] = "🎯"
#                     break

#     # Calls: MIDAS_Bull → first Heaven_Cloud ☁️
#     if "MIDAS_Bull" in df.columns and "Heaven_Cloud" in df.columns:
#         anchor_idx = df["MIDAS_Bull"].first_valid_index()
#         if anchor_idx is not None:
#             start_pos = df.index.get_loc(anchor_idx)
#             for i in range(start_pos, len(df)):
#                 if df.iloc[i]["Heaven_Cloud"] == "☁️":
#                     df.at[df.index[i], "Call_FirstEntry_Emoji"] = "🎯"
#                     break

#     # ---------------------------------
#     # 5) ENTRY 2 – SIMPLE KIJUN CROSS AFTER ENTRY 1 (NO PHYSICS)
#     # ---------------------------------
#     f = df["F_numeric"]
#     kijun = df.get("Kijun_F")

#     # --- PUT 🎯2 ---
#     first_put = df.index[df["Put_FirstEntry_Emoji"] == "🎯"]
#     if len(first_put) > 0 and kijun is not None:
#         start_loc = df.index.get_loc(first_put[0])

#         for i in range(start_loc + 1, len(df)):
#             prev_f = f.iloc[i - 1]
#             curr_f = f.iloc[i]
#             prev_k = kijun.iloc[i - 1]
#             curr_k = kijun.iloc[i]

#             if (
#                 pd.notna(prev_f)
#                 and pd.notna(curr_f)
#                 and pd.notna(prev_k)
#                 and pd.notna(curr_k)
#             ):
#                 # First cross *down* Kijun_F after E1
#                 if prev_f > prev_k and curr_f <= curr_k:
#                     df.at[df.index[i], "Put_SecondEntry_Emoji"] = "🎯2"
#                     break

#     # --- CALL 🎯2 ---
#     first_call = df.index[df["Call_FirstEntry_Emoji"] == "🎯"]
#     if len(first_call) > 0 and kijun is not None:
#         start_loc = df.index.get_loc(first_call[0])

#         for i in range(start_loc + 1, len(df)):
#             prev_f = f.iloc[i - 1]
#             curr_f = f.iloc[i]
#             prev_k = kijun.iloc[i - 1]
#             curr_k = kijun.iloc[i]

#             if (
#                 pd.notna(prev_f)
#                 and pd.notna(curr_f)
#                 and pd.notna(prev_k)
#                 and pd.notna(curr_k)
#             ):
#                 # First cross *up* Kijun_F after E1
#                 if prev_f < prev_k and curr_f >= curr_k:
#                     df.at[df.index[i], "Call_SecondEntry_Emoji"] = "🎯2"
#                     break

#     # ---------------------------------
#     # 6) ENTRY 3 – IB break after 🎯 + 🎯2
#     #     with physics direction + Volatility_Composite regime
#     # ---------------------------------
#     ib_low_series = pd.to_numeric(df["IB_Low"], errors="coerce")
#     ib_high_series = pd.to_numeric(df["IB_High"], errors="coerce")

#     # --- PUT 🎯3 (IB_Low break) ---
#     first_put = df.index[df["Put_FirstEntry_Emoji"] == "🎯"]
#     second_put = df.index[df["Put_SecondEntry_Emoji"] == "🎯2"]

#     if len(first_put) > 0 and len(second_put) > 0:
#         i_first = df.index.get_loc(first_put[0])
#         i_second = df.index.get_loc(second_put[0])

#         # Check if IB_Low already crossed between E1 and E2
#         ib_low_crossed_by_second = False
#         for i in range(i_first, i_second + 1):
#             val_f = f.iloc[i]
#             val_ib = ib_low_series.iloc[i]
#             if pd.notna(val_f) and pd.notna(val_ib) and val_f < val_ib:
#                 ib_low_crossed_by_second = True
#                 break

#         if not ib_low_crossed_by_second:
#             for i in range(i_second + 1, len(df) - 1):
#                 f_prev = f.iloc[i - 1]
#                 f_curr = f.iloc[i]
#                 ib_prev = ib_low_series.iloc[i - 1]
#                 ib_curr = ib_low_series.iloc[i]

#                 if (
#                     pd.notna(f_prev)
#                     and pd.notna(f_curr)
#                     and pd.notna(ib_prev)
#                     and pd.notna(ib_curr)
#                 ):
#                     # First cross below IB_Low
#                     if f_prev > ib_prev and f_curr <= ib_curr:
#                         j = i + 1
#                         f_next = f.iloc[j]
#                         if pd.notna(f_next) and f_next < f_curr:
#                             ok = True
#                             if use_physics:
#                                 sign = last_vector_sign(j)
#                                 if sign is not None and sign >= 0:
#                                     ok = False
#                                 if ok and not vol_composite_ok(j):
#                                     ok = False
#                             if ok:
#                                 df.at[df.index[j], "Put_ThirdEntry_Emoji"] = "🎯3"
#                                 break

#     # --- CALL 🎯3 (IB_High break) ---
#     first_call = df.index[df["Call_FirstEntry_Emoji"] == "🎯"]
#     second_call = df.index[df["Call_SecondEntry_Emoji"] == "🎯2"]

#     if len(first_call) > 0 and len(second_call) > 0:
#         i_first = df.index.get_loc(first_call[0])
#         i_second = df.index.get_loc(second_call[0])

#         # Check if IB_High already crossed between E1 and E2
#         crossed_by_second = False
#         for i in range(i_first, i_second + 1):
#             val_f = f.iloc[i]
#             val_ib = ib_high_series.iloc[i]
#             if pd.notna(val_f) and pd.notna(val_ib) and val_f > val_ib:
#                 crossed_by_second = True
#                 break

#         if not crossed_by_second:
#             for i in range(i_second + 1, len(df) - 1):
#                 f_prev = f.iloc[i - 1]
#                 f_curr = f.iloc[i]
#                 ib_prev = ib_high_series.iloc[i - 1]
#                 ib_curr = ib_high_series.iloc[i]

#                 if (
#                     pd.notna(f_prev)
#                     and pd.notna(f_curr)
#                     and pd.notna(ib_prev)
#                     and pd.notna(ib_curr)
#                 ):
#                     # First cross above IB_High
#                     if f_prev < ib_prev and f_curr >= ib_curr:
#                         j = i + 1
#                         f_next = f.iloc[j]
#                         if pd.notna(f_next) and f_next > f_curr:
#                             ok = True
#                             if use_physics:
#                                 sign = last_vector_sign(j)
#                                 if sign is not None and sign <= 0:
#                                     ok = False
#                                 if ok and not vol_composite_ok(j):
#                                     ok = False
#                             if ok:
#                                 df.at[df.index[j], "Call_ThirdEntry_Emoji"] = "🎯3"
#                                 break

#     return df



# pages/components/entrySystem.py

import numpy as np
import pandas as pd


EMOJI_EXEC    = "🎯"
EMOJI_HOLD    = "⏳"
EMOJI_RECLAIM = "🧿"
EMOJI_HORSE   = "❗️"


def apply_entry_system(
    intraday: pd.DataFrame,
    ib_info: dict | None = None,
    use_physics: bool = True,
    z3_k: float = 1.2,      # Z3 threshold for reclaim
    rvol_k: float = 1.3,    # RVOL threshold for horse
    horse_window: int = 3,  # bars around reclaim to check RVOL
) -> pd.DataFrame:
    """
    Attach Mike's Entry markers — mirrors JS entrySystem.js exactly:

      E1:
        🎯  = Executable Entry 1 (good IB location)
        ⏳  = Blocked Entry 1    (IB cellar / loft — bad location)

      Deferred (after ⏳):
        🧿  = Reclaim  (Z3 ignition + cross or continuation through blocked level)
        ❗️  = Horse    (RVOL spike within horse_window bars of reclaim)

      E2 / E3:
        🎯2 = Second Entry (Kijun cross after executable side is live)
        🎯3 = Third Entry  (IB break confirmed after E1 + E2)
    """

    if intraday is None or intraday.empty:
        return intraday

    df = intraday.copy()
    n  = len(df)

    # ---------------------------------
    # 0) Ensure core numeric columns
    # ---------------------------------
    df["F_numeric"] = pd.to_numeric(df.get("F_numeric"), errors="coerce")
    if "Kijun_F" in df.columns:
        df["Kijun_F"] = pd.to_numeric(df["Kijun_F"], errors="coerce")

    # ---------------------------------
    # 1) Attach IB_High / IB_Low
    # ---------------------------------
    ib_high = ib_low = None
    if ib_info is not None:
        ib_high = ib_info.get("IB_High", ib_info.get("ib_high"))
        ib_low  = ib_info.get("IB_Low",  ib_info.get("ib_low"))

    if "IB_High" not in df.columns:
        df["IB_High"] = np.nan
    if "IB_Low" not in df.columns:
        df["IB_Low"] = np.nan

    if ib_high is not None:
        df["IB_High"] = float(ib_high)
    if ib_low is not None:
        df["IB_Low"]  = float(ib_low)

    has_ib    = (ib_high is not None) and (ib_low is not None)
    ib_range  = (float(ib_high) - float(ib_low)) if has_ib else None
    ib_cellar = (float(ib_low)  + ib_range / 3)       if has_ib else None
    ib_loft   = (float(ib_low)  + 2 * ib_range / 3)   if has_ib else None

    # ---------------------------------
    # 2) Prepare all emoji + level columns
    # ---------------------------------
    for col in [
        "Put_FirstEntry_Emoji",    "Call_FirstEntry_Emoji",
        "Put_SecondEntry_Emoji",   "Call_SecondEntry_Emoji",
        "Put_ThirdEntry_Emoji",    "Call_ThirdEntry_Emoji",
        "Put_DeferredEntry_Emoji", "Call_DeferredEntry_Emoji",
        "Put_DeferredReinforce_Emoji", "Call_DeferredReinforce_Emoji",
    ]:
        if col not in df.columns:
            df[col] = ""

    for col in ("Put_E1_LevelF", "Call_E1_LevelF"):
        if col not in df.columns:
            df[col] = np.nan

    # ---------------------------------
    # 3) Pre-compute arrays
    # ---------------------------------
    f_arr    = pd.to_numeric(df["F_numeric"], errors="coerce").values
    kij_arr  = (pd.to_numeric(df["Kijun_F"], errors="coerce").values
                if "Kijun_F" in df.columns else np.full(n, np.nan))
    ib_h_arr = pd.to_numeric(df["IB_High"], errors="coerce").values
    ib_l_arr = pd.to_numeric(df["IB_Low"],  errors="coerce").values
    has_kijun = not np.all(np.isnan(kij_arr))

    # Z3 — try multiple column names (mirrors JS pickZ3)
    z3_arr = np.full(n, np.nan)
    for z3_col in ("Z3_Score", "z3", "Z3", "Z3_score", "Z3_numeric"):
        if z3_col in df.columns:
            z3_arr = pd.to_numeric(df[z3_col], errors="coerce").values
            break
    has_z3 = not np.all(np.isnan(z3_arr))

    # RVOL — try multiple column names (mirrors JS pickRVOL)
    rvol_arr = np.full(n, np.nan)
    for rv_col in ("RVOL_5", "RVOL", "rvol", "rvol5", "RVOL_numeric"):
        if rv_col in df.columns:
            rvol_arr = pd.to_numeric(df[rv_col], errors="coerce").values
            break

    def z3_on_call(z): return np.isfinite(z) and z >=  abs(z3_k)
    def z3_on_put(z):  return np.isfinite(z) and z <= -abs(z3_k)

    def horse_in_window(exec_idx: int) -> bool:
        lo = max(0, exec_idx - horse_window)
        hi = min(n - 1, exec_idx + horse_window)
        for i in range(lo, hi + 1):
            rv = rvol_arr[i]
            if np.isfinite(rv) and rv > rvol_k:
                return True
        return False

    # ---------------------------------
    # 4) Physics helpers (for E3)
    # ---------------------------------
    def last_vector_sign(idx: int) -> int | None:
        if not use_physics or "Vector_pct" not in df.columns:
            return None
        series = df["Vector_pct"]
        if series.isna().all():
            return None
        sub = series.iloc[: idx + 1].dropna()
        if sub.empty:
            return None
        val = sub.iloc[-1]
        return 1 if val > 0 else (-1 if val < 0 else 0)

    def vol_composite_ok(idx: int) -> bool:
        if not use_physics:
            return True
        col = "Volatility_Composite"
        if col not in df.columns:
            return True
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            return True
        window = 20
        med = series.rolling(window, min_periods=max(5, window // 2)).median()
        if np.isnan(med.iloc[idx]):
            return True
        return bool(series.iloc[idx] >= med.iloc[idx])

    # ---------------------------------
    # 5) ENTRY 1 — location-aware (🎯 or ⏳)
    # ---------------------------------

    # PUT E1: MIDAS_Bear → first Drizzle 🌧️
    if "MIDAS_Bear" in df.columns and "Drizzle_Emoji" in df.columns:
        anchor_idx = df["MIDAS_Bear"].first_valid_index()
        if anchor_idx is not None:
            start_pos = df.index.get_loc(anchor_idx)
            for i in range(start_pos, n):
                if df.iloc[i]["Drizzle_Emoji"] == "🌧️":
                    fv    = f_arr[i]
                    emoji = EMOJI_EXEC
                    if has_ib and np.isfinite(fv):
                        if (ib_cellar is not None and fv < ib_cellar) or fv < float(ib_low):
                            emoji = EMOJI_HOLD
                    df.at[df.index[i], "Put_FirstEntry_Emoji"] = emoji
                    if emoji == EMOJI_HOLD and np.isfinite(fv):
                        df.at[df.index[i], "Put_E1_LevelF"] = fv
                    break

    # CALL E1: MIDAS_Bull → first Heaven ☁️
    if "MIDAS_Bull" in df.columns and "Heaven_Cloud" in df.columns:
        anchor_idx = df["MIDAS_Bull"].first_valid_index()
        if anchor_idx is not None:
            start_pos = df.index.get_loc(anchor_idx)
            for i in range(start_pos, n):
                if df.iloc[i]["Heaven_Cloud"] == "☁️":
                    fv    = f_arr[i]
                    emoji = EMOJI_EXEC
                    if has_ib and np.isfinite(fv):
                        if (ib_loft is not None and fv > ib_loft) or fv > float(ib_high):
                            emoji = EMOJI_HOLD
                    df.at[df.index[i], "Call_FirstEntry_Emoji"] = emoji
                    if emoji == EMOJI_HOLD and np.isfinite(fv):
                        df.at[df.index[i], "Call_E1_LevelF"] = fv
                    break

    # ---------------------------------
    # 6) DEFERRED RECLAIM after ⏳
    # ---------------------------------
    def apply_deferred(side: str) -> bool:
        is_put   = side == "put"
        e1_col   = "Put_FirstEntry_Emoji"        if is_put else "Call_FirstEntry_Emoji"
        lvl_col  = "Put_E1_LevelF"               if is_put else "Call_E1_LevelF"
        def_col  = "Put_DeferredEntry_Emoji"      if is_put else "Call_DeferredEntry_Emoji"
        rein_col = "Put_DeferredReinforce_Emoji"  if is_put else "Call_DeferredReinforce_Emoji"

        blocked_mask = df[e1_col] == EMOJI_HOLD
        if not blocked_mask.any():
            return False

        # Z3 required — no Z3 data = no reclaim (matches JS)
        if not has_z3:
            return False

        blocked_iloc = df.index.get_loc(df.index[blocked_mask][0])
        level = df[lvl_col].iloc[blocked_iloc]
        if not np.isfinite(level):
            level = f_arr[blocked_iloc]
        if not np.isfinite(level):
            return False

        seen_opposite = False

        for i in range(blocked_iloc + 1, n):
            fv    = f_arr[i]
            fprev = f_arr[i - 1]
            z3    = z3_arr[i]

            if not np.isfinite(fv):
                continue

            # Track retrace past the blocked level
            if is_put:
                if fv > level: seen_opposite = True
            else:
                if fv < level: seen_opposite = True

            cross_down = np.isfinite(fprev) and fprev > level and fv <= level
            cross_up   = np.isfinite(fprev) and fprev < level and fv >= level
            z3_ok      = z3_on_put(z3) if is_put else z3_on_call(z3)

            # PATH A: retrace → re-break with Z3 ON
            if seen_opposite:
                if (is_put and cross_down and z3_ok) or (not is_put and cross_up and z3_ok):
                    df.at[df.index[i], def_col]  = EMOJI_RECLAIM
                    if horse_in_window(i):
                        df.at[df.index[i], rein_col] = EMOJI_HORSE
                    return True

            # PATH B: no retrace, Z3 fires while already beyond level
            if not seen_opposite:
                beyond = (fv < level) if is_put else (fv > level)
                if beyond and z3_ok:
                    df.at[df.index[i], def_col]  = EMOJI_RECLAIM
                    if horse_in_window(i):
                        df.at[df.index[i], rein_col] = EMOJI_HORSE
                    return True

        return False

    apply_deferred("put")
    apply_deferred("call")

    # ---------------------------------
    # 7) Active executable index per side
    #    exec 🎯, hold ⏳, or reclaim 🧿 all anchor E2
    # ---------------------------------
    def last_exec_iloc(side: str) -> int:
        is_put  = side == "put"
        e1_col  = "Put_FirstEntry_Emoji"    if is_put else "Call_FirstEntry_Emoji"
        def_col = "Put_DeferredEntry_Emoji" if is_put else "Call_DeferredEntry_Emoji"
        for i in range(n - 1, -1, -1):
            v1 = df[e1_col].iloc[i]
            v2 = df[def_col].iloc[i]
            if v1 in (EMOJI_EXEC, EMOJI_HOLD) or v2 == EMOJI_RECLAIM:
                return i
        return -1

    put_exec_iloc  = last_exec_iloc("put")
    call_exec_iloc = last_exec_iloc("call")

    # ---------------------------------
    # 8) ENTRY 2 — Kijun cross (both sides independent, loosened cross)
    # ---------------------------------
    if has_kijun:
        # CALL E2: cross UP
        if call_exec_iloc >= 0:
            for i in range(call_exec_iloc + 1, n):
                pf, cf = f_arr[i - 1], f_arr[i]
                pk, ck = kij_arr[i - 1], kij_arr[i]
                if not all(np.isfinite(v) for v in [pf, cf, pk, ck]):
                    continue
                if pf <= pk and cf > ck:
                    df.at[df.index[i], "Call_SecondEntry_Emoji"] = "🎯2"
                    break

        # PUT E2: cross DOWN
        if put_exec_iloc >= 0:
            for i in range(put_exec_iloc + 1, n):
                pf, cf = f_arr[i - 1], f_arr[i]
                pk, ck = kij_arr[i - 1], kij_arr[i]
                if not all(np.isfinite(v) for v in [pf, cf, pk, ck]):
                    continue
                if pf >= pk and cf < ck:
                    df.at[df.index[i], "Put_SecondEntry_Emoji"] = "🎯2"
                    break

    # ---------------------------------
    # 9) ENTRY 3 — IB break after E1 + E2
    # ---------------------------------
    ib_low_series  = pd.to_numeric(df["IB_Low"],  errors="coerce")
    ib_high_series = pd.to_numeric(df["IB_High"], errors="coerce")

    # PUT 🎯3
    put_e1_rows = df.index[
        (df["Put_FirstEntry_Emoji"] == EMOJI_EXEC) |
        (df["Put_DeferredEntry_Emoji"] == EMOJI_RECLAIM)
    ]
    put_e2_rows = df.index[df["Put_SecondEntry_Emoji"] == "🎯2"]

    if len(put_e1_rows) > 0 and len(put_e2_rows) > 0:
        i_e1 = df.index.get_loc(put_e1_rows[0])
        i_e2 = df.index.get_loc(put_e2_rows[0])
        if i_e2 > i_e1:
            crossed = any(
                np.isfinite(f_arr[i]) and np.isfinite(ib_l_arr[i]) and f_arr[i] < ib_l_arr[i]
                for i in range(i_e1, i_e2 + 1)
            )
            if not crossed:
                for i in range(i_e2 + 1, n - 1):
                    f_prev, f_curr = f_arr[i - 1], f_arr[i]
                    ib_prev, ib_curr = ib_l_arr[i - 1], ib_l_arr[i]
                    if not all(np.isfinite(v) for v in [f_prev, f_curr, ib_prev, ib_curr]):
                        continue
                    if f_prev > ib_prev and f_curr <= ib_curr:
                        j = i + 1
                        if np.isfinite(f_arr[j]) and f_arr[j] < f_curr:
                            ok = True
                            if use_physics:
                                sign = last_vector_sign(j)
                                if sign is not None and sign >= 0:
                                    ok = False
                                if ok and not vol_composite_ok(j):
                                    ok = False
                            if ok:
                                df.at[df.index[j], "Put_ThirdEntry_Emoji"] = "🎯3"
                                break

    # CALL 🎯3
    call_e1_rows = df.index[
        (df["Call_FirstEntry_Emoji"] == EMOJI_EXEC) |
        (df["Call_DeferredEntry_Emoji"] == EMOJI_RECLAIM)
    ]
    call_e2_rows = df.index[df["Call_SecondEntry_Emoji"] == "🎯2"]

    if len(call_e1_rows) > 0 and len(call_e2_rows) > 0:
        i_e1 = df.index.get_loc(call_e1_rows[0])
        i_e2 = df.index.get_loc(call_e2_rows[0])
        if i_e2 > i_e1:
            crossed = any(
                np.isfinite(f_arr[i]) and np.isfinite(ib_h_arr[i]) and f_arr[i] > ib_h_arr[i]
                for i in range(i_e1, i_e2 + 1)
            )
            if not crossed:
                for i in range(i_e2 + 1, n - 1):
                    f_prev, f_curr = f_arr[i - 1], f_arr[i]
                    ib_prev, ib_curr = ib_h_arr[i - 1], ib_h_arr[i]
                    if not all(np.isfinite(v) for v in [f_prev, f_curr, ib_prev, ib_curr]):
                        continue
                    if f_prev < ib_prev and f_curr >= ib_curr:
                        j = i + 1
                        if np.isfinite(f_arr[j]) and f_arr[j] > f_curr:
                            ok = True
                            if use_physics:
                                sign = last_vector_sign(j)
                                if sign is not None and sign <= 0:
                                    ok = False
                                if ok and not vol_composite_ok(j):
                                    ok = False
                            if ok:
                                df.at[df.index[j], "Call_ThirdEntry_Emoji"] = "🎯3"
                                break

    return df