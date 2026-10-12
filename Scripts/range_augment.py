#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
range_augment.py - data augmentation for continuous multi-day time ranges
=========================================================================

Input: annotated JSONL in the format produced by time_range_generation_1127.py
    sentence, time_expression_substring, time_expression_substring_index, time_expression_type (range_N_...),
    reference_year/month/day, reference_weekday(int), reference_weekday_name,
    time_range_start, time_range_end (YYYY-MM-DD or null), start_weekday, end_weekday,
    start_weekday_name, end_weekday_name, duration, duration_unit, [id]
Output: JSONL with augmented samples only, same field layout as the input, plus aug_strategy /
    aug_source_id (off with --no-meta). Also writes <output>.report.json and <output>.audit.jsonl.


Usage
-----
  python range_augment.py --input ranges.jsonl --output augmented_ranges.jsonl
  python range_augment.py --input ranges.jsonl --audit-only
  python range_augment.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_core import (  # noqa: E402
    FLAGS, HOLIDAY, HOLIDAY_SURFACES, INT_TO_WORD, INT_WORD, LABEL_YEAR_MAX, LABEL_YEAR_MIN,
    MONTH_NAME, ORDINAL_ANY, WD_NAMES, WEEKDAY, YEAR, Augmenter, Candidate, Ctx, EngineConfig,
    LabelIO, Spec, TypeSpec, add_months, add_units, apply_case, combo, detect_polarity, gspan,
    group_edits, gval, holiday_key, infer_month_day, infer_pad_style, is_business_day, last_dom,
    load_jsonl, month_range, month_shift, number_unit_edits, parse_int_word, parse_iso,
    parse_ordinal, pick_by_polarity, print_report, quarter_range, render_count, render_day,
    render_month, render_num, render_weekday, resolve_holiday, safe_date,
    sample_length, sample_month_day, sample_text_date, sample_text_year, strat_ref_full, strat_ref_jitter, strat_ref_week_shift,
    strat_ref_year, text_candidate, week_bounds, weekday_like, weekday_str_to_int,
    month_str_to_int, year_range, _clause_window,
)

# =========================================================
# CONFIG
# =========================================================

INPUT_JSONL = "./range_dataset.jsonl"
OUTPUT_JSONL = "./augmented_ranges.jsonl"

DEFAULT_COUNT = 3000
AUGMENT_COUNTS: Dict[str, int] = {name: DEFAULT_COUNT for name in [
    "range_1_in_month", "range_2_in_year", "range_3_this_last_next_week", "range_4_this_last_next_month",
    "range_5_this_last_next_quarter", "range_6_this_last_next_year", "range_7_last_next_n_units",
    "range_8_since_date", "range_9_until_date", "range_10_from_or_starting_date",
    "range_11_from_date_to_date", "range_12_between_date_and_date", "range_13_month_day_hyphen_day",
    "range_14_ymd_iso_range", "range_15_year_hyphen_year", "range_16_from_year_to_year",
    "range_17_between_year_and_year", "range_18_numeric_md_range", "range_19_for_n_units_starting_date",
    "range_20_n_unit_period_ending_date", "range_21_all_week_month_year", "range_22_entire_month_of_month",
    "range_23_last_n_trading_business_days", "range_24_n_consecutive_days", "range_25_deadline_by_before_after_starting_ending",
    "range_26_in_n_months", "range_27_phase_of_month", "range_28_week_of_or_starting",
    "range_29_weekday_hyphen_weekday", "range_30_until_rel_unit",
]}

MIN_DURATION_DAYS = 2   # closed ranges must span at least 2 days ("multi-day"); open ranges are exempt

SPEC = Spec()
ENGINE = EngineConfig()

# =========================================================
# Patterns (from the 1127 script; a few named groups added and range_23 relaxed to fix bugs)
# =========================================================

DATE_SEP_YMD = r"[-/\.]"
MD = lambda p: rf"(?P<{p}month>{MONTH_NAME})\s+(?P<{p}day>{ORDINAL_ANY})"  # noqa: E731

P = {
    "range_1_in_month": rf"\b(?P<prefix>in|during|throughout|over)\s+(?P<month_name>{MONTH_NAME})(?:\s+(?P<year>{YEAR}))?\b",
    "range_2_in_year": rf"\b(?P<prefix>in|during|throughout|over)\s+(?P<year>{YEAR})\b",
    "range_3_this_last_next_week": r"\b(?P<rel>this|last|next)\s+week\b",
    "range_4_this_last_next_month": r"\b(?P<rel>this|last|next)\s+month\b",
    "range_5_this_last_next_quarter": r"\b(?P<rel>this|last|next)\s+quarter\b",
    "range_6_this_last_next_year": r"\b(?P<rel>this|last|next)\s+year\b",
    "range_7_last_next_n_units": (
        rf"\b(?:(?:in|over|during|for)\s+)?(?:the\s+)?(?P<dir>last|past|previous|next|coming)\s+"
        rf"(?P<num>{INT_WORD})\s+(?P<unit>days?|weeks?|months?|years?)\b"),
    "range_8_since_date": (
        rf"\b(?:since)\s+((?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?"
        rf"|(?P<year2>{YEAR}))\b"),
    "range_9_until_date": (
        rf"\b(?P<kw>until|till|til|up to|thru|through)\s+((?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})"
        rf"(?:\s*,?\s*(?P<year>{YEAR}))?|(?P<year2>{YEAR}))\b"),
    "range_10_from_or_starting_date": (
        rf"\b(?:from\s+|starting\s+(?:on\s+)?|beginning\s+(?:on\s+)?|effective\s+)"
        rf"(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?\b"),
    "range_11_from_date_to_date": (
        rf"\bfrom\s+(?P<start_month>{MONTH_NAME})\s+(?P<start_day>{ORDINAL_ANY})(?:\s*,?\s*(?P<start_year>{YEAR}))?\s+"
        rf"(?:to|through|thru)\s+(?:(?P<end_month>{MONTH_NAME})\s+)?(?P<end_day>{ORDINAL_ANY})(?:\s*,?\s*(?P<end_year>{YEAR}))?\b"),
    "range_12_between_date_and_date": (
        rf"\bbetween\s+(?P<start_month>{MONTH_NAME})\s+(?P<start_day>{ORDINAL_ANY})(?:\s*,?\s*(?P<start_year>{YEAR}))?\s+"
        rf"and\s+(?:(?P<end_month>{MONTH_NAME})\s+)?(?P<end_day>{ORDINAL_ANY})(?:\s*,?\s*(?P<end_year>{YEAR}))?\b"),
    "range_13_month_day_hyphen_day": (
        rf"\b(?P<month_name>{MONTH_NAME})\s+(?P<start_day>{ORDINAL_ANY})\s*[-–]\s*(?P<end_day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?\b"),
    "range_14_ymd_iso_range": (
        rf"\b(?P<y1>{YEAR}){DATE_SEP_YMD}(?P<m1>0[1-9]|1[0-2]){DATE_SEP_YMD}(?P<d1>0[1-9]|[12]\d|3[01])\s*[-–]\s*"
        rf"(?P<y2>{YEAR}){DATE_SEP_YMD}(?P<m2>0[1-9]|1[0-2]){DATE_SEP_YMD}(?P<d2>0[1-9]|[12]\d|3[01])\b"),
    "range_15_year_hyphen_year": rf"\b(?P<start_year>{YEAR})\s*[-–]\s*(?P<end_year>{YEAR})\b",
    "range_16_from_year_to_year": rf"\bfrom\s+(?P<start_year>{YEAR})\s+(?:to|through|thru)\s+(?P<end_year>{YEAR})\b",
    "range_17_between_year_and_year": rf"\bbetween\s+(?P<start_year>{YEAR})\s+and\s+(?P<end_year>{YEAR})\b",
    "range_18_numeric_md_range": (
        r"\b(?P<m1>0?[1-9]|1[0-2])/(?P<d1>0?[1-9]|[12]\d|3[01])\s*[-–]\s*"
        r"(?P<m2>0?[1-9]|1[0-2])/(?P<d2>0?[1-9]|[12]\d|3[01])\b"),
    "range_19_for_n_units_starting_date": (
        rf"\bfor\s+(?P<num>{INT_WORD})\s+(?P<unit>days?|weeks?)\s+(?:starting\s+(?:on\s+)?|from\s+)"
        rf"(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?\b"),
    "range_20_n_unit_period_ending_date": (
        rf"\b(?P<art>a\s+)?(?P<num>{INT_WORD})\s*-\s*(?P<unit>day|days|week|weeks)\s+(?:period|window)\s+ending\s+"
        rf"(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?\b"),
    "range_21_all_week_month_year": (
        r"\b(?:all|the whole|the entire|for the whole|for the entire|throughout)\s+(?P<unit>week|month|year)\b"),
    "range_22_entire_month_of_month": (
        rf"\b(?:the\s+)?(?:whole|entire)\s+month\s+of\s+(?P<month_name>{MONTH_NAME})(?:\s+(?P<year>{YEAR}))?\b"),
    "range_23_last_n_trading_business_days": (
        rf"\b(?:(?:over|during|in|for)\s+)?(?:the\s+)?(?P<dir>last|past|previous|next|coming)\s+"
        rf"(?P<num>{INT_WORD})\s+(?P<kind>trading|business|working)\s+(?P<dunit>days?)\b"),
    "range_24_n_consecutive_days": (
        rf"\b(?P<num>{INT_WORD})\s+(?P<kind>consecutive|straight)\s+days\b"
        rf"|\b(?P<num2>{INT_WORD})\s+days\s+in\s+a\s+row\b"),
    "range_25_deadline_by_before_after_starting_ending": (
        rf"\b(?P<kind>by|before|after|starting|beginning|effective|ending)\s+"
        rf"(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?\b"),
    "range_26_in_n_months": rf"\bin\s+(?P<num>{INT_WORD})\s+(?P<unit>months?)\b",
    "range_27_phase_of_month": (
        rf"\b(?:(?:in|during|throughout)\s+)?(?P<phase>early|mid|middle|late)\s*[-\s]+(?P<month_name>{MONTH_NAME})"
        rf"(?:\s+(?P<year>{YEAR}))?\b"),
    "range_28_week_of_or_starting": (
        rf"\b(?:the\s+)?week\s+(?P<kw>of|starting|beginning|commencing)\s+"
        rf"(?:(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})(?:\s*,?\s*(?P<year>{YEAR}))?|(?P<holiday>{HOLIDAY}))\b"),
    "range_29_weekday_hyphen_weekday": rf"\b(?P<start_wd>{WEEKDAY})\s*[-–]\s*(?P<end_wd>{WEEKDAY})\b",
    "range_30_until_rel_unit": (
        r"\b(?P<kw>until|till|til|up to|thru|through)\s+(?P<rel>this|next|last)\s+(?P<unit>week|month|year)\b"),
}
REGEX = {k: re.compile(v, FLAGS) for k, v in P.items()}


def pol(m, s):
    return detect_polarity(s, *m.span())


def pol_default(m, s, default):
    p = pol(m, s)
    return default if p == "neutral" else p


def md(m, mg="month_name", dg="day", yg="year") -> Tuple[int, int, Optional[int]]:
    y = gval(m, yg) if yg else None
    return month_str_to_int(m.group(mg)), parse_ordinal(m.group(dg)), (int(y) if y else None)


def md_date(m, ref, s, spec, polarity, mg="month_name", dg="day", yg="year") -> date:
    mm, dd, y = md(m, mg, dg, yg)
    return date(y, mm, dd) if y else infer_month_day(ref, mm, dd, polarity, spec)


# =========================================================
# Resolvers  (m, ref, sentence, spec) -> (start|None, end|None)
# =========================================================

def r1(m, ref, s, spec):
    mm = month_str_to_int(m.group("month_name"))
    y = gval(m, "year")
    if not y:
        y = infer_month_day(ref, mm, 15, pol(m, s), spec).year
    return month_range(int(y), mm)


def r2(m, ref, s, spec):
    return year_range(int(m.group("year")))


def _rel_off(m):
    return {"this": 0, "last": -1, "next": 1}[m.group("rel").lower()]


def r3(m, ref, s, spec):
    a, _ = week_bounds(ref, spec)
    a += timedelta(weeks=_rel_off(m))
    return a, a + timedelta(days=6)


def r4(m, ref, s, spec):
    return month_range(*month_shift(ref.year, ref.month, _rel_off(m)))


def r5(m, ref, s, spec):
    idx = ref.year * 4 + (ref.month - 1) // 3 + _rel_off(m)
    return quarter_range(idx // 4, idx % 4 + 1)


def r6(m, ref, s, spec):
    return year_range(ref.year + _rel_off(m))


def last_next_n(ref: date, back: bool, n: int, unit: str, spec: Spec) -> Tuple[date, date]:
    one = timedelta(days=1)
    if spec.last_n_includes_today:
        return (add_units(ref, unit, -n) + one, ref) if back else (ref, add_units(ref, unit, n) - one)
    return (add_units(ref, unit, -n), ref - one) if back else (ref + one, add_units(ref, unit, n))


def r7(m, ref, s, spec):
    back = m.group("dir").lower() in ("last", "past", "previous")
    return last_next_n(ref, back, parse_int_word(m.group("num")), m.group("unit"), spec)


def r8(m, ref, s, spec):
    if gval(m, "year2"):
        start = date(int(m.group("year2")), 1, 1)
    else:
        mm, dd, y = md(m)
        start = date(y, mm, dd) if y else pick_by_polarity(
            [d for d in (safe_date(yy, mm, dd) for yy in range(ref.year - 8, ref.year + 1)) if d], ref, "past", spec)
    if start is None or start > ref:
        raise ValueError("since: start after reference")
    return start, ref


def r9(m, ref, s, spec):
    if gval(m, "year2"):
        end = date(int(m.group("year2")), 12, 31)
    else:
        end = md_date(m, ref, s, spec, "future")
    if spec.until_start_is_ref:
        if end < ref:
            raise ValueError("until: end before reference")
        return ref, end
    return None, end


def r10(m, ref, s, spec):
    return md_date(m, ref, s, spec, pol_default(m, s, "future")), None


def resolve_pair(ref: date, sm: int, sd: int, sy: Optional[int], em: int, ed: int, ey: Optional[int],
                 polarity: str, spec: Spec) -> Tuple[date, date]:
    if sy:
        start = date(sy, sm, sd)
    elif ey:  # only the end has a year: derive the start year from it (Dec 20 - Jan 5, 2025)
        start = date(ey, sm, sd)
        if start > date(ey, em, ed):
            start = date(ey - 1, sm, sd)
    else:
        start = infer_month_day(ref, sm, sd, polarity, spec)
    if ey:
        end = date(ey, em, ed)
    else:
        cands = [d for d in (safe_date(start.year + k, em, ed) for k in (0, 1)) if d and d >= start]
        if not cands:
            raise ValueError("cannot place end date")
        end = cands[0]
    if end < start:
        raise ValueError("inverted range")
    return start, end


def r11(m, ref, s, spec):
    sm, sd, sy = md(m, "start_month", "start_day", "start_year")
    em = month_str_to_int(gval(m, "end_month") or m.group("start_month"))
    ed = parse_ordinal(m.group("end_day"))
    ey = int(m.group("end_year")) if gval(m, "end_year") else None
    return resolve_pair(ref, sm, sd, sy, em, ed, ey, pol(m, s), spec)


def r13(m, ref, s, spec):
    mm = month_str_to_int(m.group("month_name"))
    d1, d2 = parse_ordinal(m.group("start_day")), parse_ordinal(m.group("end_day"))
    if d2 <= d1:
        raise ValueError("inverted day range")  # "June 28-3": the old code emitted an inverted range
    y = gval(m, "year")
    start = date(int(y), mm, d1) if y else infer_month_day(ref, mm, d1, pol(m, s), spec)
    return start, date(start.year, mm, d2)


def r14(m, ref, s, spec):
    a = date(int(m.group("y1")), int(m.group("m1")), int(m.group("d1")))
    b = date(int(m.group("y2")), int(m.group("m2")), int(m.group("d2")))
    if b < a:
        raise ValueError("inverted")
    return a, b


def r15(m, ref, s, spec):
    sy, ey = int(m.group("start_year")), int(m.group("end_year"))
    if ey <= sy:
        raise ValueError("inverted/equal year range")  # "2022-2020": the old code emitted an inverted range
    return date(sy, 1, 1), date(ey, 12, 31)


def r18(m, ref, s, spec):
    return resolve_pair(ref, int(m.group("m1")), int(m.group("d1")), None,
                        int(m.group("m2")), int(m.group("d2")), None, pol(m, s), spec)


def r19(m, ref, s, spec):
    n = parse_int_word(m.group("num"))
    start = md_date(m, ref, s, spec, pol_default(m, s, "future"))
    days = n if m.group("unit").lower().startswith("day") else 7 * n
    return start, start + timedelta(days=days - 1)


def r20(m, ref, s, spec):
    n = parse_int_word(m.group("num"))
    end = md_date(m, ref, s, spec, pol_default(m, s, "past"))
    days = n if m.group("unit").lower().startswith("day") else 7 * n
    return end - timedelta(days=days - 1), end


def r21(m, ref, s, spec):
    u = m.group("unit").lower()
    if u == "week":
        return week_bounds(ref, spec)
    return month_range(ref.year, ref.month) if u == "month" else year_range(ref.year)


def r22(m, ref, s, spec):
    return r1(m, ref, s, spec)


def r23(m, ref, s, spec):
    n = parse_int_word(m.group("num"))
    back = m.group("dir").lower() in ("last", "past", "previous")
    step = timedelta(days=-1 if back else 1)
    cur = ref if spec.last_n_includes_today else ref + step
    days: List[date] = []
    while len(days) < n:
        if is_business_day(cur, spec):
            days.append(cur)
        cur += step
    return min(days), max(days)


def r24(m, ref, s, spec):
    parse_int_word(gval(m, "num") or m.group("num2"))
    return None, None


def r25(m, ref, s, spec):
    kind = m.group("kind").lower()
    dt = md_date(m, ref, s, spec, pol_default(m, s, "future"))
    if kind in ("by", "ending"):
        return None, dt
    if kind == "before":
        return None, dt - timedelta(days=1)
    if kind == "after":
        return dt + timedelta(days=1), None
    return dt, None


def r26(m, ref, s, spec):
    t = add_months(ref, parse_int_word(m.group("num")))
    return month_range(t.year, t.month)


def r27(m, ref, s, spec):
    phase = m.group("phase").lower()
    mm = month_str_to_int(m.group("month_name"))
    rep = 5 if phase == "early" else (15 if phase in ("mid", "middle") else 25)
    y = gval(m, "year")
    y = int(y) if y else infer_month_day(ref, mm, rep, pol(m, s), spec).year
    if phase == "early":
        return date(y, mm, 1), date(y, mm, 10)
    if phase in ("mid", "middle"):
        return date(y, mm, 11), date(y, mm, 20)
    return date(y, mm, 21), date(y, mm, last_dom(y, mm))


def r28(m, ref, s, spec):
    p = pol(m, s)
    d = resolve_holiday(m.group("holiday"), ref, p, spec) if gval(m, "holiday") else md_date(m, ref, s, spec, p)
    if m.group("kw").lower() == "of":
        return week_bounds(d, spec)
    return d, d + timedelta(days=6)  # week starting/beginning/commencing X


_RECURRING = re.compile(r"\b(?:every|each|weekly|daily|open|opens|hours|usually|typically|always|"
                        r"normally|routinely|regularly|weekdays)\b", re.I)


def f29(m, s, spec):
    if spec.skip_recurring_weekday_ranges:
        lo, hi = _clause_window(s, *m.span())
        if _RECURRING.search(s[lo:m.start()]) or _RECURRING.search(s[m.end():hi]):
            return "recurring_weekday_range"
    return None


def r29(m, ref, s, spec):
    sw, ew = weekday_str_to_int(m.group("start_wd")), weekday_str_to_int(m.group("end_wd"))
    length = (ew - sw) % 7
    if length == 0:
        raise ValueError("same weekday")
    mon, _ = week_bounds(ref, spec)
    cands = []
    for k in range(-3, 4):
        a = mon + timedelta(weeks=k, days=(sw - spec.week_start) % 7)
        cands.append((a, a + timedelta(days=length)))
    p = pol(m, s)
    if p == "future":
        ok = [c for c in cands if c[0] >= ref] or [c for c in cands if c[1] >= ref]
        return min(ok)
    if p == "past":
        ok = [c for c in cands if c[1] <= ref] or [c for c in cands if c[0] <= ref]
        return max(ok)
    containing = [c for c in cands if c[0] <= ref <= c[1]]
    if containing:
        return containing[0]
    return min(cands, key=lambda c: (abs((c[0] + (c[1] - c[0]) / 2 - ref).days), c[0] < ref))


def r30(m, ref, s, spec):
    kw, rel, unit = m.group("kw").lower(), m.group("rel").lower(), m.group("unit").lower()
    off = {"this": 0, "last": -1, "next": 1}[rel]
    if rel == "next" and kw not in ("through", "thru"):
        off = 0  # until next week -> end of this week; through next week -> includes next week
    if unit == "week":
        end = week_bounds(ref, spec)[1] + timedelta(weeks=off)
    elif unit == "month":
        end = month_range(*month_shift(ref.year, ref.month, off))[1]
    else:
        end = date(ref.year + off, 12, 31)
    if spec.until_start_is_ref:
        if end < ref:
            raise ValueError("until: end before reference")
        return ref, end
    return None, end


# =========================================================
# Text rewriting strategies
# =========================================================

def _vowel_sound(n: int) -> bool:
    return n in (8, 11, 18) or 80 <= n <= 89


def sample_n(ctx: Ctx, template: str, unit: str, old: int, min_n: int = 2) -> int:
    hi = {"day": 30, "week": 12, "month": 18, "year": 10}.get(unit.lower().rstrip("s"), 10)
    pool = [k for k in INT_TO_WORD if min_n <= k <= max(hi, 12)] if not template.isdigit() \
        else list(range(min_n, hi + 1))
    pool = [k for k in pool if k != old] or [old + 1]
    return ctx.rng.choice(pool)


def _date_edits(m, d: date, mg: str, dg: str, yg: Optional[str], pad: Optional[bool] = None) -> List[Tuple[int, int, str]]:
    repl: Dict[str, str] = {}
    if gval(m, mg) is not None:
        repl[mg] = render_month(m.group(mg), d.month)
    if gval(m, dg) is not None:
        v = m.group(dg)
        repl[dg] = render_day(v, d.day, pad if v[:1].isdigit() else None)
    if yg and gval(m, yg) is not None:
        repl[yg] = f"{d.year:04d}"
    return group_edits(m, repl)


def _sample_year(ctx: Ctx, explicit: bool) -> int:
    if explicit:
        return sample_text_year(ctx)
    return ctx.ref.year + ctx.rng.choice((-1, 0, 0, 1))


def _sample_start(ctx: Ctx, explicit_year: bool, max_day: int = 31) -> date:
    y = _sample_year(ctx, explicit_year)
    mo, d = sample_month_day(ctx.rng)
    return date(y, mo, min(d, max_day, last_dom(y, mo)))


def text_single_date(ctx: Ctx) -> Optional[Candidate]:
    """range_8/9/10/19/20/25/28: change the start/end date (month + day [+ year])."""
    m = ctx.m
    if gval(m, "holiday"):
        return text_holiday(ctx)
    if gval(m, "year2"):
        y = ctx.rng.randint(ctx.cfg.text_year_min, min(ctx.cfg.text_year_max, ctx.ref.year))
        return text_candidate(ctx, group_edits(m, {"year2": str(y)}), "text_date")
    if gval(m, "month_name") is None:
        return None
    d = _sample_start(ctx, bool(gval(m, "year")))
    pad = infer_pad_style(m, ["day"], ctx.rng)
    return text_candidate(ctx, _date_edits(m, d, "month_name", "day", "year", pad), "text_date")


def text_num_and_date(ctx: Ctx) -> Optional[Candidate]:
    """range_19/20: change both the count and the date."""
    m = ctx.m
    edits: List[Tuple[int, int, str]] = []
    if ctx.rng.random() < 0.6:
        v = m.group("num")
        n = sample_n(ctx, v, m.group("unit"), parse_int_word(v), min_n=2)
        if gval(m, "art") and _vowel_sound(n):
            n = 7 if parse_int_word(v) != 7 else 5  # avoid "a eight-day"
        e = number_unit_edits(ctx.sentence, m, "num", n)
        if e:
            edits += e
    if not edits or ctx.rng.random() < 0.6:
        d = _sample_start(ctx, bool(gval(m, "year")))
        edits += _date_edits(m, d, "month_name", "day", "year", infer_pad_style(m, ["day"], ctx.rng))
    return text_candidate(ctx, edits, "text_num_date")


def text_pair(ctx: Ctx) -> Optional[Candidate]:
    """range_11/12: from/between X to/and Y."""
    m, rng = ctx.m, ctx.rng
    has_sy, has_ey = bool(gval(m, "start_year")), bool(gval(m, "end_year"))
    same_month = gval(m, "end_month") is None
    for _ in range(20):
        start = _sample_start(ctx, has_sy or has_ey, max_day=30 if same_month else 31)
        if same_month:
            room = last_dom(start.year, start.month) - start.day
            if room < 1:
                continue
            end = start + timedelta(days=rng.randint(1, min(room, 20)))
        else:
            end = start + timedelta(days=sample_length(rng))
        break
    else:
        return None
    pad = infer_pad_style(m, ["start_day", "end_day"], rng)
    edits = _date_edits(m, start, "start_month", "start_day", "start_year", pad)
    edits += _date_edits(m, end, "end_month", "end_day", "end_year", pad)
    return text_candidate(ctx, edits, "text_pair")


def text_month_d1_d2(ctx: Ctx) -> Optional[Candidate]:
    """range_13: Month d1-d2[, Y]."""
    m, rng = ctx.m, ctx.rng
    y = _sample_year(ctx, bool(gval(m, "year")))
    mo = rng.randint(1, 12)
    ld = last_dom(y, mo)
    d1 = rng.randint(1, ld - 1)
    d2 = rng.randint(d1 + 1, min(ld, d1 + 14))
    pad = infer_pad_style(m, ["start_day", "end_day"], rng)
    repl = {"month_name": render_month(m.group("month_name"), mo),
            "start_day": render_day(m.group("start_day"), d1, pad if m.group("start_day")[:1].isdigit() else None),
            "end_day": render_day(m.group("end_day"), d2, pad if m.group("end_day")[:1].isdigit() else None)}
    if gval(m, "year"):
        repl["year"] = str(y)
    return text_candidate(ctx, group_edits(m, repl), "text_month_days")


def text_iso_pair(ctx: Ctx) -> Optional[Candidate]:
    """range_14: two full YYYY-MM-DD dates."""
    a = sample_text_date(ctx)
    b = a + timedelta(days=sample_length(ctx.rng))
    repl = {"y1": f"{a.year:04d}", "m1": f"{a.month:02d}", "d1": f"{a.day:02d}",
            "y2": f"{b.year:04d}", "m2": f"{b.month:02d}", "d2": f"{b.day:02d}"}
    return text_candidate(ctx, group_edits(ctx.m, repl), "text_iso_pair")


def text_year_pair(ctx: Ctx) -> Optional[Candidate]:
    """range_15/16/17"""
    span = ctx.rng.choice((1, 1, 2, 2, 3, 4, 5, 10))
    sy = min(sample_text_year(ctx), ctx.cfg.text_year_max - span)
    if detect_polarity(ctx.sentence, ctx.st, ctx.ed) == "past":
        sy = min(sy, ctx.ref.year - span)
    ey = sy + span
    return text_candidate(ctx, group_edits(ctx.m, {"start_year": str(sy), "end_year": str(ey)}), "text_year_pair")


def text_numeric_pair(ctx: Ctx) -> Optional[Candidate]:
    """range_18: m/d-m/d."""
    m = ctx.m
    a = _sample_start(ctx, False)
    b = a + timedelta(days=sample_length(ctx.rng))
    pad = infer_pad_style(m, ["m1", "d1", "m2", "d2"], ctx.rng)
    repl = {"m1": render_num(m.group("m1"), a.month, pad), "d1": render_num(m.group("d1"), a.day, pad),
            "m2": render_num(m.group("m2"), b.month, pad), "d2": render_num(m.group("d2"), b.day, pad)}
    return text_candidate(ctx, group_edits(m, repl), "text_numeric_pair")


def text_month_year(ctx: Ctx) -> Optional[Candidate]:
    """range_1/22/27: month (+ year; range_27 can also swap early/mid/late)."""
    m, rng = ctx.m, ctx.rng
    repl: Dict[str, str] = {}
    old = month_str_to_int(m.group("month_name"))
    repl["month_name"] = render_month(m.group("month_name"), rng.choice([x for x in range(1, 13) if x != old]))
    if gval(m, "year"):
        new_m = month_str_to_int(repl["month_name"].rstrip(".")) if repl["month_name"].rstrip(".").lower() != "may" else 5
        y = sample_text_year(ctx)
        p = detect_polarity(ctx.sentence, ctx.st, ctx.ed)
        if p == "past" and date(y, new_m, 1) > ctx.ref:
            y -= 1  # past tense: the month must not start after the reference date
        elif p == "future" and date(y, new_m, last_dom(y, new_m)) < ctx.ref:
            y += 1
        repl["year"] = str(y)
    if gval(m, "phase") and rng.random() < 0.5:
        ph = m.group("phase")
        cur = "mid" if ph.lower() in ("mid", "middle") else ph.lower()
        new = rng.choice([p for p in ("early", "mid", "late") if p != cur])
        if new == "mid" and ph.lower() == "middle":
            new = "middle"
        repl["phase"] = apply_case(ph, new)
    return text_candidate(ctx, group_edits(m, repl), "text_month")


def text_year(ctx: Ctx) -> Optional[Candidate]:
    """range_2"""
    old = int(ctx.m.group("year"))
    y = sample_text_year(ctx)
    if y == old:
        return None
    return text_candidate(ctx, group_edits(ctx.m, {"year": str(y)}), "text_year")


def text_count(ctx: Ctx) -> Optional[Candidate]:
    """range_7/23/24/26: change the count (range_7 can also change the unit), keeping singular/plural in sync."""
    m, rng = ctx.m, ctx.rng
    g = "num" if gval(m, "num") else "num2"
    v = m.group(g)
    unit = gval(m, "unit") or "day"
    new_unit = None
    if gval(m, "unit") and ctx.tspec.name.startswith("range_7") and rng.random() < 0.3:
        base = unit.lower().rstrip("s")
        new_unit = rng.choice([u for u in ("day", "week", "month", "year") if u != base])
    n = sample_n(ctx, v, new_unit or unit, parse_int_word(v), min_n=2)
    new_num = render_count(v, n)
    if new_num is None:
        return None
    edits = [(*gspan(m, g), new_num)]  # type: ignore[misc]
    if new_unit:
        edits.append((*gspan(m, "unit"), apply_case(unit, new_unit + "s")))  # type: ignore[misc]
    elif gval(m, "unit"):
        edits.append((*gspan(m, "unit"), apply_case(unit, unit.lower().rstrip("s") + "s")))  # type: ignore[misc]
    if gval(m, "dunit"):
        edits.append((*gspan(m, "dunit"), apply_case(m.group("dunit"), "days")))  # type: ignore[misc]
    return text_candidate(ctx, edits, "text_count")


def text_holiday(ctx: Ctx) -> Optional[Candidate]:
    v = ctx.m.group("holiday")
    new = ctx.rng.choice([h for h in HOLIDAY_SURFACES if holiday_key(h) != holiday_key(v)])
    if "’" in v:
        new = new.replace("'", "’")
    return text_candidate(ctx, group_edits(ctx.m, {"holiday": new}), "text_holiday")


def text_week_of(ctx: Ctx) -> Optional[Candidate]:
    """range_28: a date or a holiday (replaced by another of the same kind)."""
    if gval(ctx.m, "holiday"):
        return text_holiday(ctx)
    return text_single_date(ctx)


def text_weekday_pair(ctx: Ctx) -> Optional[Candidate]:
    m, rng = ctx.m, ctx.rng
    s0, e0 = weekday_str_to_int(m.group("start_wd")), weekday_str_to_int(m.group("end_wd"))
    for _ in range(20):
        s = rng.randint(0, 6)
        e = (s + rng.randint(1, 6)) % 7
        if (s, e) != (s0, e0):
            break
    repl = {"start_wd": render_weekday(m.group("start_wd"), s), "end_wd": render_weekday(m.group("end_wd"), e)}
    return text_candidate(ctx, group_edits(m, repl), "text_weekdays")


# =========================================================
# Type table
# =========================================================

def _w(*items):
    return list(items)


REF_MIX = _w(("ref_full", 0.3, strat_ref_full), ("ref_jitter", 0.2, strat_ref_jitter), ("ref_year", 0.1, strat_ref_year))


def with_text(fn, name, w_text=0.2, w_combo=0.2, extra=()):
    return REF_MIX + list(extra) + _w((name, w_text, fn), (name + "+ref_full", w_combo, combo(fn, strat_ref_full, name + "+ref_full")))


REF_ONLY = _w(("ref_full", 0.5, strat_ref_full), ("ref_jitter", 0.35, strat_ref_jitter),
              ("ref_week_shift", 0.15, strat_ref_week_shift))


def build_types() -> Dict[str, TypeSpec]:
    T: Dict[str, TypeSpec] = {}

    def add(name, resolve, strategies, flt=None):
        T[name] = TypeSpec(name, REGEX[name], resolve, strategies, flt)

    text_only = lambda fn, n: _w((n, 0.5, fn), (n + "+ref_full", 0.5, combo(fn, strat_ref_full, n + "+ref_full")))  # noqa: E731
    add("range_1_in_month", r1, with_text(text_month_year, "text_month"))
    add("range_2_in_year", r2, text_only(text_year, "text_year"))
    add("range_3_this_last_next_week", r3, REF_ONLY)
    add("range_4_this_last_next_month", r4, REF_ONLY)
    add("range_5_this_last_next_quarter", r5, REF_ONLY)
    add("range_6_this_last_next_year", r6, REF_ONLY)
    add("range_7_last_next_n_units", r7, with_text(text_count, "text_count"))
    add("range_8_since_date", r8, with_text(text_single_date, "text_date"))
    add("range_9_until_date", r9, with_text(text_single_date, "text_date"))
    add("range_10_from_or_starting_date", r10, with_text(text_single_date, "text_date"))
    add("range_11_from_date_to_date", r11, with_text(text_pair, "text_pair"))
    add("range_12_between_date_and_date", r11, with_text(text_pair, "text_pair"))
    add("range_13_month_day_hyphen_day", r13, with_text(text_month_d1_d2, "text_month_days"))
    add("range_14_ymd_iso_range", r14, text_only(text_iso_pair, "text_iso_pair"))
    add("range_15_year_hyphen_year", r15, text_only(text_year_pair, "text_year_pair"))
    add("range_16_from_year_to_year", r15, text_only(text_year_pair, "text_year_pair"))
    add("range_17_between_year_and_year", r15, text_only(text_year_pair, "text_year_pair"))
    add("range_18_numeric_md_range", r18, with_text(text_numeric_pair, "text_numeric_pair"))
    add("range_19_for_n_units_starting_date", r19, with_text(text_num_and_date, "text_num_date"))
    add("range_20_n_unit_period_ending_date", r20, with_text(text_num_and_date, "text_num_date"))
    add("range_21_all_week_month_year", r21, REF_ONLY)
    add("range_22_entire_month_of_month", r22, with_text(text_month_year, "text_month"))
    add("range_23_last_n_trading_business_days", r23, with_text(text_count, "text_count"))
    add("range_24_n_consecutive_days", r24, _w(("text_count", 1.0, text_count)))  # no start/end dates, so changing the reference date is pointless
    add("range_25_deadline_by_before_after_starting_ending", r25, with_text(text_single_date, "text_date"))
    add("range_26_in_n_months", r26, with_text(text_count, "text_count"))
    add("range_27_phase_of_month", r27, with_text(text_month_year, "text_month"))
    add("range_28_week_of_or_starting", r28, with_text(text_week_of, "text_date"))
    add("range_29_weekday_hyphen_weekday", r29,
        with_text(text_weekday_pair, "text_weekdays", extra=[("ref_week_shift", 0.1, strat_ref_week_shift)]), f29)
    add("range_30_until_rel_unit", r30, REF_ONLY)
    return T


# =========================================================
# Label I/O
# =========================================================

def _read(rec):
    s, e = rec.get("time_range_start"), rec.get("time_range_end")
    if "time_range_start" not in rec and "time_range_end" not in rec:
        return None
    return (parse_iso(s) if s else None, parse_iso(e) if e else None)


def _write(out, lab, src):
    s, e = lab
    out["time_range_start"] = s.isoformat() if s else None
    out["time_range_end"] = e.isoformat() if e else None
    for key, d in (("start_weekday", s), ("end_weekday", e)):
        out[key] = weekday_like(src.get(key, 0) if src.get(key) is not None else 0, d.weekday()) if d else None
        out[key + "_name"] = WD_NAMES[d.weekday()] if d else None
    out["duration"] = (e - s).days + 1 if (s and e) else None
    out["duration_unit"] = src.get("duration_unit", "day")


def _validate(lab, ts):
    if not isinstance(lab, tuple) or len(lab) != 2:
        return "label_not_range"
    s, e = lab
    if s is None and e is None:
        return None if ts.name.startswith("range_24") else "empty_range"
    for d in (s, e):
        if d is not None and not (LABEL_YEAR_MIN <= d.year <= LABEL_YEAR_MAX):
            return "label_out_of_range"
    if s and e:
        if e < s:
            return "inverted_range"
        if (e - s).days + 1 < MIN_DURATION_DAYS:
            return "too_short"
    return None


def _key(lab):
    s, e = lab
    return f"{s.isoformat() if s else None}|{e.isoformat() if e else None}"


LABEL_IO = LabelIO(read=_read, write=_write, validate=_validate, key=_key)
BUILD = build_types()


# =========================================================
# Self-test
# =========================================================

def selftest() -> int:
    spec = Spec()
    fails = 0

    def check(name, got, exp):
        nonlocal fails
        ok = got == exp
        fails += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got} expected={exp}")

    def res(t, sentence, ref, sp=spec):
        m = REGEX[t].search(sentence)
        assert m, (t, sentence)
        try:
            if BUILD[t].source_filter and BUILD[t].source_filter(m, sentence, sp):
                return "FILTERED"
            lab = BUILD[t].resolve(m, ref, sentence, sp)
        except ValueError:
            return "REJECT"
        bad = _validate(lab, BUILD[t])
        return bad.upper() if bad else tuple(d.isoformat() if d else None for d in lab)

    ref = date(2025, 11, 1)  # Saturday
    print("== resolver regressions ==")
    check("since June fifteenth (old: 06-01)", res("range_8_since_date", "Prices rose since June fifteenth.", ref), ("2025-06-15", "2025-11-01"))
    check("from Oct 28 to Nov 5 (old: swapped into ~1 year)", res("range_11_from_date_to_date", "The sale runs from Oct 28 to Nov 5.", ref), ("2025-10-28", "2025-11-05"))
    check("from Dec 20 to Jan 5, 2025", res("range_11_from_date_to_date", "Closed from Dec 20 to Jan 5, 2025.", ref), ("2024-12-20", "2025-01-05"))
    check("between March third and March tenth", res("range_12_between_date_and_date", "It was busy between March third and March tenth.", ref), ("2025-03-03", "2025-03-10"))
    check("June 28-3 inverted -> rejected", res("range_13_month_day_hyphen_day", "It ran June 28-3.", ref), "REJECT")
    check("2022-2020 inverted -> rejected", res("range_15_year_hyphen_year", "Data for 2022-2020 is missing.", ref), "REJECT")
    check("last 3 days (excluding today)", res("range_7_last_next_n_units", "Sales grew in the last 3 days.", ref), ("2025-10-29", "2025-10-31"))
    check("next two weeks", res("range_7_last_next_n_units", "Over the next two weeks we will ship.", ref), ("2025-11-02", "2025-11-15"))
    check("last 3 months spans exactly 3 months", res("range_7_last_next_n_units", "in the last 3 months", ref), ("2025-08-01", "2025-10-31"))
    check("last 3 days (including-today convention)", res("range_7_last_next_n_units", "in the last 3 days", ref, Spec(last_n_includes_today=True)), ("2025-10-30", "2025-11-01"))
    check("week starting June 1, 2025", res("range_28_week_of_or_starting", "the week starting June 1, 2025", ref), ("2025-06-01", "2025-06-07"))
    check("week of June 4, 2025", res("range_28_week_of_or_starting", "the week of June 4, 2025", ref), ("2025-06-02", "2025-06-08"))
    check("week of Independence Day (old: unresolved)", res("range_28_week_of_or_starting", "The week of Independence Day was volatile.", ref), ("2025-06-30", "2025-07-06"))
    check("open Mon-Fri -> skipped as recurring", res("range_29_weekday_hyphen_weekday", "We are open Mon-Fri.", ref), "FILTERED")
    check("worked Mon-Fri, past", res("range_29_weekday_hyphen_weekday", "We worked Mon-Fri on this project.", date(2025, 10, 29)), ("2025-10-20", "2025-10-24"))
    check("last 5 trading days (no preposition)", res("range_23_last_n_trading_business_days", "Stocks fell the last 5 trading days.", ref), ("2025-10-27", "2025-10-31"))
    check("next 3 business days skip Thanksgiving", res("range_23_last_n_trading_business_days", "Delivery in the next 3 business days.", date(2025, 11, 26)), ("2025-11-28", "2025-12-02"))
    check("until next week", res("range_30_until_rel_unit", "We will be busy until next week.", ref), (None, "2025-11-02"))
    check("through next week", res("range_30_until_rel_unit", "We will be busy through next week.", ref), (None, "2025-11-09"))
    check("conference lasted March 1-3", res("range_13_month_day_hyphen_day", "The conference lasted March 1-3.", ref), ("2025-03-01", "2025-03-03"))
    check("for ten days starting May 3rd", res("range_19_for_n_units_starting_date", "We travel for ten days starting May 3rd.", ref), ("2026-05-03", "2026-05-12"))
    check("mid-June, future", res("range_27_phase_of_month", "We will be in mid-June.", ref), ("2026-06-11", "2026-06-20"))
    check("in May, past", res("range_1_in_month", "The market rallied in May.", ref), ("2025-05-01", "2025-05-31"))

    print("== end-to-end round trip ==")
    def rec(i, sent, sub, t, start, end):
        st = sent.index(sub)
        r = {"id": f"r{i}", "sentence": sent, "time_expression_substring": sub,
             "time_expression_substring_index": [st, st + len(sub)], "time_expression_type": t,
             "reference_year": 2025, "reference_month": 11, "reference_day": 1,
             "reference_weekday": 5, "reference_weekday_name": "Saturday",
             "time_range_start": start, "time_range_end": end, "duration_unit": "day"}
        _write(r, (parse_iso(start) if start else None, parse_iso(end) if end else None), {"start_weekday": 0, "end_weekday": 0})
        return r
    recs = [
        rec(1, "He was on vacation from June 1st to June 3rd.", "from June 1st to June 3rd", "range_11_from_date_to_date", "2025-06-01", "2025-06-03"),
        rec(2, "Sales grew in the last three days.", "in the last three days", "range_7_last_next_n_units", "2025-10-29", "2025-10-31"),
        rec(3, "We shipped orders from 11/01-11/15.", "11/01-11/15", "range_18_numeric_md_range", "2025-11-01", "2025-11-15"),
        rec(4, "The event runs 2025.11.01-2025.11.10.", "2025.11.01-2025.11.10", "range_14_ymd_iso_range", "2025-11-01", "2025-11-10"),
        rec(5, "We worked Mon-Fri on this project.", "Mon-Fri", "range_29_weekday_hyphen_weekday", "2025-10-27", "2025-10-31"),
        rec(6, "He traded for five consecutive days.", "five consecutive days", "range_24_n_consecutive_days", None, None),
    ]
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest_range.jsonl")
    counts = {r["time_expression_type"]: 25 for r in recs}
    aug = Augmenter(BUILD, LABEL_IO, spec, EngineConfig(seed=7))
    aug.run(recs, counts, tmp)
    outs = load_jsonl(tmp)
    bad = []
    for o in outs:
        t = o["time_expression_type"]
        st, ed = o["time_expression_substring_index"]
        m = REGEX[t].match(o["sentence"], st)
        r = date(o["reference_year"], o["reference_month"], o["reference_day"])
        try:
            lab = BUILD[t].resolve(m, r, o["sentence"], spec)
        except Exception:
            bad.append(("resolve", o)); continue
        if not m or m.end() != ed or _key(lab) != f"{o['time_range_start']}|{o['time_range_end']}" \
                or o["sentence"][st:ed] != o["time_expression_substring"]:
            bad.append(("label", o))
        if o["time_range_start"] and o["time_range_end"]:
            a, b = parse_iso(o["time_range_start"]), parse_iso(o["time_range_end"])
            if b < a or o["duration"] != (b - a).days + 1 or o["start_weekday"] != a.weekday() \
                    or o["start_weekday_name"] != WD_NAMES[a.weekday()] or o["reference_weekday"] != r.weekday():
                bad.append(("fields", o))
        if t.startswith("range_18") and not re.fullmatch(r"\d{2}/\d{2}-\d{2}/\d{2}", o["time_expression_substring"]):
            bad.append(("pad", o))
        if t.startswith("range_14") and not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}-\d{4}\.\d{2}\.\d{2}", o["time_expression_substring"]):
            bad.append(("format", o))
    for why, o in bad[:5]:
        print("    BAD", why, o["time_expression_substring"], o["time_range_start"], o["time_range_end"])
    check(f"generated {len(outs)} records, all pass round-trip/field/format checks", len(bad), 0)
    for ext in ("", ".report.json", ".audit.jsonl"):
        p = os.path.splitext(tmp)[0] + (ext or ".jsonl")
        if os.path.exists(p):
            os.remove(p)
    print(f"\nselftest: {'PASS' if fails == 0 else f'{fails} FAIL'}")
    return 1 if fails else 0


# =========================================================
# Main
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Data augmentation for continuous multi-day time ranges")
    ap.add_argument("--input", default=INPUT_JSONL)
    ap.add_argument("--output", default=OUTPUT_JSONL)
    ap.add_argument("--counts", help="JSON file {type: target count}")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=ENGINE.seed)
    ap.add_argument("--audit", choices=["strict", "report"], default=ENGINE.audit_mode)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    ENGINE.seed, ENGINE.audit_mode, ENGINE.write_meta = a.seed, a.audit, not a.no_meta
    counts = dict(AUGMENT_COUNTS)
    if a.counts:
        with open(a.counts, encoding="utf-8") as f:
            counts = {k: int(v) for k, v in json.load(f).items()}
    counts = {k: int(round(v * a.scale)) for k, v in counts.items()}
    if not os.path.exists(a.input):
        sys.exit(f"input file not found: {a.input}")
    records = load_jsonl(a.input)
    print(f"[INFO] loaded {len(records)} records from {a.input}")
    print(f"[INFO] spec: {SPEC}")
    aug = Augmenter(BUILD, LABEL_IO, SPEC, ENGINE)
    summary = aug.run(records, counts, a.output, audit_only=a.audit_only)
    print_report(summary, a.output, aug.audit_examples, a.audit_only)


if __name__ == "__main__":
    main()
