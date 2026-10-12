#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_date_augment.py - data augmentation for single-date samples 
================================================================================================

Input: annotated JSONL (same fields as the original data)
    id, sentence, time_expression_substring, time_expression_substring_index,
    time_expression_type (pattern_x_y_...), reference_year/month/day, reference_weekday,
    time_standard_format (YYYY-MM-DD), time_weekday
Output: JSONL with augmented samples only, same field layout as the input (weekday fields keep their
    original type, int or string), plus two debug fields aug_strategy / aug_source_id (off with --no-meta).
    Also writes <output>.report.json (per-type statistics) and <output>.audit.jsonl
    (source records whose label disagrees with the spec).



Usage
-----
  python single_date_augment.py --input data.jsonl --output augmented_data.jsonl
  python single_date_augment.py --input data.jsonl --audit-only          # audit sources only, no output
  python single_date_augment.py --selftest                               # regression tests
  python single_date_augment.py --input data.jsonl --counts counts.json  # custom per-type targets
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
    DATE_SEP_NUM, DATE_SEP_YMD, FLAGS, HOLIDAY, HOLIDAY_SURFACES, INT_WORD, LABEL_YEAR_MAX,
    LABEL_YEAR_MIN, MONTH_NAME, MONTH_TO_NUM, ORDINAL_ANY, ORDINAL_DIGIT, ORDINAL_WORD, WD_NAMES,
    WEEKDAY, YEAR, Augmenter, Candidate, Ctx, EngineConfig, LabelIO, Spec, TypeSpec, apply_case,
    combo, detect_polarity, first_business_day_in, group_edits, gval, holiday_key,
    infer_month_day, infer_pad_style, last_business_day_in, last_weekday_of_month,
    load_jsonl, month_range, month_shift, nth_weekday_of_month, number_unit_edits, parse_int_word,
    parse_iso, parse_ordinal, pick_by_polarity, print_report, quarter_range, render_day,
    render_month, render_nth, render_num, render_weekday, resolve_holiday, safe_date, sample_count,
    sample_month_day, sample_text_date, strat_ref_full, strat_ref_jitter, strat_ref_week_shift,
    strat_ref_year, text_candidate, weekday_like, weekday_str_to_int, month_str_to_int,
    yearly_candidates, year_range,
)

# =========================================================
# CONFIG (can also be overridden from the command line)
# =========================================================

INPUT_JSONL = "./data.jsonl"
OUTPUT_JSONL = "./augmented_data.jsonl"

AUGMENT_COUNTS: Dict[str, int] = {
    "pattern_1_1_iso_ymd": 5,
    "pattern_1_2_mdy": 5,
    "pattern_1_3_dmy": 5,
    "pattern_1_4_month_day_year": 5,
    "pattern_1_5_day_month_year": 5,
    "pattern_1_6_weekday_full_date": 5,
    "pattern_1_7_compact_ymd": 5,
    "pattern_1_9_day_monthname_year_sep": 5,
    "pattern_1_10_day_monthname_year_compact": 5,
    "pattern_1_11_year_monthname_day": 5,
    "pattern_1_12_year_numeric_month_day": 5,

    "pattern_2_1_month_day_no_year": 5,
    "pattern_2_2_day_month_no_year": 5,
    "pattern_2_3_day_of_month_no_year": 5,
    "pattern_2_4_numeric_md_dm_no_year": 5,
    "pattern_2_6_month_ordinalword_no_year": 5,
    "pattern_2_7_the_ordinalword_of_month_no_year": 5,
    "pattern_2_8_ordinalword_month_no_year": 5,
    "pattern_2_11_day_month_sep_no_year": 5,

    "pattern_3_1_relative_single_words": 5,
    "pattern_3_2_relative_two_day": 5,
    "pattern_3_3_offset_single_day": 5,
    "pattern_3_4_offset_weeks": 5,

    "pattern_4_1_rel_weekday_simple": 5,
    "pattern_4_2_rel_weekday_this_past_coming": 5,
    "pattern_4_3_on_weekday": 5,

    "pattern_5_1_rel_month_day": 5,
    "pattern_5_2_rel_day_month": 5,

    "pattern_6_1_first_last_day_of": 5,
    "pattern_6_2_holiday_single_day": 5,
    "pattern_6_3_first_last_business_day_of": 5,

    "pattern_7_1_nth_weekday_of_month": 5,
    "pattern_7_2_last_weekday_of_month": 5,
    "pattern_7_3_weekday_the_ordinal_day": 5,
}

SPEC = Spec()            # annotation conventions, see temporal_core.Spec
ENGINE = EngineConfig()  # engine settings, see temporal_core.EngineConfig

# =========================================================
# Patterns (same as the original script; only WEEKDAY/HOLIDAY are supersets)
# =========================================================

REL_WORD = r"(?:this|next|last|past|coming|upcoming|previous|prev)"
DAY_OF_TARGET = (rf"(?:{MONTH_NAME}|this month|next month|last month|"
                 r"this quarter|next quarter|last quarter|q[1-4]|"
                 r"this year|next year|last year|the month|the quarter|the year)")
WORKDAY_WORD = r"(?:business day|working day|workday)"

P = {
    "pattern_1_1_iso_ymd": rf"\b(?P<year>{YEAR}){DATE_SEP_YMD}(?P<month>0[1-9]|1[0-2]){DATE_SEP_YMD}(?P<day>0[1-9]|[12]\d|3[01])\b",
    "pattern_1_2_mdy": rf"\b(?P<month>0?[1-9]|1[0-2]){DATE_SEP_NUM}(?P<day>0?[1-9]|[12]\d|3[01]){DATE_SEP_NUM}(?P<year>{YEAR})\b",
    "pattern_1_3_dmy": rf"\b(?P<day>0?[1-9]|[12]\d|3[01]){DATE_SEP_NUM}(?P<month>0?[1-9]|1[0-2]){DATE_SEP_NUM}(?P<year>{YEAR})\b",
    "pattern_1_4_month_day_year": rf"\b(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})\s*,?\s*(?P<year>{YEAR})\b",
    "pattern_1_5_day_month_year": rf"\b(?P<day>{ORDINAL_ANY})\s+(?P<month_name>{MONTH_NAME})\s+(?P<year>{YEAR})\b",
    "pattern_1_6_weekday_full_date": (
        rf"\b(?P<weekday>{WEEKDAY})\s*,?\s+(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})\s*,?\s*(?P<year>{YEAR})\b"
        rf"|\b(?P<weekday2>{WEEKDAY})\s*,?\s+(?P<day2>{ORDINAL_ANY})\s+(?P<month_name2>{MONTH_NAME})\s*,?\s*(?P<year2>{YEAR})\b"),
    "pattern_1_7_compact_ymd": rf"\b(?P<year>{YEAR})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])\b",
    "pattern_1_9_day_monthname_year_sep": rf"\b(?P<day>{ORDINAL_DIGIT}){DATE_SEP_YMD}(?P<month_name>{MONTH_NAME}){DATE_SEP_YMD}(?P<year>{YEAR})\b",
    "pattern_1_10_day_monthname_year_compact": rf"\b(?P<day>{ORDINAL_DIGIT})(?P<month_name>{MONTH_NAME})(?P<year>{YEAR})\b",
    "pattern_1_11_year_monthname_day": rf"\b(?P<year>{YEAR})\s+(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})\b",
    "pattern_1_12_year_numeric_month_day": rf"\b(?P<year>{YEAR})\s+(?P<month>0?[1-9]|1[0-2])\s+(?P<day>0?[1-9]|[12]\d|3[01])\b",

    "pattern_2_1_month_day_no_year": rf"\b(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})\b(?!\s*{YEAR})",
    "pattern_2_2_day_month_no_year": rf"\b(?P<day>{ORDINAL_ANY})\s+(?P<month_name>{MONTH_NAME})\b(?!\s*{YEAR})",
    "pattern_2_3_day_of_month_no_year": rf"\b(?P<day>{ORDINAL_ANY})\s+of\s+(?P<month_name>{MONTH_NAME})\b(?!\s*{YEAR})",
    "pattern_2_4_numeric_md_dm_no_year": (
        rf"\b(?P<month1>0?[1-9]|1[0-2]){DATE_SEP_NUM}(?P<day1>0?[1-9]|[12]\d|3[01])\b(?![-/]\d)"
        rf"|\b(?P<day2>0?[1-9]|[12]\d|3[01]){DATE_SEP_NUM}(?P<month2>0?[1-9]|1[0-2])\b(?![-/]\d)"),
    "pattern_2_6_month_ordinalword_no_year": rf"\b(?P<month_name>{MONTH_NAME})\s+(?:the\s+)?(?P<ord_word>{ORDINAL_WORD})\b(?!\s*{YEAR})",
    "pattern_2_7_the_ordinalword_of_month_no_year": rf"\bthe\s+(?P<ord_word>{ORDINAL_WORD})\s+of\s+(?P<month_name>{MONTH_NAME})\b(?!\s*{YEAR})",
    "pattern_2_8_ordinalword_month_no_year": rf"\b(?:the\s+)?(?P<ord_word>{ORDINAL_WORD})\s*,?\s+(?P<month_name>{MONTH_NAME})\b(?!\s*{YEAR})",
    "pattern_2_11_day_month_sep_no_year": rf"\b(?P<day>{ORDINAL_DIGIT}){DATE_SEP_YMD}(?P<month_name>{MONTH_NAME})\b(?!{DATE_SEP_YMD}\d)",

    "pattern_3_1_relative_single_words": (
        r"\b(?P<word>(today|yesterday|tomorrow|tonight|this morning|this afternoon|this evening|"
        r"this night|last night|noon|midday|midnight))\b"),
    "pattern_3_2_relative_two_day": r"\b(?P<phrase>the day (?:before yesterday|after tomorrow))\b",
    "pattern_3_3_offset_single_day": (
        rf"\b(?P<num>{INT_WORD})\s+days?\s+ago\b"
        rf"|\b(?:in|after)\s+(?P<num2>{INT_WORD})\s+days?\b"
        rf"|\b(?P<num3>{INT_WORD})\s+days?\s+from\s+now\b"
        rf"|\b(?P<num4>{INT_WORD})\s+days?\s+later\b"
        rf"|\b(?P<num5>{INT_WORD})\s+days?\s+(?:before|prior)\b"),
    "pattern_3_4_offset_weeks": (
        rf"\b(?P<num_ago>{INT_WORD})\s+weeks?\s+ago\b"
        rf"|\b(?P<num_from_now>{INT_WORD})\s+weeks?\s+from\s+now\b"
        rf"|\b(?P<num_from_today>{INT_WORD})\s+weeks?\s+from\s+today\b"
        rf"|\b(?P<num_from_yesterday>{INT_WORD})\s+weeks?\s+from\s+yesterday\b"
        rf"|\b(?P<num_from_tomorrow>{INT_WORD})\s+weeks?\s+from\s+tomorrow\b"
        rf"|\bin\s+(?P<num_in>{INT_WORD})\s+weeks?\b"
        rf"|\ba\s+week\s+from\s+now\b"),

    "pattern_4_1_rel_weekday_simple": rf"\b(?P<rel>{REL_WORD})\s+(?P<weekday>{WEEKDAY})\b",
    "pattern_4_2_rel_weekday_this_past_coming": rf"\bthis\s+(?P<rel2>past|coming)\s+(?P<weekday2>{WEEKDAY})\b",
    "pattern_4_3_on_weekday": rf"\bon\s+(?P<weekday>{WEEKDAY})\b",

    "pattern_5_1_rel_month_day": rf"\b(?P<rel>{REL_WORD})\s+(?P<month_name>{MONTH_NAME})\s+(?P<day>{ORDINAL_ANY})\b(?!\s*{YEAR})",
    "pattern_5_2_rel_day_month": rf"\b(?P<rel>{REL_WORD})\s+(?P<day>{ORDINAL_ANY})\s+(?P<month_name>{MONTH_NAME})\b(?!\s*{YEAR})",

    "pattern_6_1_first_last_day_of": rf"\b(?:the\s+)?(?P<which>first|1st|last|final)\s+day of\s+(?P<target>{DAY_OF_TARGET})\b",
    "pattern_6_2_holiday_single_day": rf"\b(?P<holiday>{HOLIDAY})\b",
    "pattern_6_3_first_last_business_day_of": (
        rf"\b(?:the\s+)?(?P<which>first|1st|last|final)\s+(?P<kind>{WORKDAY_WORD})s?\s+of\s+(?P<target>{DAY_OF_TARGET})\b"),

    "pattern_7_1_nth_weekday_of_month": (
        rf"\b(?:the\s+)?(?P<nth>(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th))\s+"
        rf"(?P<weekday>{WEEKDAY})\s+(?:of|in)\s+(?P<month_name>{MONTH_NAME})\b"),
    "pattern_7_2_last_weekday_of_month": rf"\b(?:the\s+)?(?P<which>last|final)\s+(?P<weekday>{WEEKDAY})\s+(?:of|in)\s+(?P<month_name>{MONTH_NAME})\b",
    "pattern_7_3_weekday_the_ordinal_day": rf"\b(?P<weekday>{WEEKDAY})\s+the\s+(?P<day>{ORDINAL_ANY})\b",
}
REGEX = {k: re.compile(v, FLAGS) for k, v in P.items()}


def pol(m: re.Match, sentence: str) -> str:
    return detect_polarity(sentence, *m.span())


# =========================================================
# Resolvers  (m, ref, sentence, spec) -> date
# =========================================================

def r_ymd(m, ref, s, spec):
    return date(int(m.group("year")), int(m.group("month")), int(m.group("day")))


def r_name_ymd(m, ref, s, spec):
    if gval(m, "year2"):
        return date(int(m.group("year2")), month_str_to_int(m.group("month_name2")), parse_ordinal(m.group("day2")))
    d = date(int(m.group("year")), month_str_to_int(m.group("month_name")), parse_ordinal(m.group("day")))
    return d


def r_weekday_full_date(m, ref, s, spec):
    d = r_name_ymd(m, ref, s, spec)
    wd_tok = gval(m, "weekday") or gval(m, "weekday2")
    if wd_tok and weekday_str_to_int(wd_tok) != d.weekday():
        raise ValueError("weekday does not match date")  # weekday contradicts the date in the sentence -> unusable
    return d


def r_md_noyear(m, ref, s, spec):
    mm = month_str_to_int(m.group("month_name"))
    dd = parse_ordinal(gval(m, "day") or m.group("ord_word"))
    return infer_month_day(ref, mm, dd, pol(m, s), spec)


def r_numeric_md(m, ref, s, spec):
    if gval(m, "month1"):
        mm, dd = int(m.group("month1")), int(m.group("day1"))
    else:
        mm, dd = int(m.group("month2")), int(m.group("day2"))
    return infer_month_day(ref, mm, dd, pol(m, s), spec)


def r_rel_words(m, ref, s, spec):
    w = re.sub(r"\s+", " ", m.group("word").lower())
    if w in ("yesterday", "last night"):
        return ref - timedelta(days=1)
    if w == "tomorrow":
        return ref + timedelta(days=1)
    return ref


def r_two_day(m, ref, s, spec):
    p = m.group("phrase").lower()
    return ref - timedelta(days=2) if "before" in p else ref + timedelta(days=2)


def f_event_relative(m, s, spec):
    if not spec.allow_event_relative_offsets and (gval(m, "num4") or gval(m, "num5")):
        return "event_relative_offset"
    return None


def r_offset_days(m, ref, s, spec):
    for g, sign in (("num", -1), ("num2", 1), ("num3", 1), ("num4", 1), ("num5", -1)):
        v = gval(m, g)
        if v:
            return ref + timedelta(days=sign * parse_int_word(v))
    raise ValueError("no number")


def r_offset_weeks(m, ref, s, spec):
    if re.fullmatch(r"a\s+week\s+from\s+now", m.group(0), re.I):
        return ref + timedelta(weeks=1)
    table = (("num_ago", ref, -1), ("num_from_now", ref, 1), ("num_from_today", ref, 1), ("num_in", ref, 1),
             ("num_from_yesterday", ref - timedelta(days=1), 1), ("num_from_tomorrow", ref + timedelta(days=1), 1))
    for g, base, sign in table:
        v = gval(m, g)
        if v:
            return base + timedelta(weeks=sign * parse_int_word(v))
    raise ValueError("no week offset")


def rel_weekday(rel: str, wd: int, ref: date, spec: Spec) -> date:
    rel = rel.lower()
    today = ref.weekday()
    week_mon = ref - timedelta(days=(today - spec.week_start) % 7)
    pos = (wd - spec.week_start) % 7
    after = (wd - today) % 7 or 7     # nearest strictly after
    before = (today - wd) % 7 or 7    # nearest strictly before
    if rel == "this":
        if spec.this_weekday == "upcoming":
            return ref + timedelta(days=(wd - today) % 7)
        return week_mon + timedelta(days=pos)
    if rel in ("coming", "upcoming"):
        return ref + timedelta(days=after)
    if rel == "next":
        if spec.next_weekday == "next_week":
            return week_mon + timedelta(days=7 + pos)  # fix: no longer adds 7 on top of the modulo
        return ref + timedelta(days=after)
    if rel == "past":
        return ref - timedelta(days=before)
    if rel in ("last", "previous", "prev"):
        if spec.last_weekday == "last_week":
            return week_mon - timedelta(days=7 - pos)
        return ref - timedelta(days=before)
    raise ValueError(rel)


def r_rel_weekday(m, ref, s, spec):
    return rel_weekday(m.group("rel"), weekday_str_to_int(m.group("weekday")), ref, spec)


def r_this_past_coming(m, ref, s, spec):
    rel = "past" if m.group("rel2").lower() == "past" else "coming"
    return rel_weekday(rel, weekday_str_to_int(m.group("weekday2")), ref, spec)


def r_on_weekday(m, ref, s, spec):
    wd = weekday_str_to_int(m.group("weekday"))
    p = pol(m, s)
    if p == "neutral":
        p = spec.on_weekday_neutral
    after = (wd - ref.weekday()) % 7 or 7
    before = (ref.weekday() - wd) % 7 or 7
    if p == "future":
        return ref + timedelta(days=after)
    if p == "past":
        return ref - timedelta(days=before)
    return ref + timedelta(days=after) if after <= before else ref - timedelta(days=before)


def r_rel_month_day(m, ref, s, spec):
    rel = m.group("rel").lower()
    mm, dd = month_str_to_int(m.group("month_name")), parse_ordinal(m.group("day"))
    cands = [d for d in (safe_date(y, mm, dd) for y in range(ref.year - 8, ref.year + 9)) if d]
    if not cands:
        raise ValueError("bad date")
    if rel == "this":
        same = [d for d in cands if d.year == ref.year]
        if not same:
            raise ValueError("no such date this year")
        return same[0]
    if rel in ("next", "coming", "upcoming"):
        return min(d for d in cands if d > ref)
    return max(d for d in cands if d < ref)


# ---- 6.1 / 6.3: first/last (business) day of <target> ----

def _target_periods(target: str, ref: date) -> Tuple[List[Tuple[date, date]], bool]:
    """Return (candidate periods, whether the year must be chosen by tense)."""
    t = re.sub(r"\s+", " ", target.lower().strip())
    first_tok = t.split(" ")[0].rstrip(".")
    if first_tok in MONTH_TO_NUM:
        mm = MONTH_TO_NUM[first_tok]
        return [month_range(y, mm) for y in range(ref.year - 1, ref.year + 2)], True
    qm = re.fullmatch(r"q([1-4])", t)
    if qm:
        q = int(qm.group(1))
        return [quarter_range(y, q) for y in range(ref.year - 1, ref.year + 2)], True
    cur_q = (ref.month - 1) // 3 + 1
    if t in ("this month", "the month"):
        return [month_range(ref.year, ref.month)], False
    if t in ("next month", "last month"):
        y, mm = month_shift(ref.year, ref.month, 1 if t.startswith("next") else -1)
        return [month_range(y, mm)], False
    if t in ("this year", "the year"):
        return [year_range(ref.year)], False
    if t in ("next year", "last year"):
        return [year_range(ref.year + (1 if t.startswith("next") else -1))], False
    if t in ("this quarter", "the quarter"):
        return [quarter_range(ref.year, cur_q)], False
    if t in ("next quarter", "last quarter"):
        idx = ref.year * 4 + (cur_q - 1) + (1 if t.startswith("next") else -1)
        return [quarter_range(idx // 4, idx % 4 + 1)], False
    raise ValueError(f"unknown target {target!r}")


def _is_first(which: str) -> bool:
    return which.lower() in ("first", "1st")


def r_first_last_day_of(m, ref, s, spec):
    periods, choose = _target_periods(m.group("target"), ref)
    first = _is_first(m.group("which"))
    days = [a if first else b for a, b in periods]
    return pick_by_polarity(days, ref, pol(m, s), spec) if choose else days[0]


def r_first_last_business_day_of(m, ref, s, spec):
    periods, choose = _target_periods(m.group("target"), ref)
    first = _is_first(m.group("which"))
    days = [(first_business_day_in if first else last_business_day_in)(a, b, spec) for a, b in periods]
    days = [d for d in days if d]
    if not days:
        raise ValueError("no business day")
    return pick_by_polarity(days, ref, pol(m, s), spec) if choose else days[0]


def r_holiday(m, ref, s, spec):
    return resolve_holiday(m.group("holiday"), ref, pol(m, s), spec)


_NTH = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}


def r_nth_weekday(m, ref, s, spec):
    n = _NTH[m.group("nth").lower()]
    wd, mm = weekday_str_to_int(m.group("weekday")), month_str_to_int(m.group("month_name"))
    # A 5th weekday does not exist every year; search only +-1 year and drop the sample otherwise (avoids labels 2-3 years away)
    d = pick_by_polarity(yearly_candidates(ref, lambda y: nth_weekday_of_month(y, mm, wd, n), 1), ref, pol(m, s), spec)
    if d is None:
        raise ValueError("no such nth weekday")
    return d


def r_last_weekday(m, ref, s, spec):
    wd, mm = weekday_str_to_int(m.group("weekday")), month_str_to_int(m.group("month_name"))
    return pick_by_polarity(yearly_candidates(ref, lambda y: last_weekday_of_month(y, mm, wd)), ref, pol(m, s), spec)


def r_weekday_the_day(m, ref, s, spec):
    """'Friday the 13th': find every date within +-6 years matching (weekday, day) and pick the nearest by tense."""
    wd, dd = weekday_str_to_int(m.group("weekday")), parse_ordinal(m.group("day"))
    if not 1 <= dd <= 31:
        raise ValueError("bad day")
    cands = []
    for y in range(ref.year - 6, ref.year + 7):
        for mo in range(1, 13):
            d = safe_date(y, mo, dd)
            if d and d.weekday() == wd:
                cands.append(d)
    d = pick_by_polarity(cands, ref, pol(m, s), spec)
    if d is None:
        raise ValueError("no match")
    return d


# =========================================================
# Text rewriting strategies
# =========================================================

def make_text_full_date(kinds: Dict[str, str], force_pad: bool = False):
    """1.x: replace the year/month/day/weekday tokens inside the span with a new date, keeping the format."""
    numeric = [g for g, k in kinds.items() if k in ("month_num", "day")]

    def fn(ctx: Ctx) -> Optional[Candidate]:
        m = ctx.m
        pad = infer_pad_style(m, numeric, ctx.rng, force_pad)
        new = sample_text_date(ctx)
        repl: Dict[str, str] = {}
        for g, k in kinds.items():
            v = gval(m, g)
            if v is None:
                continue
            if k == "year":
                repl[g] = f"{new.year:04d}"
            elif k == "month_num":
                repl[g] = render_num(v, new.month, pad)
            elif k == "day":
                repl[g] = render_day(v, new.day, pad if v[:1].isdigit() else None)
            elif k == "month_name":
                repl[g] = render_month(v, new.month)
            elif k == "weekday":
                repl[g] = render_weekday(v, new.weekday())
        return text_candidate(ctx, group_edits(m, repl), "text_date")
    return fn


def make_text_md(kinds: Dict[str, str]):
    """2.x / 5.x: change month and day in the sentence (no year). Ambiguous numeric D/M forms force day > 12."""
    numeric = [g for g, k in kinds.items() if k in ("month_num", "day")]

    def fn(ctx: Ctx) -> Optional[Candidate]:
        m = ctx.m
        dm_alt = gval(m, "day2") is not None and gval(m, "month2") is not None
        mo, dd = sample_month_day(ctx.rng, allow_feb29=False, min_day=13 if dm_alt else 1)
        pad = infer_pad_style(m, numeric, ctx.rng)
        repl: Dict[str, str] = {}
        for g, k in kinds.items():
            v = gval(m, g)
            if v is None:
                continue
            if k == "month_num":
                repl[g] = render_num(v, mo, pad)
            elif k == "day":
                repl[g] = render_day(v, dd, pad if v[:1].isdigit() else None)
            elif k == "month_name":
                repl[g] = render_month(v, mo)
        return text_candidate(ctx, group_edits(m, repl), "text_md")
    return fn


def make_text_number(num_groups: Dict[str, str]):
    """3.3 / 3.4: change the count (digits or words, as in the original) and keep singular/plural in sync."""
    def fn(ctx: Ctx) -> Optional[Candidate]:
        for g, unit in num_groups.items():
            v = gval(ctx.m, g)
            if v is None:
                continue
            old = parse_int_word(v)
            n = sample_count(ctx.rng, unit, not v.isdigit(), old)
            return text_candidate(ctx, number_unit_edits(ctx.sentence, ctx.m, g, n), "text_number")
        return None
    return fn


def make_text_weekday(groups: List[str]):
    def fn(ctx: Ctx) -> Optional[Candidate]:
        for g in groups:
            v = gval(ctx.m, g)
            if v is None:
                continue
            old = weekday_str_to_int(v)
            new = ctx.rng.choice([w for w in range(7) if w != old])
            return text_candidate(ctx, group_edits(ctx.m, {g: render_weekday(v, new)}), "text_weekday")
        return None
    return fn


def text_holiday(ctx: Ctx) -> Optional[Candidate]:
    v = ctx.m.group("holiday")
    old = holiday_key(v)
    choices = [h for h in HOLIDAY_SURFACES if holiday_key(h) != old]
    new = ctx.rng.choice(choices)
    if "’" in v:
        new = new.replace("'", "’")
    if v.islower():
        new = new.lower()
    elif v.isupper():
        new = new.upper()
    return text_candidate(ctx, group_edits(ctx.m, {"holiday": new}), "text_holiday")


_WHICH_SWAP = {"first": "last", "1st": "last", "last": "first", "final": "first"}


def text_day_of_target(ctx: Ctx) -> Optional[Candidate]:
    m, rng = ctx.m, ctx.rng
    repl: Dict[str, str] = {}
    if rng.random() < 0.5:
        w = m.group("which")
        repl["which"] = apply_case(w, _WHICH_SWAP[w.lower()]) if not w[:1].isdigit() else _WHICH_SWAP[w.lower()]
    tgt = m.group("target")
    tok = tgt.split()[0]
    if tok.lower().rstrip(".") in MONTH_TO_NUM:
        old = month_str_to_int(tok)
        repl["target"] = render_month(tgt, rng.choice([x for x in range(1, 13) if x != old]))
    elif re.fullmatch(r"q[1-4]", tgt, re.I):
        q = rng.choice([x for x in range(1, 5) if x != int(tgt[1])])
        repl["target"] = f"{tgt[0]}{q}"
    if not repl:
        return None
    return text_candidate(ctx, group_edits(m, repl), "text_target")


def text_nth_weekday(ctx: Ctx) -> Optional[Candidate]:
    m, rng = ctx.m, ctx.rng
    repl: Dict[str, str] = {}
    while not repl:
        if rng.random() < 0.5:
            old = _NTH[m.group("nth").lower()]
            n = rng.choice([x for x in (1, 1, 2, 2, 3, 3, 4, 4, 5) if x != old])
            repl["nth"] = render_nth(m.group("nth"), n)
        if rng.random() < 0.5:
            old = weekday_str_to_int(m.group("weekday"))
            repl["weekday"] = render_weekday(m.group("weekday"), rng.choice([w for w in range(7) if w != old]))
        if rng.random() < 0.5:
            old = month_str_to_int(m.group("month_name"))
            repl["month_name"] = render_month(m.group("month_name"), rng.choice([x for x in range(1, 13) if x != old]))
    return text_candidate(ctx, group_edits(m, repl), "text_nth_weekday")


def text_last_weekday(ctx: Ctx) -> Optional[Candidate]:
    m, rng = ctx.m, ctx.rng
    repl: Dict[str, str] = {}
    while not repl:
        if rng.random() < 0.5:
            old = weekday_str_to_int(m.group("weekday"))
            repl["weekday"] = render_weekday(m.group("weekday"), rng.choice([w for w in range(7) if w != old]))
        if rng.random() < 0.5:
            old = month_str_to_int(m.group("month_name"))
            repl["month_name"] = render_month(m.group("month_name"), rng.choice([x for x in range(1, 13) if x != old]))
    return text_candidate(ctx, group_edits(m, repl), "text_last_weekday")


def text_weekday_the_day(ctx: Ctx) -> Optional[Candidate]:
    m, rng = ctx.m, ctx.rng
    repl: Dict[str, str] = {}
    while not repl:
        if rng.random() < 0.5:
            old = weekday_str_to_int(m.group("weekday"))
            repl["weekday"] = render_weekday(m.group("weekday"), rng.choice([w for w in range(7) if w != old]))
        if rng.random() < 0.6:
            old = parse_ordinal(m.group("day"))
            repl["day"] = render_day(m.group("day"), rng.choice([d for d in range(1, 29) if d != old]))
    return text_candidate(ctx, group_edits(m, repl), "text_weekday_day")


# =========================================================
# Type table: regex + resolver + strategy weights
# =========================================================

def _w(*items):
    return [(n, w, f) for n, w, f in items]


REF_MIX = _w(("ref_full", 0.3, strat_ref_full), ("ref_jitter", 0.2, strat_ref_jitter), ("ref_year", 0.1, strat_ref_year))


def with_text(text_fn, name: str, w_text: float = 0.2, w_combo: float = 0.2):
    return REF_MIX + _w((name, w_text, text_fn), (name + "+ref_full", w_combo, combo(text_fn, strat_ref_full, name + "+ref_full")))


def text_and_ref(fn):
    """1.x: change the date in the sentence; half the samples also change the reference date (explicit labels do not depend on it, but the year stays consistent with tense)."""
    return _w(("text_date", 0.5, fn), ("text_date+ref_full", 0.5, combo(fn, strat_ref_full, "text_date+ref_full")))


def build_types() -> Dict[str, TypeSpec]:
    T: Dict[str, TypeSpec] = {}

    def add(name, resolve, strategies, source_filter=None):
        T[name] = TypeSpec(name, REGEX[name], resolve, strategies, source_filter)

    ymd_num = {"year": "year", "month": "month_num", "day": "day"}
    ymd_name = {"year": "year", "month_name": "month_name", "day": "day"}
    add("pattern_1_1_iso_ymd", r_ymd, text_and_ref(make_text_full_date(ymd_num, force_pad=True)))
    add("pattern_1_2_mdy", r_ymd, text_and_ref(make_text_full_date(ymd_num)))
    add("pattern_1_3_dmy", r_ymd, text_and_ref(make_text_full_date(ymd_num)))
    add("pattern_1_4_month_day_year", r_name_ymd, text_and_ref(make_text_full_date(ymd_name)))
    add("pattern_1_5_day_month_year", r_name_ymd, text_and_ref(make_text_full_date(ymd_name)))
    add("pattern_1_6_weekday_full_date", r_weekday_full_date, text_and_ref(make_text_full_date({
        **ymd_name, "weekday": "weekday", "year2": "year", "month_name2": "month_name", "day2": "day", "weekday2": "weekday"})))
    add("pattern_1_7_compact_ymd", r_ymd, text_and_ref(make_text_full_date(ymd_num, force_pad=True)))
    add("pattern_1_9_day_monthname_year_sep", r_name_ymd, text_and_ref(make_text_full_date(ymd_name)))
    add("pattern_1_10_day_monthname_year_compact", r_name_ymd, text_and_ref(make_text_full_date(ymd_name)))
    add("pattern_1_11_year_monthname_day", r_name_ymd, text_and_ref(make_text_full_date(ymd_name)))
    add("pattern_1_12_year_numeric_month_day", r_ymd, text_and_ref(make_text_full_date(ymd_num)))

    md_name = {"month_name": "month_name", "day": "day"}
    md_word = {"month_name": "month_name", "ord_word": "day"}
    add("pattern_2_1_month_day_no_year", r_md_noyear, with_text(make_text_md(md_name), "text_md"))
    add("pattern_2_2_day_month_no_year", r_md_noyear, with_text(make_text_md(md_name), "text_md"))
    add("pattern_2_3_day_of_month_no_year", r_md_noyear, with_text(make_text_md(md_name), "text_md"))
    add("pattern_2_4_numeric_md_dm_no_year", r_numeric_md, with_text(make_text_md(
        {"month1": "month_num", "day1": "day", "day2": "day", "month2": "month_num"}), "text_md"))
    add("pattern_2_6_month_ordinalword_no_year", r_md_noyear, with_text(make_text_md(md_word), "text_md"))
    add("pattern_2_7_the_ordinalword_of_month_no_year", r_md_noyear, with_text(make_text_md(md_word), "text_md"))
    add("pattern_2_8_ordinalword_month_no_year", r_md_noyear, with_text(make_text_md(md_word), "text_md"))
    add("pattern_2_11_day_month_sep_no_year", r_md_noyear, with_text(make_text_md(md_name), "text_md"))

    ref_only = _w(("ref_full", 0.7, strat_ref_full), ("ref_jitter", 0.3, strat_ref_jitter))
    add("pattern_3_1_relative_single_words", r_rel_words, ref_only)
    add("pattern_3_2_relative_two_day", r_two_day, ref_only)
    num_days = make_text_number({"num": "day", "num2": "day", "num3": "day", "num4": "day", "num5": "day"})
    num_weeks = make_text_number({g: "week" for g in ("num_ago", "num_from_now", "num_from_today",
                                                       "num_from_yesterday", "num_from_tomorrow", "num_in")})
    num_mix = lambda f: _w(("ref_full", 0.35, strat_ref_full), ("ref_jitter", 0.15, strat_ref_jitter),
                           ("text_number", 0.25, f), ("text_number+ref_full", 0.25, combo(f, strat_ref_full, "text_number+ref_full")))
    add("pattern_3_3_offset_single_day", r_offset_days, num_mix(num_days), f_event_relative)
    add("pattern_3_4_offset_weeks", r_offset_weeks, num_mix(num_weeks))

    def wd_mix(groups):
        f = make_text_weekday(groups)
        return _w(("ref_week_shift", 0.25, strat_ref_week_shift), ("ref_full", 0.3, strat_ref_full),
                  ("ref_jitter", 0.15, strat_ref_jitter), ("text_weekday", 0.15, f),
                  ("text_weekday+ref_full", 0.15, combo(f, strat_ref_full, "text_weekday+ref_full")))
    add("pattern_4_1_rel_weekday_simple", r_rel_weekday, wd_mix(["weekday"]))
    add("pattern_4_2_rel_weekday_this_past_coming", r_this_past_coming, wd_mix(["weekday2"]))
    add("pattern_4_3_on_weekday", r_on_weekday, wd_mix(["weekday"]))

    add("pattern_5_1_rel_month_day", r_rel_month_day, with_text(make_text_md(md_name), "text_md"))
    add("pattern_5_2_rel_day_month", r_rel_month_day, with_text(make_text_md(md_name), "text_md"))

    add("pattern_6_1_first_last_day_of", r_first_last_day_of, with_text(text_day_of_target, "text_target"))
    add("pattern_6_2_holiday_single_day", r_holiday, with_text(text_holiday, "text_holiday"))
    add("pattern_6_3_first_last_business_day_of", r_first_last_business_day_of, with_text(text_day_of_target, "text_target"))

    add("pattern_7_1_nth_weekday_of_month", r_nth_weekday, with_text(text_nth_weekday, "text_nth_weekday"))
    add("pattern_7_2_last_weekday_of_month", r_last_weekday, with_text(text_last_weekday, "text_last_weekday"))
    add("pattern_7_3_weekday_the_ordinal_day", r_weekday_the_day, with_text(text_weekday_the_day, "text_weekday_day"))
    return T


# =========================================================
# Label I/O
# =========================================================

def _read(rec):
    return parse_iso(rec.get("time_standard_format"))


def _write(out, d: date, src):
    out["time_standard_format"] = d.isoformat()
    out["time_weekday"] = weekday_like(src.get("time_weekday", "Monday"), d.weekday())


def _validate(d, ts):
    if not isinstance(d, date):
        return "label_not_date"
    if not (LABEL_YEAR_MIN <= d.year <= LABEL_YEAR_MAX):
        return "label_out_of_range"
    return None


LABEL_IO = LabelIO(read=_read, write=_write, validate=_validate, key=lambda d: d.isoformat())


# =========================================================
# Self-test (covers every bug from the earlier review)
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
        rx = REGEX[t]
        m = rx.search(sentence)
        assert m, (t, sentence)
        return BUILD[t].resolve(m, ref, sentence, sp)

    print("== resolver regressions ==")
    sat = date(2025, 11, 1)   # Saturday
    wed = date(2025, 10, 29)  # Wednesday
    check("Sat next Monday (old: 11-10)", res("pattern_4_1_rel_weekday_simple", "See you next Monday.", sat), date(2025, 11, 3))
    check("Wed next Friday", res("pattern_4_1_rel_weekday_simple", "See you next Friday.", wed), date(2025, 11, 7))
    check("Wed next Wednesday", res("pattern_4_1_rel_weekday_simple", "See you next Wednesday.", wed), date(2025, 11, 5))
    check("Sat last Friday (nearest)", res("pattern_4_1_rel_weekday_simple", "I saw him last Friday.", sat), date(2025, 10, 31))
    check("Sat last Friday (last_week)", res("pattern_4_1_rel_weekday_simple", "I saw him last Friday.", sat, Spec(last_weekday="last_week")), date(2025, 10, 24))
    check("this past Friday", res("pattern_4_2_rel_weekday_this_past_coming", "We met this past Friday.", wed), date(2025, 10, 24))
    check("on Friday + past tense", res("pattern_4_3_on_weekday", "We met on Friday.", wed), date(2025, 10, 24))
    check("on Friday + future tense", res("pattern_4_3_on_weekday", "We will meet on Friday.", wed), date(2025, 10, 31))
    for h, exp in (("Valentine's Day", date(2026, 2, 14)), ("Independence Day", date(2026, 7, 4)),
                   ("Veterans Day", date(2025, 11, 11)), ("April Fools' Day", date(2026, 4, 1)),
                   ("New Year's Eve", date(2025, 12, 31)), ("Mother’s Day", date(2026, 5, 10))):
        check(f"holiday {h}", res("pattern_6_2_holiday_single_day", f"We will celebrate {h} together.", sat), exp)
    check("Thanksgiving past tense", res("pattern_6_2_holiday_single_day", "We had dinner on Thanksgiving.", sat), date(2024, 11, 28))
    check("no year, neutral -> nearest (Jan 3 said on 12-28)", res("pattern_2_1_month_day_no_year", "The report covers Jan 3 results.", date(2025, 12, 28)), date(2026, 1, 3))
    check("no year, past_bias", res("pattern_2_1_month_day_no_year", "The report covers Jan 3 results.", date(2025, 12, 28), Spec(neutral_year="past_bias")), date(2025, 1, 3))
    check("Friday the 13th future tense", res("pattern_7_3_weekday_the_ordinal_day", "The party will be on Friday the 13th.", sat), date(2026, 2, 13))
    check("Friday the 13th past tense", res("pattern_7_3_weekday_the_ordinal_day", "It was Friday the 13th.", sat), date(2025, 6, 13))
    check("last day of February (leap year)", res("pattern_6_1_first_last_day_of", "Pay by the last day of February.", date(2024, 1, 10)), date(2024, 2, 29))
    check("last business day of this year (2021-12-31 is observed holiday)", res("pattern_6_3_first_last_business_day_of", "Close on the last business day of this year.", date(2021, 6, 1)), date(2021, 12, 30))
    check("first business day of January 2022", res("pattern_6_3_first_last_business_day_of", "Start on the first business day of January.", date(2021, 12, 1)), date(2022, 1, 3))
    from temporal_core import is_business_day
    check("Juneteenth: 2020 is a business day / 2021-06-18 observed",
          (is_business_day(date(2020, 6, 19), spec), is_business_day(date(2021, 6, 18), spec)), (True, False))
    check("NYSE: closed Good Friday, open Columbus Day",
          (is_business_day(date(2025, 4, 18), Spec(business_calendar="nyse")), is_business_day(date(2025, 10, 13), Spec(business_calendar="nyse"))), (False, True))
    check("2nd Monday of October", res("pattern_7_1_nth_weekday_of_month", "It is the second Monday of October.", date(2025, 9, 1)), date(2025, 10, 13))
    check("next May 5", res("pattern_5_1_rel_month_day", "Next May 5 is open.", date(2025, 5, 5)), date(2026, 5, 5))
    check("in three weeks", res("pattern_3_4_offset_weeks", "Ship it in three weeks.", sat), date(2025, 11, 22))

    print("== polarity ==")
    def P(sent, sub):
        i = sent.index(sub)
        return detect_polarity(sent, i, i + len(sub))
    check("will be closed -> future", P("We will be closed on Jan 2 for inventory.", "Jan 2"), "future")
    check("was after the span -> past", P("May 5 was a great day.", "May 5"), "past")
    check("due to is not a future cue", P("The delay is due to May 5 storms.", "May 5"), "neutral")
    check("Chad does not match had", P("Chad meets on May 5.", "May 5"), "neutral")
    check("nearest cue wins", P("It was announced that we will ship on May 5.", "May 5"), "future")

    print("== rendering ==")
    from temporal_core import render_unit
    check("compact zero padding", "".join([f"{2019:04d}", render_num("11", 3, True), render_num("15", 7, True)]), "20190307")
    check("2-digit template without leading zero", render_num("12", 3, None), "3")
    check("leading-zero template pads", render_num("05", 3, None), "03")
    check("singular/plural", render_unit("days", 1) + "/" + render_unit("Day", 4), "day/Days")
    check("month Sept.", render_month("Sept.", 9) + "|" + render_month("Sept.", 5) + "|" + render_month("Nov.", 3), "Sept.|May|Mar.")
    check("ordinal word case", render_day("Twenty-First", 2) + "|" + render_day("3RD", 22), "Second|22ND")
    check("weekday_like", [weekday_like(3, 0), weekday_like("friday", 0), weekday_like("FRI", 0)], [0, "monday", "MON"])

    print("== end-to-end round trip ==")
    recs = [
        {"id": "a", "sentence": "The file was created on 20231115 by the bot.", "time_expression_substring": "20231115",
         "time_expression_substring_index": [24, 32], "time_expression_type": "pattern_1_7_compact_ymd",
         "reference_year": 2025, "reference_month": 11, "reference_day": 1, "reference_weekday": "Saturday",
         "time_standard_format": "2023-11-15", "time_weekday": "Wednesday"},
        {"id": "b", "sentence": "He left three days ago.", "time_expression_substring": "three days ago",
         "time_expression_substring_index": [8, 22], "time_expression_type": "pattern_3_3_offset_single_day",
         "reference_year": 2025, "reference_month": 11, "reference_day": 1, "reference_weekday": 5,
         "time_standard_format": "2025-10-29", "time_weekday": 2},
    ]
    with_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest_single.jsonl")
    aug = Augmenter(BUILD, LABEL_IO, spec, EngineConfig(seed=1))
    aug.run(recs, {"pattern_1_7_compact_ymd": 30, "pattern_3_3_offset_single_day": 30}, with_tmp)
    outs = load_jsonl(with_tmp)
    bad = 0
    for o in outs:
        t = o["time_expression_type"]
        m = REGEX[t].fullmatch(o["time_expression_substring"])
        ref = date(o["reference_year"], o["reference_month"], o["reference_day"])
        lab = BUILD[t].resolve(REGEX[t].match(o["sentence"], o["time_expression_substring_index"][0]), ref, o["sentence"], spec)
        if not m or lab.isoformat() != o["time_standard_format"] or o["sentence"][slice(*o["time_expression_substring_index"])] != o["time_expression_substring"]:
            bad += 1
        if t == "pattern_1_7_compact_ymd" and not re.fullmatch(r"\d{8}", o["time_expression_substring"]):
            bad += 1
        if t == "pattern_3_3_offset_single_day":
            nm = re.match(r"(\w+)\s+(days?)", o["time_expression_substring"])
            if (parse_int_word(nm.group(1)) == 1) != (nm.group(2) == "day"):
                bad += 1
        if not isinstance(o["time_weekday"], type(recs[0 if t.startswith("pattern_1") else 1]["time_weekday"])):
            bad += 1
    check(f"generated {len(outs)} records, all pass round-trip/format/type checks", bad, 0)
    os.remove(with_tmp)
    for ext in (".report.json", ".audit.jsonl"):
        p = os.path.splitext(with_tmp)[0] + ext
        if os.path.exists(p):
            os.remove(p)
    print(f"\nselftest: {'PASS' if fails == 0 else f'{fails} FAIL'}")
    return 1 if fails else 0


BUILD = build_types()


# =========================================================
# Main
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Data augmentation for single-date time expressions")
    ap.add_argument("--input", default=INPUT_JSONL)
    ap.add_argument("--output", default=OUTPUT_JSONL)
    ap.add_argument("--counts", help="JSON file {type: target count}; overrides AUGMENT_COUNTS in this script")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every target count by this factor")
    ap.add_argument("--seed", type=int, default=ENGINE.seed)
    ap.add_argument("--audit", choices=["strict", "report"], default=ENGINE.audit_mode,
                    help="strict: records whose label disagrees with this spec are not used as sources")
    ap.add_argument("--audit-only", action="store_true", help="audit the source data only, generate nothing")
    ap.add_argument("--no-meta", action="store_true", help="do not write aug_strategy / aug_source_id")
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
