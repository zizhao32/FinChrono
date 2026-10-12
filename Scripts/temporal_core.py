#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
temporal_core.py - shared core for the single-date and multi-day augmentation scripts

"""

from __future__ import annotations

import calendar
import json
import os
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# =====================================================================
# 1. Annotation spec (all ambiguous conventions live here)
# =====================================================================


@dataclass
class Spec:
    # Weeks start on Monday (0=Mon). Used by this/next/last week and "week of X".
    week_start: int = 0

    # "next Friday":
    #   "next_week" = the Friday of the next calendar week (said on Wed -> +9 days; on Sat -> +6 days)
    #   "nearest"   = the nearest Friday strictly after today
    next_weekday: str = "next_week"

    # "last/previous Friday":
    #   "nearest"   = the nearest Friday strictly before today (said on Sat -> yesterday)
    #   "last_week" = the Friday of the previous calendar week (symmetric with next_week)
    # "this past Friday" / "past Friday" are always nearest.
    last_weekday: str = "nearest"

    # "this Friday":
    #   "this_week" = the Friday of the current calendar week (may already be past)
    #   "upcoming"  = the nearest Friday on or after today
    this_weekday: str = "this_week"

    # "on Friday" with no tense cue: future or past (a tense cue, when present, wins)
    on_weekday_neutral: str = "future"  # "future" | "past" | "nearest"

    # Dates without a year and with no tense cue:
    #   "nearest"   = the year closest to the reference date ("Jan 3" said on 12/28 -> next year)
    #   "past_bias" = the latest occurrence not after the reference date (old single_data_simu / multiple_dates behaviour)
    neutral_year: str = "nearest"

    # Whether "last/next N days|weeks|months|years" includes today:
    #   False: last 3 days = [ref-3, ref-1]; next 3 days = [ref+1, ref+3]
    #   True : last 3 days = [ref-2, ref];   next 3 days = [ref, ref+2]
    # Same rule for every unit; the length is exactly N units (the old 1127 script differed per unit).
    last_n_includes_today: bool = False

    # Business-day calendar: "us_federal" | "nyse" | "weekends_only"
    business_calendar: str = "us_federal"

    # "3 days later / 3 days before / prior" are usually relative to an event, not the reference date; not augmented by default
    allow_event_relative_offsets: bool = False

    # "until June 5": start=None (old 1127 convention) or start=ref
    until_start_is_ref: bool = False

    # Skip "Mon-Fri" in recurring contexts (every / open / hours ...)
    skip_recurring_weekday_ranges: bool = True


# =====================================================================
# 2. Regex fragments (a superset compatible with the original scripts)
# =====================================================================

FLAGS = re.IGNORECASE

MONTH_NAME = (
    r"(?:"
    r"Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|May|"
    r"Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|Sep(?:t(?:ember)?)?\.?|"
    r"Oct(?:ober)?\.?|Nov(?:ember)?\.?|Dec(?:ember)?\.?"
    r")"
)

# The original regex missed Thur/Thurs; added here (superset, original spans unaffected)
WEEKDAY = (
    r"(?:Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|"
    r"Thu(?:r(?:s(?:day)?)?)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"
)

YEAR = r"(?:19\d{2}|20\d{2})"
DATE_SEP_YMD = r"[-/\.]"
DATE_SEP_NUM = r"[-/]"

ORDINAL_DIGIT = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
ORDINAL_WORD = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth|twentieth|twenty-first|"
    r"twenty-second|twenty-third|twenty-fourth|twenty-fifth|"
    r"twenty-sixth|twenty-seventh|twenty-eighth|twenty-ninth|"
    r"thirtieth|thirty-first)"
)
ORDINAL_ANY = rf"(?:{ORDINAL_DIGIT}|{ORDINAL_WORD})"

INT_WORD = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)"
)

APOS = r"['’]"

# Holidays (superset: adds Eves / Mother's / Father's / St. Patrick's; accepts curly apostrophes)
# Order matters: New Year's Eve must precede New Year, Christmas Eve must precede Christmas.
HOLIDAY = (
    rf"(?:New Year{APOS}s Eve|New Year(?:{APOS}s Day|{APOS}s)?|"
    rf"Christmas Eve|Christmas(?: Day)?|"
    rf"Valentine{APOS}?s Day|April Fools{APOS}?(?: Day)?|Juneteenth|Independence Day|"
    rf"Halloween|Veterans{APOS}? Day|"
    rf"Martin Luther King,? Jr\.? Day|MLK Day|Presidents{APOS}? Day|"
    rf"Memorial Day|Labor Day|Columbus Day|Thanksgiving(?: Day)?|"
    rf"Easter(?: Sunday)?|Good Friday|Black Friday|Cyber Monday|Tax Day|"
    rf"Mother{APOS}s Day|Father{APOS}s Day|(?:St\.?|Saint) Patrick{APOS}s Day)"
)

MONTH_TO_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
WEEKDAY_TO_NUM = {
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4, "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}
WORD_INT_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30,
}
INT_TO_WORD = {v: k for k, v in WORD_INT_MAP.items()}
ORDINAL_WORD_TO_DAY = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "twenty-first": 21, "twenty-second": 22, "twenty-third": 23,
    "twenty-fourth": 24, "twenty-fifth": 25, "twenty-sixth": 26,
    "twenty-seventh": 27, "twenty-eighth": 28, "twenty-ninth": 29,
    "thirtieth": 30, "thirty-first": 31,
}
DAY_TO_ORDINAL_WORD = {v: k for k, v in ORDINAL_WORD_TO_DAY.items()}

WD_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WD_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

LABEL_YEAR_MIN = 1900  # the YEAR regex only accepts 19xx/20xx, so labels are limited to that range too
LABEL_YEAR_MAX = 2099


# =====================================================================
# 3. Parsing & date utilities
# =====================================================================

def parse_int_word(token: str) -> int:
    t = token.strip().lower()
    if t.isdigit():
        return int(t)
    if t in WORD_INT_MAP:
        return WORD_INT_MAP[t]
    raise ValueError(f"unknown number token: {token!r}")


def parse_ordinal(token: str) -> int:
    """'5' / '05' / '5th' / 'fifth' / 'twenty-first' -> int (the old 1127 script parsed every ordinal word as 1)"""
    t = token.strip().lower().rstrip(".")
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", t)
    if m:
        return int(m.group(1))
    t = re.sub(r"\s+", "-", t)
    if t in ORDINAL_WORD_TO_DAY:
        return ORDINAL_WORD_TO_DAY[t]
    raise ValueError(f"unknown ordinal token: {token!r}")


def month_str_to_int(token: str) -> int:
    t = token.strip().lower().rstrip(".")
    if t in MONTH_TO_NUM:
        return MONTH_TO_NUM[t]
    raise ValueError(f"unknown month token: {token!r}")


def weekday_str_to_int(token: str) -> int:
    t = token.strip().lower().rstrip(".")
    if t in WEEKDAY_TO_NUM:
        return WEEKDAY_TO_NUM[t]
    raise ValueError(f"unknown weekday token: {token!r}")


def last_dom(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def month_range(y: int, m: int) -> Tuple[date, date]:
    return date(y, m, 1), date(y, m, last_dom(y, m))


def quarter_range(y: int, q: int) -> Tuple[date, date]:
    if q not in (1, 2, 3, 4):
        raise ValueError(q)
    m0 = 3 * (q - 1) + 1
    return date(y, m0, 1), date(y, m0 + 2, last_dom(y, m0 + 2))


def year_range(y: int) -> Tuple[date, date]:
    return date(y, 1, 1), date(y, 12, 31)


def month_shift(y: int, m: int, delta: int) -> Tuple[int, int]:
    idx = y * 12 + (m - 1) + delta
    return idx // 12, idx % 12 + 1


def add_months(d: date, n: int) -> date:
    y, m = month_shift(d.year, d.month, n)
    return date(y, m, min(d.day, last_dom(y, m)))


def add_years(d: date, n: int) -> date:
    y = d.year + n
    return date(y, d.month, min(d.day, last_dom(y, d.month)))


def add_units(d: date, unit: str, n: int) -> date:
    u = unit.lower().rstrip("s")
    if u == "day":
        return d + timedelta(days=n)
    if u == "week":
        return d + timedelta(weeks=n)
    if u == "month":
        return add_months(d, n)
    if u == "year":
        return add_years(d, n)
    raise ValueError(unit)


def week_bounds(d: date, spec: Spec) -> Tuple[date, date]:
    start = d - timedelta(days=(d.weekday() - spec.week_start) % 7)
    return start, start + timedelta(days=6)


def nth_weekday_of_month(y: int, m: int, wd: int, n: int) -> Optional[date]:
    first = date(y, m, 1)
    d = first + timedelta(days=(wd - first.weekday()) % 7 + 7 * (n - 1))
    return d if d.month == m else None


def last_weekday_of_month(y: int, m: int, wd: int) -> date:
    d = date(y, m, last_dom(y, m))
    return d - timedelta(days=(d.weekday() - wd) % 7)


def easter_sunday(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


# =====================================================================
# 4. Choosing among candidate dates by tense polarity
# =====================================================================

def pick_by_polarity(cands: Iterable[Optional[date]], ref: date, polarity: str,
                     spec: Spec, inclusive: bool = True) -> Optional[date]:
    """
    future : earliest candidate >= ref (> ref when inclusive=False)
    past   : latest candidate <= ref (< ref when inclusive=False)
    neutral: spec.neutral_year == "nearest" -> closest (ties go to the future)
             spec.neutral_year == "past_bias" -> latest candidate <= ref
    Falls back to the closest candidate when none matches the polarity.
    """
    cs = sorted({c for c in cands if c is not None})
    if not cs:
        return None
    if polarity == "future":
        fut = [c for c in cs if (c >= ref if inclusive else c > ref)]
        if fut:
            return fut[0]
    elif polarity == "past":
        past = [c for c in cs if (c <= ref if inclusive else c < ref)]
        if past:
            return past[-1]
    elif spec.neutral_year == "past_bias":
        past = [c for c in cs if c <= ref]
        if past:
            return past[-1]
    return min(cs, key=lambda c: (abs((c - ref).days), c < ref))


def month_day_candidates(ref: date, month: int, day: int) -> List[date]:
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        raise ValueError(f"bad month/day {month}/{day}")
    span = 8 if (month, day) == (2, 29) else 1
    out = [d for d in (safe_date(y, month, day) for y in range(ref.year - span, ref.year + span + 1)) if d]
    if not out:
        raise ValueError(f"no valid date for {month}/{day}")
    return out


def infer_month_day(ref: date, month: int, day: int, polarity: str, spec: Spec) -> date:
    d = pick_by_polarity(month_day_candidates(ref, month, day), ref, polarity, spec)
    if d is None:
        raise ValueError("cannot infer year")
    return d


def yearly_candidates(ref: date, fn: Callable[[int], Optional[date]], span: int = 1) -> List[date]:
    out = []
    for y in range(ref.year - span, ref.year + span + 1):
        try:
            d = fn(y)
        except ValueError:
            d = None
        if d is not None:
            out.append(d)
    return out


# =====================================================================
# 5. Holidays (fix: the old code looked up h.replace(" day", ""), so four fixed holidays never resolved)
# =====================================================================

def _norm_holiday(raw: str) -> str:
    h = raw.lower().replace("’", "'")
    return re.sub(r"\s+", " ", h).strip()


_HOLIDAY_CANON: List[Tuple[str, str]] = [
    (r"new year'?s eve", "new_years_eve"),
    (r"new year(?:'s(?: day)?)?", "new_year"),
    (r"christmas eve", "christmas_eve"),
    (r"christmas(?: day)?", "christmas"),
    (r"valentine'?s day", "valentine"),
    (r"april fools'?(?: day)?", "april_fools"),
    (r"juneteenth", "juneteenth"),
    (r"independence day", "independence"),
    (r"halloween", "halloween"),
    (r"veterans'? day", "veterans"),
    (r"(?:martin luther king,? jr\.? day|mlk day)", "mlk"),
    (r"presidents'? day", "presidents"),
    (r"memorial day", "memorial"),
    (r"labor day", "labor"),
    (r"columbus day", "columbus"),
    (r"thanksgiving(?: day)?", "thanksgiving"),
    (r"easter(?: sunday)?", "easter"),
    (r"good friday", "good_friday"),
    (r"black friday", "black_friday"),
    (r"cyber monday", "cyber_monday"),
    (r"tax day", "tax_day"),
    (r"mother's day", "mothers_day"),
    (r"father's day", "fathers_day"),
    (r"(?:st\.?|saint) patrick's day", "st_patrick"),
]


def _thanksgiving(y: int) -> date:
    return nth_weekday_of_month(y, 11, 3, 4)  # type: ignore[return-value]


HOLIDAY_RULES: Dict[str, Callable[[int], Optional[date]]] = {
    "new_year": lambda y: date(y, 1, 1),
    "new_years_eve": lambda y: date(y, 12, 31),
    "christmas_eve": lambda y: date(y, 12, 24),
    "christmas": lambda y: date(y, 12, 25),
    "valentine": lambda y: date(y, 2, 14),
    "april_fools": lambda y: date(y, 4, 1),
    "juneteenth": lambda y: date(y, 6, 19),
    "independence": lambda y: date(y, 7, 4),
    "halloween": lambda y: date(y, 10, 31),
    "veterans": lambda y: nth_weekday_of_month(y, 10, 0, 4) if 1971 <= y <= 1977 else date(y, 11, 11),
    "mlk": lambda y: nth_weekday_of_month(y, 1, 0, 3),
    # Fixed dates before the Uniform Monday Holiday Act (effective 1971)
    "presidents": lambda y: nth_weekday_of_month(y, 2, 0, 3) if y >= 1971 else date(y, 2, 22),
    "memorial": lambda y: last_weekday_of_month(y, 5, 0) if y >= 1971 else date(y, 5, 30),
    "labor": lambda y: nth_weekday_of_month(y, 9, 0, 1),
    "columbus": lambda y: nth_weekday_of_month(y, 10, 0, 2) if y >= 1971 else date(y, 10, 12),
    "thanksgiving": _thanksgiving,
    "easter": easter_sunday,
    "good_friday": lambda y: easter_sunday(y) - timedelta(days=2),
    "black_friday": lambda y: _thanksgiving(y) + timedelta(days=1),
    "cyber_monday": lambda y: _thanksgiving(y) + timedelta(days=4),
    "tax_day": lambda y: date(y, 4, 15),
    "mothers_day": lambda y: nth_weekday_of_month(y, 5, 6, 2),
    "fathers_day": lambda y: nth_weekday_of_month(y, 6, 6, 3),
    "st_patrick": lambda y: date(y, 3, 17),
}

# Holiday surface forms used for text substitution (each is fully matched by the HOLIDAY regex)
HOLIDAY_SURFACES = [
    "New Year's Day", "New Year's Eve", "Christmas", "Christmas Day", "Christmas Eve",
    "Valentine's Day", "April Fools' Day", "Juneteenth", "Independence Day", "Halloween",
    "Veterans Day", "MLK Day", "Presidents' Day", "Memorial Day", "Labor Day",
    "Columbus Day", "Thanksgiving", "Easter", "Good Friday", "Black Friday",
    "Cyber Monday", "Tax Day", "Mother's Day", "Father's Day", "St. Patrick's Day",
]


def holiday_key(raw: str) -> Optional[str]:
    h = _norm_holiday(raw)
    for pat, key in _HOLIDAY_CANON:
        if re.fullmatch(pat, h):
            return key
    return None


def resolve_holiday(raw: str, ref: date, polarity: str, spec: Spec) -> date:
    key = holiday_key(raw)
    if key is None:
        raise ValueError(f"unknown holiday {raw!r}")
    d = pick_by_polarity(yearly_candidates(ref, HOLIDAY_RULES[key]), ref, polarity, spec)
    if d is None:
        raise ValueError("holiday unresolved")
    return d


# =====================================================================
# 6. Business-day calendars (year-aware, with weekend observance)
# =====================================================================

def _raw_holidays(y: int, cal: str) -> List[Tuple[str, date]]:
    """Return (name, nominal date) pairs, listing only holidays actually in effect that year."""
    out: List[Tuple[str, date]] = [("new_year", date(y, 1, 1)), ("independence", date(y, 7, 4)),
                                   ("labor", nth_weekday_of_month(y, 9, 0, 1)),  # type: ignore[list-item]
                                   ("thanksgiving", _thanksgiving(y)), ("christmas", date(y, 12, 25))]
    # Before the Uniform Monday Holiday Act (1971), Washington's Birthday and Memorial Day were fixed dates
    out.append(("presidents", nth_weekday_of_month(y, 2, 0, 3) if y >= 1971 else date(y, 2, 22)))  # type: ignore[arg-type]
    out.append(("memorial", last_weekday_of_month(y, 5, 0) if y >= 1971 else date(y, 5, 30)))
    if cal == "us_federal":
        if y >= 1986:
            out.append(("mlk", nth_weekday_of_month(y, 1, 0, 3)))  # type: ignore[arg-type]
        if y >= 2021:
            out.append(("juneteenth", date(y, 6, 19)))
        if y >= 1971:
            out.append(("columbus", nth_weekday_of_month(y, 10, 0, 2)))  # type: ignore[arg-type]
        elif y >= 1937:
            out.append(("columbus", date(y, 10, 12)))
        if 1971 <= y <= 1977:
            out.append(("veterans", nth_weekday_of_month(y, 10, 0, 4)))  # type: ignore[arg-type]
        else:
            out.append(("veterans", date(y, 11, 11)))
    elif cal == "nyse":
        if y >= 1998:
            out.append(("mlk", nth_weekday_of_month(y, 1, 0, 3)))  # type: ignore[arg-type]
        if y >= 2022:
            out.append(("juneteenth", date(y, 6, 19)))
        out.append(("good_friday", easter_sunday(y) - timedelta(days=2)))
    return out


# NYSE special closures (national days of mourning, disasters, ...); extend as needed
NYSE_SPECIAL_CLOSURES = {
    date(1985, 9, 27), date(1994, 4, 27), date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13),
    date(2001, 9, 14), date(2004, 6, 11), date(2007, 1, 2), date(2012, 10, 29), date(2012, 10, 30),
    date(2018, 12, 5), date(2025, 1, 9),
}

_CAL_CACHE: Dict[Tuple[int, str], frozenset] = {}


def business_holidays(year: int, cal: str) -> frozenset:
    key = (year, cal)
    if key in _CAL_CACHE:
        return _CAL_CACHE[key]
    res = set()
    if cal != "weekends_only":
        for y in (year - 1, year, year + 1):
            for name, d in _raw_holidays(y, cal):
                obs = d
                if y < 1971 and cal == "us_federal":
                    if d.weekday() == 6:  # before 1971 (EO 11582) only Sunday -> Monday applied
                        obs = d + timedelta(days=1)
                elif d.weekday() == 5:  # Saturday -> observed Friday
                    obs = None if (cal == "nyse" and name == "new_year") else d - timedelta(days=1)
                elif d.weekday() == 6:  # Sunday -> observed Monday
                    obs = d + timedelta(days=1)
                if obs is not None and obs.year == year:
                    res.add(obs)
        if cal == "nyse":
            res |= {d for d in NYSE_SPECIAL_CLOSURES if d.year == year}
    fs = frozenset(res)
    _CAL_CACHE[key] = fs
    return fs


def is_business_day(d: date, spec: Spec) -> bool:
    return d.weekday() < 5 and d not in business_holidays(d.year, spec.business_calendar)


def first_business_day_in(start: date, end: date, spec: Spec) -> Optional[date]:
    d = start
    while d <= end:
        if is_business_day(d, spec):
            return d
        d += timedelta(days=1)
    return None


def last_business_day_in(start: date, end: date, spec: Spec) -> Optional[date]:
    d = end
    while d >= start:
        if is_business_day(d, spec):
            return d
        d -= timedelta(days=1)
    return None


# =====================================================================
# 7. Tense polarity detection
# =====================================================================
# Problems with the old version: substring matching ("had " hit "Chad", "due " hit "due to", "next " hit "context"),
# only the 80 chars before the span were inspected, and any future cue beat any past cue.
# New version: whole-word regexes; context before and after the span (same clause); the strong cue nearest the span wins;
# a past cue preceded by a modal/future auxiliary is ignored ("will be closed", "should have").

_FUTURE_CUES = [
    r"will", r"won't", r"wont", r"shall", r"\w+'ll", r"gonna", r"going to",
    r"(?:is|are|am) set to", r"scheduled", r"plans? to", r"planning to", r"planned for",
    r"expects? to", r"intends? to", r"upcoming", r"forthcoming", r"looking forward",
    r"look forward", r"join us", r"due (?:on|by)", r"(?:is|are) due(?!\s+to\b)", r"(?:is|are) coming",
    r"(?:is|are) expected", r"tomorrow", r"from now", r"later this",
]
_PAST_CUES = [
    r"was", r"wasn't", r"were", r"weren't", r"had", r"hadn't", r"did", r"didn't",
    r"(?:has|have) been", r"ago", r"earlier", r"previously", r"already", r"yesterday",
    r"happened", r"occurred", r"began", r"begun", r"ended", r"finished", r"completed",
    r"went", r"came", r"saw", r"met", r"took", r"made", r"said", r"told", r"got",
    r"won", r"lost", r"held", r"paid", r"sold", r"bought", r"spent", r"wrote", r"gave",
    r"found", r"felt", r"kept", r"brought", r"thought", r"announced", r"reported",
    r"released", r"launched", r"closed", r"opened", r"visited", r"arrived", r"left",
]
_ED_EXCLUDE = {
    "scheduled", "planned", "expected", "needed", "intended", "supposed", "based", "related",
    "interested", "limited", "red", "bed", "need", "feed", "speed", "seed", "proceed",
    "exceed", "succeed", "indeed", "united", "detailed", "advanced", "required", "proposed",
    "estimated", "projected", "anticipated", "designed", "dedicated", "listed", "located",
    "concerned", "involved", "unlimited", "registered", "reserved", "closed",
}
_FUT_RE = re.compile(r"\b(?:" + "|".join(_FUTURE_CUES) + r")\b", re.I)
_PAST_RE = re.compile(r"\b(?:" + "|".join(_PAST_CUES) + r")\b", re.I)
_ED_RE = re.compile(r"\b[a-z]{3,}ed\b", re.I)
_MODAL_BEFORE = re.compile(
    r"\b(?:will|shall|would|could|should|might|may|must|can|to|won't|gonna|'ll|going to|is|are|am|'s|'re)"
    r"(?:\s+(?:not|be|have|been|get|also|then|still|probably))*\s*$", re.I)
_CLAUSE_BOUND = re.compile(r"(?:[.!?;]\s+(?=[A-Z\"'])|\n|;)")


def _clause_window(sentence: str, st: int, ed: int, before: int = 120, after: int = 80) -> Tuple[int, int]:
    lo, hi = max(0, st - before), min(len(sentence), ed + after)
    for mb in _CLAUSE_BOUND.finditer(sentence, lo, st):
        lo = mb.end()
    mb = _CLAUSE_BOUND.search(sentence, ed, hi)
    if mb:
        hi = mb.start()
    return lo, hi


def detect_polarity(sentence: str, st: int, ed: Optional[int] = None) -> str:
    """Return 'future' / 'past' / 'neutral'. Only context in the span's clause counts; text inside the span is ignored."""
    if ed is None:
        ed = st
    lo, hi = _clause_window(sentence, st, ed)
    cues: List[Tuple[int, int, str]] = []  # (distance, strength, polarity)

    def dist(a: int, b: int) -> Optional[int]:
        if b <= st:
            return st - b
        if a >= ed:
            return a - ed
        return None  # inside the span

    for m in _FUT_RE.finditer(sentence, lo, hi):
        d = dist(*m.span())
        if d is not None:
            cues.append((d, 0, "future"))
    for rx, strength in ((_PAST_RE, 0), (_ED_RE, 1)):
        for m in rx.finditer(sentence, lo, hi):
            if strength == 1 and m.group(0).lower() in _ED_EXCLUDE:
                continue
            d = dist(*m.span())
            if d is None:
                continue
            if _MODAL_BEFORE.search(sentence[max(lo, m.start() - 30):m.start()]):
                continue  # will be closed / to have finished -> not past
            cues.append((d, strength, "past"))
    if not cues:
        return "neutral"
    strong = [c for c in cues if c[1] == 0]
    pool = strong or cues
    pool.sort()
    if len(pool) > 1 and pool[0][0] == pool[1][0] and pool[0][2] != pool[1][2]:
        return "neutral"
    return pool[0][2]


# =====================================================================
# 8. Format-preserving token rendering
# =====================================================================

def apply_case(template: str, out: str) -> str:
    letters = [c for c in template if c.isalpha()]
    if not letters:
        return out
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return out.upper()
    if all(c.islower() for c in letters):
        return out.lower()
    parts = template.split("-")
    if len(parts) > 1 and all(p[:1].isupper() for p in parts if p):
        return "-".join(p[:1].upper() + p[1:].lower() for p in out.split("-"))
    return out[:1].upper() + out[1:].lower()


def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def render_num(template: str, value: int, pad: Optional[bool] = None) -> str:
    """Numeric month/day. With pad=None, pad iff the template starts with 0 (the old code turned '11' into '3')."""
    if pad is None:
        pad = len(template) >= 2 and template.startswith("0")
    return f"{value:02d}" if pad else str(value)


def render_day(template: str, day: int, pad: Optional[bool] = None) -> str:
    t = template.strip()
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)?", t, re.I)
    if m:
        num = render_num(m.group(1), day, pad)
        if m.group(2):
            suf = ordinal_suffix(day)
            return num + (suf.upper() if m.group(2).isupper() else suf)
        return num
    return apply_case(t, DAY_TO_ORDINAL_WORD[day])


def render_month(template: str, month: int) -> str:
    t = template.strip()
    has_dot = t.endswith(".")
    low = t.lower().rstrip(".")
    if low in ("january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"):
        out = MONTH_FULL[month - 1]
        has_dot = False
    elif low == "sept":
        out = "Sept" if month == 9 else MONTH_ABBR[month - 1]
    else:
        out = MONTH_ABBR[month - 1]
    if has_dot and out != "May":
        out += "."
    return apply_case(t, out)


def render_weekday(template: str, wd: int) -> str:
    t = template.strip()
    low = t.lower().rstrip(".")
    if low in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        out = WD_NAMES[wd]
    elif low == "tues":
        out = "Tues" if wd == 1 else WD_ABBR[wd]
    elif low in ("thur", "thurs"):
        out = ("Thurs" if low == "thurs" else "Thur") if wd == 3 else WD_ABBR[wd]
    else:
        out = WD_ABBR[wd]
    return apply_case(t, out)


def render_count(template: str, n: int) -> Optional[str]:
    """Counts: digits in, digits out; English word in, English word out (None if the number has no word form)."""
    t = template.strip()
    if t.isdigit():
        return str(n)
    if n in INT_TO_WORD:
        return apply_case(t, INT_TO_WORD[n])
    return None


def render_nth(template: str, n: int) -> str:
    t = template.strip()
    if t[:1].isdigit():
        return f"{n}{ordinal_suffix(n)}"
    return apply_case(t, DAY_TO_ORDINAL_WORD[n])


def render_unit(template: str, n: int) -> str:
    """'day'/'days'/'Week' -> singular/plural by n (the old code produced 'one days ago')."""
    base = template.lower().rstrip("s")
    out = base if n == 1 else base + "s"
    return apply_case(template, out)


def infer_pad_style(m: re.Match, numeric_groups: Sequence[str], rng: random.Random,
                    force: bool = False) -> bool:
    """One zero-pad style per span: any token with a leading 0 -> pad; any single-digit token -> no pad; otherwise random."""
    if force:
        return True
    vals = [m.group(g) for g in numeric_groups if g in m.re.groupindex and m.group(g)]
    vals = [re.match(r"\d+", v).group(0) for v in vals if re.match(r"\d+", v)]  # type: ignore[union-attr]
    if any(len(v) >= 2 and v.startswith("0") for v in vals):
        return True
    if any(len(v) == 1 for v in vals):
        return False
    return rng.random() < 0.5


# =====================================================================
# 9. Span utilities
# =====================================================================

def find_match_at(regex: re.Pattern, sentence: str, st: int, ed: int) -> Optional[re.Match]:
    """Re-match exactly on [st, ed). Try match(pos=st) first (keeps right context for lookaheads), then fall back to finditer."""
    m = regex.match(sentence, st)
    if m and m.span() == (st, ed):
        return m
    for m in regex.finditer(sentence):
        if m.span() == (st, ed):
            return m
        if m.start() > st:
            break
    return None


def gspan(m: re.Match, g: str) -> Optional[Tuple[int, int]]:
    if g not in m.re.groupindex:
        return None
    s, e = m.span(g)
    return None if s < 0 else (s, e)


def gval(m: re.Match, g: str) -> Optional[str]:
    if g not in m.re.groupindex:
        return None
    return m.group(g)


def apply_edits(sentence: str, st: int, ed: int, edits: List[Tuple[int, int, str]]) -> Optional[Tuple[str, int, int]]:
    """edits: absolute (start, end, new_text) triples; must lie within [st, ed] and must not overlap."""
    edits = sorted(edits)
    for i, (a, b, _) in enumerate(edits):
        if a < st or b > ed or a > b:
            return None
        if i and a < edits[i - 1][1]:
            return None
    out, cur, new_ed = [], 0, ed
    for a, b, txt in edits:
        out.append(sentence[cur:a])
        out.append(txt)
        cur = b
        new_ed += len(txt) - (b - a)
    out.append(sentence[cur:])
    return "".join(out), st, new_ed


def group_edits(m: re.Match, repl: Dict[str, str]) -> List[Tuple[int, int, str]]:
    edits = []
    for g, txt in repl.items():
        sp = gspan(m, g)
        if sp is not None:
            edits.append((sp[0], sp[1], txt))
    return edits


_UNIT_AFTER = re.compile(r"(\s*)(-?)(\s*)(days?|weeks?|months?|years?)\b", re.I)


def number_unit_edits(sentence: str, m: re.Match, num_group: str, new_n: int) -> Optional[List[Tuple[int, int, str]]]:
    """Replace a count and fix the plurality of the unit right after it (hyphenated adjectives like three-day stay singular)."""
    sp = gspan(m, num_group)
    if sp is None:
        return None
    new_num = render_count(sentence[sp[0]:sp[1]], new_n)
    if new_num is None:
        return None
    edits = [(sp[0], sp[1], new_num)]
    um = _UNIT_AFTER.match(sentence, sp[1])
    if um and um.end() <= m.end() and not um.group(2):
        edits.append((um.start(4), um.end(4), render_unit(um.group(4), new_n)))
    return edits


# =====================================================================
# 10. Record field I/O
# =====================================================================

_ISO_IN_STR = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
REF_STRING_KEYS = ("reference_date", "reference_date_str", "reference_time")


def parse_iso(s: Any) -> Optional[date]:
    if not isinstance(s, str):
        return None
    m = _ISO_IN_STR.search(s)
    if not m:
        return None
    return safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def get_ref(rec: Dict[str, Any]) -> Optional[date]:
    y, m, d = rec.get("reference_year"), rec.get("reference_month"), rec.get("reference_day")
    if y is not None and m is not None and d is not None:
        try:
            return date(int(y), int(m), int(d))
        except (TypeError, ValueError):
            return None
    for k in REF_STRING_KEYS:
        if k in rec:
            r = parse_iso(rec[k])
            if r:
                return r
    return None


def weekday_like(template: Any, wd: int) -> Any:
    """Write weekday in the original field type: int -> int; 'Monday'/'monday'/'MON'/'Mon' -> a string in the same style."""
    if isinstance(template, bool):
        return WD_NAMES[wd]
    if isinstance(template, int):
        return wd
    if isinstance(template, str) and template:
        base = WD_ABBR[wd] if len(template.rstrip(".")) <= 4 else WD_NAMES[wd]
        if template.isupper():
            return base.upper()
        if template.islower():
            return base.lower()
        return base
    return WD_NAMES[wd]


def set_ref(out: Dict[str, Any], ref: date, src: Dict[str, Any]) -> None:
    if any(k in src for k in ("reference_year", "reference_month", "reference_day")) or not any(k in src for k in REF_STRING_KEYS):
        out["reference_year"], out["reference_month"], out["reference_day"] = ref.year, ref.month, ref.day
    if "reference_weekday" in src:
        out["reference_weekday"] = weekday_like(src["reference_weekday"], ref.weekday())
    if "reference_weekday_name" in src:
        out["reference_weekday_name"] = WD_NAMES[ref.weekday()]
    for k in REF_STRING_KEYS:
        if isinstance(src.get(k), str) and _ISO_IN_STR.search(src[k]):
            out[k] = _ISO_IN_STR.sub(ref.isoformat(), src[k], count=1)


# =====================================================================
# 11. Sampling utilities
# =====================================================================

def sample_date(rng: random.Random, ymin: int, ymax: int) -> date:
    y = rng.randint(ymin, ymax)
    m = rng.randint(1, 12)
    return date(y, m, rng.randint(1, last_dom(y, m)))


def sample_month_day(rng: random.Random, allow_feb29: bool = False, min_day: int = 1) -> Tuple[int, int]:
    while True:
        m = rng.randint(1, 12)
        maxd = 29 if (m == 2 and allow_feb29) else last_dom(2001, m)
        if min_day > maxd:
            continue
        return m, rng.randint(min_day, maxd)


def sample_count(rng: random.Random, unit: str, word_form: bool, old: Optional[int] = None) -> int:
    u = unit.lower().rstrip("s")
    hi = {"day": 45, "week": 12, "month": 18, "year": 10}.get(u, 10)
    for _ in range(50):
        if word_form:
            n = rng.choice([k for k in INT_TO_WORD if k <= max(hi, 12)])
        else:
            n = rng.randint(1, hi) if rng.random() < 0.85 else rng.randint(hi, hi * 3)
        if n != old:
            return n
    return (old or 1) + 1


def sample_text_year(ctx: "Ctx") -> int:
    """Explicit year in the sentence, kept consistent with tense and reference date (past -> not after the reference year; future -> not before it)."""
    lo, hi = ctx.cfg.text_year_min, ctx.cfg.text_year_max
    p = detect_polarity(ctx.sentence, ctx.st, ctx.ed)
    if p == "past":
        lo, hi = max(lo, ctx.ref.year - 40), min(hi, ctx.ref.year)
    elif p == "future":
        lo, hi = max(lo, ctx.ref.year), min(hi, ctx.ref.year + 15)
    if lo > hi:
        lo, hi = ctx.cfg.text_year_min, ctx.cfg.text_year_max
    return ctx.rng.randint(lo, hi)


def sample_text_date(ctx: "Ctx") -> date:
    y = sample_text_year(ctx)
    m = ctx.rng.randint(1, 12)
    d = date(y, m, ctx.rng.randint(1, last_dom(y, m)))
    p = detect_polarity(ctx.sentence, ctx.st, ctx.ed)
    if p == "past" and d > ctx.ref:
        d = ctx.ref - timedelta(days=ctx.rng.randint(1, 365))
    elif p == "future" and d < ctx.ref:
        d = ctx.ref + timedelta(days=ctx.rng.randint(1, 365))
    return d


def sample_length(rng: random.Random) -> int:
    """Extra days L of a range (end = start + L); L >= 1 guarantees a multi-day range."""
    r = rng.random()
    if r < 0.6:
        return rng.randint(1, 7)
    if r < 0.88:
        return rng.randint(8, 31)
    return rng.randint(32, 120)


# =====================================================================
# 12. Generic augmentation engine
# =====================================================================

@dataclass
class EngineConfig:
    seed: int = 42
    audit_mode: str = "report"        # "strict": drop sources whose label disagrees with this spec; "report": only log them
    write_meta: bool = True           # write aug_strategy / aug_source_id fields
    ref_year_min: int = 1970
    ref_year_max: int = 2035
    text_year_min: int = 1950
    text_year_max: int = 2040
    week_shift_max: int = 104
    guard_context_weekday: bool = True   # if a weekday name appears outside the span, a new reference date must keep the weekday
    guard_context_year: bool = True      # if a year appears outside the span, a new reference date must keep the year
    allow_ref_only_same_label: bool = False  # keep samples where only the reference date changed and the label did not
    max_attempt_factor: int = 40
    patience_passes: int = 6
    tries_per_strategy: int = 4
    max_audit_examples: int = 200
    id_infix: str = "_rdm_"


@dataclass
class Candidate:
    sentence: str
    st: int
    ed: int
    ref: date
    strategy: str


@dataclass
class Ctx:
    rec: Dict[str, Any]
    sentence: str
    st: int
    ed: int
    m: re.Match
    ref: date
    rng: random.Random
    spec: Spec
    cfg: EngineConfig
    tspec: "TypeSpec"


StrategyFn = Callable[[Ctx], Optional[Candidate]]


@dataclass
class TypeSpec:
    name: str
    regex: re.Pattern
    resolve: Callable[[re.Match, date, str, Spec], Any]
    strategies: List[Tuple[str, float, StrategyFn]]
    source_filter: Optional[Callable[[re.Match, str, Spec], Optional[str]]] = None


@dataclass
class LabelIO:
    read: Callable[[Dict[str, Any]], Any]
    write: Callable[[Dict[str, Any], Any, Dict[str, Any]], None]
    validate: Callable[[Any, TypeSpec], Optional[str]]
    key: Callable[[Any], str]


# ---------- generic reference-date strategies ----------

def _cand_ref(ctx: Ctx, new_ref: date, name: str) -> Optional[Candidate]:
    if new_ref == ctx.ref:
        return None
    if not (ctx.cfg.ref_year_min <= new_ref.year <= ctx.cfg.ref_year_max):
        return None
    return Candidate(ctx.sentence, ctx.st, ctx.ed, new_ref, name)


def strat_ref_full(ctx: Ctx) -> Optional[Candidate]:
    return _cand_ref(ctx, sample_date(ctx.rng, ctx.cfg.ref_year_min, ctx.cfg.ref_year_max), "ref_full")


def strat_ref_year(ctx: Ctx) -> Optional[Candidate]:
    y = ctx.rng.randint(ctx.cfg.ref_year_min, ctx.cfg.ref_year_max)
    d = safe_date(y, ctx.ref.month, ctx.ref.day)
    return _cand_ref(ctx, d, "ref_year") if d else None


def strat_ref_week_shift(ctx: Ctx) -> Optional[Candidate]:
    k = ctx.rng.randint(1, max(1, ctx.cfg.week_shift_max)) * ctx.rng.choice((-1, 1))
    return _cand_ref(ctx, ctx.ref + timedelta(weeks=k), "ref_week_shift")


def strat_ref_jitter(ctx: Ctx) -> Optional[Candidate]:
    """Move the reference date by +-1..400 days: month/day change too, which teaches year/month boundaries."""
    k = ctx.rng.randint(1, 400) * ctx.rng.choice((-1, 1))
    return _cand_ref(ctx, ctx.ref + timedelta(days=k), "ref_jitter")


def combo(text_fn: StrategyFn, ref_fn: StrategyFn, name: str) -> StrategyFn:
    """Change the reference date first, then rewrite the text against it (keeps explicit years consistent with tense)."""
    def fn(ctx: Ctx) -> Optional[Candidate]:
        c1 = ref_fn(ctx)
        if c1 is None:
            return None
        ctx2 = Ctx(ctx.rec, ctx.sentence, ctx.st, ctx.ed, ctx.m, c1.ref, ctx.rng, ctx.spec, ctx.cfg, ctx.tspec)
        c2 = text_fn(ctx2)
        if c2 is None:
            return None
        return Candidate(c2.sentence, c2.st, c2.ed, c1.ref, name)
    return fn


def text_candidate(ctx: Ctx, edits: Optional[List[Tuple[int, int, str]]], name: str) -> Optional[Candidate]:
    if not edits:
        return None
    r = apply_edits(ctx.sentence, ctx.st, ctx.ed, edits)
    if r is None or r[0] == ctx.sentence:
        return None
    return Candidate(r[0], r[1], r[2], ctx.ref, name)


# ---------- context conflicts ----------

_CTX_WEEKDAY = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon|Tues?|Wed|Thur?s?|Fri|Sat|Sun)\b\.?")  # capitalised only, to avoid the common words sun/sat/wed
_CTX_YEAR = re.compile(rf"\b{YEAR}\b")


def context_conflict(c: Candidate, old_ref: date, cfg: EngineConfig) -> Optional[str]:
    if c.ref == old_ref:
        return None
    outside = c.sentence[:c.st] + " \u0000 " + c.sentence[c.ed:]
    if cfg.guard_context_weekday and c.ref.weekday() != old_ref.weekday() and _CTX_WEEKDAY.search(outside):
        return "ctx_weekday_conflict"
    if cfg.guard_context_year and c.ref.year != old_ref.year and _CTX_YEAR.search(outside):
        return "ctx_year_conflict"
    return None


# ---------- engine ----------

class Augmenter:
    def __init__(self, types: Dict[str, TypeSpec], label_io: LabelIO, spec: Spec, cfg: EngineConfig):
        self.types, self.io, self.spec, self.cfg = types, label_io, spec, cfg
        self.rng = random.Random(cfg.seed)
        self.stats: Dict[str, Counter] = defaultdict(Counter)
        self.audit_examples: List[Dict[str, Any]] = []
        self.seen: set = set()
        self.used_ids: set = set()

    # --- source audit: recompute the label on the original reference date under this spec and compare ---
    def audit_source(self, rec: Dict[str, Any], ts: TypeSpec) -> Tuple[str, Optional[re.Match], Optional[date]]:
        s = rec.get("sentence")
        idx = rec.get("time_expression_substring_index")
        sub = rec.get("time_expression_substring")
        if not isinstance(s, str) or not isinstance(idx, (list, tuple)) or len(idx) != 2:
            return "bad_record", None, None
        st, ed = int(idx[0]), int(idx[1])
        if not (0 <= st < ed <= len(s)) or s[st:ed] != sub:
            return "bad_span", None, None
        ref = get_ref(rec)
        if ref is None:
            return "no_ref", None, None
        m = find_match_at(ts.regex, s, st, ed)
        if m is None:
            return "regex_no_match", None, ref
        if ts.source_filter:
            why = ts.source_filter(m, s, self.spec)
            if why:
                return "skip:" + why, m, ref
        try:
            lab = ts.resolve(m, ref, s, self.spec)
        except Exception:
            return "unresolvable", m, ref
        if self.io.validate(lab, ts):
            return "unresolvable", m, ref
        stored = self.io.read(rec)
        if stored is not None and self.io.key(stored) == self.io.key(lab):
            return "agree", m, ref
        if len(self.audit_examples) < self.cfg.max_audit_examples * max(1, len(self.types)):
            if sum(1 for e in self.audit_examples if e["type"] == ts.name) < self.cfg.max_audit_examples:
                self.audit_examples.append({
                    "type": ts.name, "id": rec.get("id"), "sentence": s, "substring": sub,
                    "reference": ref.isoformat(), "stored_label": self.io.key(stored) if stored is not None else None,
                    "spec_label": self.io.key(lab), "polarity": detect_polarity(s, st, ed),
                })
        return "disagree", m, ref

    def _record_key(self, sentence: str, ref: date, label: Any) -> str:
        return f"{sentence}\u241f{ref.isoformat()}\u241f{self.io.key(label)}"

    def _try_once(self, rec: Dict[str, Any], ts: TypeSpec, m: re.Match, ref: date) -> Tuple[Optional[Dict[str, Any]], str]:
        st, ed = m.span()
        ctx = Ctx(rec, rec["sentence"], st, ed, m, ref, self.rng, self.spec, self.cfg, ts)
        order = self._strategy_order(ts)
        last_reason = "no_strategy"
        old_label = None
        try:
            old_label = ts.resolve(m, ref, rec["sentence"], self.spec)
        except Exception:
            pass
        for name, fn in order:
            for _ in range(self.cfg.tries_per_strategy):
                try:
                    c = fn(ctx)
                except Exception:
                    c = None
                if c is None:
                    last_reason = f"strategy_none:{name}"
                    continue
                reason, out = self._finalize(rec, ts, c, ref, old_label)
                if out is not None:
                    return out, "ok"
                last_reason = reason
                self.stats[ts.name]["reject:" + reason] += 1
        return None, last_reason

    def _strategy_order(self, ts: TypeSpec) -> List[Tuple[str, StrategyFn]]:
        items = [(n, w, f) for n, w, f in ts.strategies if w > 0]
        order = []
        while items:
            tot = sum(w for _, w, _ in items)
            r = self.rng.random() * tot
            acc = 0.0
            for i, (n, w, f) in enumerate(items):
                acc += w
                if r <= acc:
                    order.append((n, f))
                    items.pop(i)
                    break
            else:
                n, w, f = items.pop()
                order.append((n, f))
        return order

    def _finalize(self, rec, ts: TypeSpec, c: Candidate, old_ref: date, old_label: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not (0 <= c.st < c.ed <= len(c.sentence)):
            return "bad_new_span", None
        # Round trip: the same regex must match the new sentence exactly on the new span, and the label is re-resolved
        m2 = find_match_at(ts.regex, c.sentence, c.st, c.ed)
        if m2 is None:
            return "roundtrip_no_match", None
        if ts.source_filter:
            why = ts.source_filter(m2, c.sentence, self.spec)
            if why:
                return "filtered:" + why, None
        why = context_conflict(c, old_ref, self.cfg)
        if why:
            return why, None
        try:
            label = ts.resolve(m2, c.ref, c.sentence, self.spec)
        except Exception:
            return "unresolvable", None
        bad = self.io.validate(label, ts)
        if bad:
            return bad, None
        if (not self.cfg.allow_ref_only_same_label and c.sentence == rec["sentence"]
                and old_label is not None and self.io.key(label) == self.io.key(old_label)):
            return "ref_only_same_label", None
        k = self._record_key(c.sentence, c.ref, label)
        if k in self.seen:
            return "duplicate", None
        out = deepcopy(rec)
        out["sentence"] = c.sentence
        out["time_expression_substring"] = c.sentence[c.st:c.ed]
        out["time_expression_substring_index"] = [c.st, c.ed]
        set_ref(out, c.ref, rec)
        self.io.write(out, label, rec)
        out["time_expression_type"] = rec.get("time_expression_type", ts.name)
        # final hard constraint
        if out["sentence"][c.st:c.ed] != out["time_expression_substring"]:
            return "final_span_mismatch", None
        self.seen.add(k)
        out["id"] = self._new_id(rec)
        out["aug_strategy"] = c.strategy
        out["aug_source_id"] = rec.get("id")
        return "ok", out

    def _new_id(self, rec: Dict[str, Any]) -> str:
        base = str(rec.get("id", "noid"))
        while True:
            nid = f"{base}{self.cfg.id_infix}{self.rng.randint(10_000_000, 999_999_999)}"
            if nid not in self.used_ids:
                self.used_ids.add(nid)
                return nid

    def run(self, records: List[Dict[str, Any]], counts: Dict[str, int], out_path: str,
            audit_only: bool = False) -> Dict[str, Any]:
        for r in records:
            ref = get_ref(r)
            s = r.get("sentence")
            if isinstance(s, str) and ref is not None:
                lab = self.io.read(r)
                if lab is not None:
                    self.seen.add(self._record_key(s, ref, lab))
            if r.get("id") is not None:
                self.used_ids.add(str(r["id"]))

        by_type: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(records):
            t = r.get("time_expression_type")
            if t:
                by_type[t].append(i)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fout = None if audit_only else open(out_path, "w", encoding="utf-8")
        summary: Dict[str, Any] = {}
        try:
            for tname, target in counts.items():
                st = self.stats[tname]
                ts = self.types.get(tname)
                if ts is None:
                    st["no_typespec"] += 1
                    print(f"[WARN] no augmentation definition for {tname}, skipped")
                    continue
                idxs = by_type.get(tname, [])
                st["sources"] = len(idxs)
                usable: List[Tuple[Dict[str, Any], re.Match, date]] = []
                for i in idxs:
                    status, m, ref = self.audit_source(records[i], ts)
                    st["audit:" + status] += 1
                    ok = status == "agree" or (status == "disagree" and self.cfg.audit_mode == "report")
                    if ok and m is not None and ref is not None:
                        usable.append((records[i], m, ref))
                st["usable_sources"] = len(usable)
                if audit_only or target <= 0:
                    continue
                if not usable:
                    print(f"[WARN] {tname}: no usable source records (sources={len(idxs)}), skipped")
                    continue
                self._generate(tname, ts, usable, target, fout)
                if st["generated"] < target:
                    print(f"[WARN] {tname}: only generated {st['generated']}/{target}")
        finally:
            if fout:
                fout.close()

        for tname in counts:
            summary[tname] = dict(self.stats[tname])
        return summary

    def _generate(self, tname: str, ts: TypeSpec, usable, target: int, fout) -> None:
        st = self.stats[tname]
        order = list(range(len(usable)))
        attempts, max_attempts = 0, max(2000, target * self.cfg.max_attempt_factor)
        dry_passes = 0
        while st["generated"] < target and attempts < max_attempts and dry_passes < self.cfg.patience_passes:
            self.rng.shuffle(order)  # round-robin: at most one output per source per pass, so a few sources cannot dominate
            made = 0
            for j in order:
                if st["generated"] >= target or attempts >= max_attempts:
                    break
                attempts += 1
                rec, m, ref = usable[j]
                out, reason = self._try_once(rec, ts, m, ref)
                if out is None:
                    st["attempt_failed"] += 1
                    continue
                st["strategy:" + str(out.get("aug_strategy"))] += 1
                if not self.cfg.write_meta:
                    out.pop("aug_strategy", None)
                    out.pop("aug_source_id", None)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                st["generated"] += 1
                made += 1
            dry_passes = dry_passes + 1 if made == 0 else 0
        st["attempts"] = attempts


# =====================================================================
# 13. I/O & reporting
# =====================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(obj)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[WARN] {path}: skipped {bad} invalid JSON lines")
    return data


def print_report(summary: Dict[str, Any], out_path: str, audit_examples: List[Dict[str, Any]],
                 audit_only: bool) -> None:
    print("\n" + "=" * 100)
    hdr = f"{'type':<52}{'src':>6}{'agree':>7}{'disagr':>7}{'nomatch':>8}{'unres':>6}{'skip':>6}{'gen':>8}"
    print(hdr)
    print("-" * 100)
    tot_gen = 0
    for t, s in summary.items():
        skip = sum(v for k, v in s.items() if k.startswith("audit:skip:"))
        gen = s.get("generated", 0)
        tot_gen += gen
        print(f"{t:<52}{s.get('sources', 0):>6}{s.get('audit:agree', 0):>7}{s.get('audit:disagree', 0):>7}"
              f"{s.get('audit:regex_no_match', 0):>8}{s.get('audit:unresolvable', 0):>6}{skip:>6}{gen:>8}")
    print("-" * 100)
    print(f"Total generated: {tot_gen}")
    base = os.path.splitext(out_path)[0]
    rep = base + ".report.json"
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    aud = base + ".audit.jsonl"
    with open(aud, "w", encoding="utf-8") as f:
        for e in audit_examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Report: {rep}")
    print(f"Source labels disagreeing with the spec: {aud}  ({len(audit_examples)} examples)")
    if not audit_only:
        print(f"Augmented data: {out_path}")
