#!/usr/bin/env python3
"""
FinChrono evaluation script.

Computes the offline metrics reported in the paper:
  * Acc      - A single-day gold date is correct only
               if the predicted interval is exactly [gold, gold]; a multi-day gold
               interval is correct only if BOTH predicted start and end equal the gold
               start and end. No partial credit.
  * Acc@7    - fraction of temporal instances whose predicted start AND end are each
               within 7 days of the gold start/end.
  * MAE      - mean absolute error in days, averaged over the two interval endpoints
               . See --missing-policy for instances with no usable prediction.
  * Direct / Indirect accuracy, or any other breakdown via --group-field.
  * Invalid-output rate: unparsable / schema-violating predictions.
  * No-date false-positive rate: share of no-date negatives for which a date was predicted.

By default Acc, Acc@7 and MAE are computed over TEMPORAL instances only (no-date
negatives excluded); no-date negatives are scored separately (nodate_fpr). An
"all-instance" accuracy that also counts a no-date negative as correct when nothing
is predicted is reported as acc_all.

------------------------------------------------------------------------------
Gold file (JSONL, one instance per line). Three record types are recognized:
  single day : {"id", "sentence", ..., "time_standard_format": "YYYY-MM-DD"}
  interval   : {"id", "sentence", ..., "time_range_start": "YYYY-MM-DD",
                                       "time_range_end":   "YYYY-MM-DD"}
  no-date    : {"id", "sentence"}                     (no time fields)
Optional gold fields: "time_expression_substring_index": [s, e] (char offsets),
and any grouping field (e.g. "is_indirect", "category").

Prediction file (JSONL, one line per gold id):
  {"id": "...", "start": "YYYY-MM-DD" | null, "end": "YYYY-MM-DD" | null,
   "span": [s, e] (optional), "valid": true | false (optional)}
  - start = end = null means "no temporal mention predicted".
  - For a single-day prediction set start == end.
  - "valid": false marks an unparsable / schema-violating model output.
  - A gold id missing from the prediction file is treated as an invalid output.
------------------------------------------------------------------------------

Usage:
  python evaluate_finchrono.py --gold test.jsonl --pred fcr_seed1.jsonl
  python evaluate_finchrono.py --gold test.jsonl \
      --pred fcr_seed1.jsonl fcr_seed2.jsonl fcr_seed3.jsonl \
      --group-field is_indirect --json-out results.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date


# ----------------------------------------------------------------------------- io
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno} unparsable line skipped ({e})", file=sys.stderr)
    return rows


def parse_date(s):
    """Return a datetime.date, or None if s is missing or not a valid calendar date."""
    if s is None:
        return None
    try:
        return date.fromisoformat(str(s).strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- gold
def gold_interval(rec):
    """Return (start, end) dates for temporal instances, or None for no-date negatives."""
    if rec.get("time_range_start") is not None or rec.get("time_range_end") is not None:
        a, b = parse_date(rec.get("time_range_start")), parse_date(rec.get("time_range_end"))
        if a is None or b is None or b < a:
            raise ValueError(f"gold id {rec.get('id')}: invalid gold interval")
        return a, b
    if rec.get("time_standard_format") is not None:
        d = parse_date(rec.get("time_standard_format"))
        if d is None:
            raise ValueError(f"gold id {rec.get('id')}: invalid gold date")
        return d, d
    return None


# --------------------------------------------------------------------- prediction
def pred_interval(p):
    """
    Returns (status, interval):
      status = "invalid"  -> unparsable / schema-violating output
               "none"     -> model predicted no temporal mention
               "date"     -> valid interval returned in `interval`
    Mirrors the production validation rules: invalid dates and reversed bounds
    are invalid outputs.
    """
    if p is None or p.get("valid") is False:
        return "invalid", None
    s, e = p.get("start"), p.get("end")
    if s is None and e is None:
        return "none", None
    if s is None or e is None:
        return "invalid", None
    a, b = parse_date(s), parse_date(e)
    if a is None or b is None or b < a:
        return "invalid", None
    return "date", (a, b)


# ------------------------------------------------------------------------ scoring
def evaluate(gold_rows, preds_by_id, group_field, missing_policy, penalty_days):
    n_temporal = n_nodate = 0
    exact = within7 = 0
    abs_err_sum, abs_err_n = 0.0, 0
    nodate_correct = nodate_fp = 0
    invalid = 0
    missing_ids = 0
    span_total = span_exact = 0
    groups = defaultdict(lambda: [0, 0])   # group value -> [correct, total] (temporal only)

    for g in gold_rows:
        gid = g.get("id")
        gi = gold_interval(g)
        p = preds_by_id.get(gid)
        if p is None:
            missing_ids += 1
        status, pi = pred_interval(p)
        if status == "invalid":
            invalid += 1

        if gi is None:                      # no-date negative
            n_nodate += 1
            if status == "none":
                nodate_correct += 1
            elif status == "date":
                nodate_fp += 1
            continue

        n_temporal += 1
        ok = status == "date" and pi == gi
        exact += ok
        if group_field is not None:
            key = str(g.get(group_field, "<missing>"))
            groups[key][0] += ok
            groups[key][1] += 1

        if status == "date":
            ea = abs((pi[0] - gi[0]).days)
            eb = abs((pi[1] - gi[1]).days)
            within7 += (ea <= 7 and eb <= 7)
            abs_err_sum += (ea + eb) / 2.0
            abs_err_n += 1
        elif missing_policy == "penalty":
            abs_err_sum += penalty_days
            abs_err_n += 1
        # missing_policy == "exclude": no contribution to MAE (coverage reported)

        gs, ps = g.get("time_expression_substring_index"), (p or {}).get("span")
        if gs is not None and ps is not None:
            span_total += 1
            span_exact += list(gs) == list(ps)

    n_all = n_temporal + n_nodate
    res = {
        "n_temporal": n_temporal,
        "n_nodate": n_nodate,
        "acc": exact / n_temporal if n_temporal else float("nan"),
        "acc@7": within7 / n_temporal if n_temporal else float("nan"),
        "mae_days": abs_err_sum / abs_err_n if abs_err_n else float("nan"),
        "mae_coverage": abs_err_n / n_temporal if n_temporal else float("nan"),
        "acc_all": (exact + nodate_correct) / n_all if n_all else float("nan"),
        "nodate_fpr": nodate_fp / n_nodate if n_nodate else float("nan"),
        "invalid_rate": invalid / n_all if n_all else float("nan"),
        "missing_predictions": missing_ids,
    }
    if span_total:
        res["span_exact_match"] = span_exact / span_total
    for k, (c, t) in sorted(groups.items()):
        res[f"acc[{group_field}={k}]"] = c / t if t else float("nan")
        res[f"n[{group_field}={k}]"] = t
    return res


def mean_sd(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))   # sample SD
    return m, sd


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Evaluate temporal grounding predictions on FinChrono-format data.")
    ap.add_argument("--gold", required=True, help="gold JSONL (FinChrono split or converted external benchmark)")
    ap.add_argument("--pred", required=True, nargs="+", help="one prediction JSONL per seed")
    ap.add_argument("--group-field", default=None,
                    help="gold field for accuracy breakdown, e.g. is_indirect or category")
    ap.add_argument("--missing-policy", choices=["exclude", "penalty"], default="exclude",
                    help="MAE handling for temporal instances without a valid predicted interval "
                         "(default: exclude, coverage reported as mae_coverage)")
    ap.add_argument("--penalty-days", type=float, default=365.0,
                    help="error assigned per such instance when --missing-policy=penalty")
    ap.add_argument("--json-out", default=None, help="optional path to write results as JSON")
    args = ap.parse_args()

    gold = read_jsonl(args.gold)
    ids = [g.get("id") for g in gold]
    if len(ids) != len(set(ids)):
        sys.exit("[ERROR] duplicate ids in gold file")

    per_seed = []
    for path in args.pred:
        preds = {}
        for p in read_jsonl(path):
            if p.get("id") in preds:
                print(f"[WARN] {path}: duplicate prediction for id {p.get('id')}, keeping the first",
                      file=sys.stderr)
                continue
            preds[p.get("id")] = p
        extra = set(preds) - set(ids)
        if extra:
            print(f"[WARN] {path}: {len(extra)} prediction ids not in gold, ignored", file=sys.stderr)
        r = evaluate(gold, preds, args.group_field, args.missing_policy, args.penalty_days)
        if r["missing_predictions"]:
            print(f"[WARN] {path}: {r['missing_predictions']} gold ids have no prediction "
                  f"(counted as invalid outputs)", file=sys.stderr)
        per_seed.append(r)

    keys = []
    for r in per_seed:                       # union of keys, first-seen order
        keys += [k for k in r if k not in keys]
    summary = {}
    for k in keys:
        m, sd = mean_sd([r.get(k, float("nan")) for r in per_seed])
        summary[k] = {"mean": m, "sd": sd}

    print(f"Gold: {args.gold} | seeds: {len(per_seed)} | MAE missing-policy: {args.missing_policy}")
    print(f"{'metric':<34}{'mean':>12}{'(sd)':>10}")
    for k in keys:
        m, sd = summary[k]["mean"], summary[k]["sd"]
        if k.startswith("n_") or k.startswith("n[") or k == "missing_predictions":
            print(f"{k:<34}{m:>12g}")
        else:
            print(f"{k:<34}{m:>12.4f}{f'({sd:.4f})':>10}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "per_seed": per_seed, "summary": summary},
                      f, ensure_ascii=False, indent=2)
        print(f"[INFO] results written to {args.json_out}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
