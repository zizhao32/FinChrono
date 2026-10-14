#!/usr/bin/env python3
"""
Full contamination / overlap analysis between FinChrono and the external test sets
(TempEval-3, WikiWars). 

  1. External test-set sizes (instances and unique sentences).
  2. N-gram overlap (default n = 8, 13; plus 5, 6 as a sensitivity check):
       - exact duplicate sentences
       - share of external sentences sharing >= 1 n-gram with FinChrono
       - share of the external set's distinct n-grams found in FinChrono
       - example matched n-grams for manual inspection
     Overlap is measured on UNIQUE external sentences (one sentence can carry several
     time expressions). Only sentences with >= n tokens are eligible for size n.
  3. Reference pair: overlap of WikiWars against TempEval-3 (background level).
  4. Positive control: FinChrono sentences (half verbatim, half with a prefix added)
     injected as a fake "external" set, to verify the detector catches real overlap.
  5. Supporting statistics:
       - share of external temporal-expression strings never seen in FinChrono
       - external instances whose (expression string, gold date) pair occurs in FinChrono
       - target-year range of each set; share of WikiWars targets before 1980 vs. the
         share of FinChrono targets in 1900-1979
       - share of FinChrono targets inside 1900-2030

Input format: JSONL, one instance per line, with a "sentence" field; temporal instances
carry "time_expression_substring" and either "time_standard_format" (single day) or
"time_range_start"/"time_range_end" (interval). No-date instances have no time fields.

Usage:
  python ngram_overlap_full_analysis.py \
      --finchrono finchrono_train.jsonl \
      --tempeval tempeval_data.jsonl \
      --wikiwars wikiwars_data.jsonl \
      --json-out overlap_report.json

  --finchrono accepts several files, e.g. the full A/B/C annotation pool, to obtain an
  upper bound on overlap for any subset of it.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ------------------------------------------------------------------ utilities
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
                print(f"[WARN] {path}:{lineno} skipped ({e})", file=sys.stderr)
    return rows


def tokens(text):
    """NFKC-normalize, lowercase, split into word tokens (punctuation dropped)."""
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def ngram_hashes(toks, n):
    """Deterministic 64-bit hashes of all n-grams of a token list."""
    return [int.from_bytes(hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"),
                                           digest_size=8).digest(), "little")
            for i in range(len(toks) - n + 1)]


def unique_sentences(rows):
    """Unique sentences after whitespace normalization, original order preserved."""
    seen, out = set(), []
    for r in rows:
        s = " ".join(str(r.get("sentence", "")).split())
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def gold_dates(r):
    if r.get("time_range_start") is not None:
        return [r["time_range_start"], r["time_range_end"]]
    if r.get("time_standard_format") is not None:
        return [r["time_standard_format"]]
    return []


def expr(r):
    s = r.get("time_expression_substring")
    return s.lower().strip() if isinstance(s, str) else None


# ------------------------------------------------------------------ n-gram core
def build_index(sentences, ns):
    index = {n: set() for n in ns}
    norm = set()
    for s in sentences:
        t = tokens(s)
        norm.add(" ".join(t))
        for n in ns:
            index[n].update(ngram_hashes(t, n))
    return index, norm


def overlap(ext_sentences, index, norm, ns, n_examples=5):
    toks = [tokens(s) for s in ext_sentences]
    res = {"n_sentences": len(ext_sentences),
           "exact_duplicates": sum(1 for t in toks if t and " ".join(t) in norm)}
    for n in ns:
        eligible = hit_sents = 0
        all_g, hit_g = set(), set()
        examples = Counter()
        for t in toks:
            hs = ngram_hashes(t, n)
            if not hs:
                continue
            eligible += 1
            hit = False
            for i, h in enumerate(hs):
                all_g.add(h)
                if h in index[n]:
                    hit_g.add(h)
                    hit = True
                    examples[" ".join(t[i:i + n])] += 1
            hit_sents += hit
        res[n] = {
            "eligible": eligible,
            "contaminated": hit_sents,
            "contaminated_rate": hit_sents / eligible if eligible else float("nan"),
            "distinct_ngrams": len(all_g),
            "ngram_coverage": len(hit_g) / len(all_g) if all_g else float("nan"),
            "examples": [g for g, _ in examples.most_common(n_examples)],
        }
    return res


def print_overlap(title, r, ns):
    print(f"\n== {title} ==  unique sentences={r['n_sentences']}  "
          f"exact duplicates={r['exact_duplicates']}")
    for n in ns:
        x = r[n]
        print(f"  {n:>2}-gram: contaminated {x['contaminated']}/{x['eligible']} "
              f"({x['contaminated_rate']:.2%}); n-gram coverage {x['ngram_coverage']:.2%} "
              f"of {x['distinct_ngrams']} distinct")
        for g in x["examples"]:
            print(f"           e.g. \"{g}\"")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="FinChrono vs. external test-set overlap analysis.")
    ap.add_argument("--finchrono", nargs="+", required=True, help="FinChrono JSONL file(s)")
    ap.add_argument("--tempeval", required=True, help="converted TempEval-3 JSONL")
    ap.add_argument("--wikiwars", required=True, help="converted WikiWars JSONL")
    ap.add_argument("--n", nargs="+", type=int, default=[8, 13], help="main n-gram sizes")
    ap.add_argument("--sensitivity-n", nargs="*", type=int, default=[5, 6],
                    help="shorter n-gram sizes reported as a sensitivity check")
    ap.add_argument("--control-size", type=int, default=200, help="positive-control sentences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    fin_rows = []
    for p in args.finchrono:
        fin_rows += read_jsonl(p)
    ext_rows = {"TempEval-3": read_jsonl(args.tempeval), "WikiWars": read_jsonl(args.wikiwars)}

    fin_sents = [str(r.get("sentence", "")) for r in fin_rows if r.get("sentence")]
    ext_sents = {k: unique_sentences(v) for k, v in ext_rows.items()}
    all_n = sorted(set(args.n) | set(args.sensitivity_n))

    report = {"config": vars(args), "finchrono_instances": len(fin_rows), "sizes": {},
              "overlap": {}, "sensitivity": {}, "reference": {}, "positive_control": {},
              "supporting": {}}

    # 1. sizes
    print(f"[INFO] FinChrono: {len(fin_rows)} instances from {len(args.finchrono)} file(s)")
    for k in ext_rows:
        report["sizes"][k] = {"instances": len(ext_rows[k]), "unique_sentences": len(ext_sents[k])}
        print(f"[INFO] {k}: {len(ext_rows[k])} instances, {len(ext_sents[k])} unique sentences")

    # 2. n-gram overlap against FinChrono
    index, norm = build_index(fin_sents, all_n)
    for k, sents in ext_sents.items():
        r = overlap(sents, index, norm, all_n)
        report["overlap"][k] = {"n_sentences": r["n_sentences"],
                                "exact_duplicates": r["exact_duplicates"],
                                **{str(n): r[n] for n in args.n}}
        report["sensitivity"][k] = {str(n): r[n] for n in args.sensitivity_n}
        print_overlap(f"{k} vs FinChrono", r, args.n)
        if args.sensitivity_n:
            print("  sensitivity (short n-grams; expected to match only generic phrases):")
            for n in args.sensitivity_n:
                x = r[n]
                print(f"    {n}-gram: {x['contaminated']}/{x['eligible']} ({x['contaminated_rate']:.2%}); "
                      f"e.g. {x['examples'][:3]}")

    # 3. reference pair: WikiWars against TempEval-3
    ref_index, ref_norm = build_index(ext_sents["TempEval-3"], all_n)
    r = overlap(ext_sents["WikiWars"], ref_index, ref_norm, all_n)
    report["reference"]["WikiWars_vs_TempEval-3"] = {str(n): r[n] for n in all_n}
    print_overlap("reference: WikiWars vs TempEval-3 (background level)", r, args.n)

    # 4. positive control
    rng = random.Random(args.seed)
    k = min(args.control_size, len(fin_sents))
    ctrl = [s if i % 2 == 0 else "Analysts noted that " + s
            for i, s in enumerate(rng.sample(fin_sents, k))]
    r = overlap(ctrl, index, norm, args.n, n_examples=0)
    report["positive_control"] = {"n_sentences": k, "exact_duplicates": r["exact_duplicates"],
                                  **{str(n): r[n] for n in args.n}}
    print_overlap(f"positive control ({k} FinChrono sentences, half with a prefix)", r, args.n)

    # 5. supporting statistics
    fin_exprs = Counter(e for e in (expr(x) for x in fin_rows) if e)
    fin_pairs = set((expr(x), d) for x in fin_rows if expr(x) for d in gold_dates(x)[:1])
    fin_years = [int(d[:4]) for x in fin_rows for d in gold_dates(x)]
    fin_share_1900_1979 = sum(1900 <= y < 1980 for y in fin_years) / len(fin_years)
    fin_share_1900_2030 = sum(1900 <= y <= 2030 for y in fin_years) / len(fin_years)
    print("\n== supporting statistics ==")
    print(f"  FinChrono target years {min(fin_years)}-{max(fin_years)}; "
          f"in 1900-2030: {fin_share_1900_2030:.1%}; in 1900-1979: {fin_share_1900_1979:.2%}")
    report["supporting"]["FinChrono"] = {
        "target_year_min": min(fin_years), "target_year_max": max(fin_years),
        "share_targets_1900_2030": fin_share_1900_2030,
        "share_targets_1900_1979": fin_share_1900_1979}
    for k, rows in ext_rows.items():
        temporal = [x for x in rows if expr(x) and gold_dates(x)]
        exs = [expr(x) for x in temporal]
        unseen = sum(e not in fin_exprs for e in exs)
        pair_hits = sum((expr(x), gold_dates(x)[0]) in fin_pairs for x in temporal)
        years = [int(gold_dates(x)[0][:4]) for x in temporal]
        pre1980 = sum(y < 1980 for y in years) / len(years)
        report["supporting"][k] = {
            "temporal_instances": len(temporal),
            "expression_strings_unseen_in_finchrono": unseen / len(exs),
            "expr_and_gold_date_pairs_seen_in_finchrono": pair_hits,
            "target_year_min": min(years), "target_year_max": max(years),
            "share_targets_before_1980": pre1980}
        print(f"  {k}: expression strings never seen in FinChrono {unseen}/{len(exs)} "
              f"({unseen / len(exs):.1%}); (expression, gold date) pairs seen {pair_hits}; "
              f"target years {min(years)}-{max(years)}; before 1980: {pre1980:.1%}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] report written to {args.json_out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")
