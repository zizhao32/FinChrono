# FinChrono

**FinChrono** is an ongoing project on **temporal understanding and calendar date reasoning** in financial text.

This repository currently hosts the first released dataset:

> **FinChronoDate** — a sentence-level dataset for **precise calendar date resolution**  
> in **English financial and economic news and reports**.

FinChronoDate is designed for training and evaluating models that can:

- detect time expressions in a sentence,
- reason about missing or implicit temporal information using a reference date,
- and resolve them to **concrete calendar dates** (`YYYY-MM-DD`), along with weekday consistency.

---

## Table of Contents

1. [FinChronoDate Overview](#1-finchronodate-overview)  
2. [Data Source & Granularity](#2-data-source--granularity)  
3. [Data Schema](#3-data-schema)  
   - [id](#31-id-string)  
   - [sentence](#32-sentence-string)  
   - [time_expression_substring](#33-time_expression_substring-string)  
   - [time_expression_substring_index](#34-time_expression_substring_index-listint)  
   - [reference_year](#35-reference_year-integer)  
   - [reference_month](#36-reference_month-integer)  
   - [reference_day](#37-reference_day-integer)  
   - [reference_weekday](#38-reference_weekday-string)  
   - [time_standard_format](#39-time_standard_format-string)  
   - [time_weekday](#310-time_weekday-string)  
   - [time_expression_type](#311-time_expression_type-string-optional)  
4. [Example](#4-example)  
5. [Suggested File Layout](#5-suggested-file-layout)  
6. [Basic Usage](#6-basic-usage)  
7. [Intended Use](#7-intended-use)  
8. [Limitations](#8-limitations)  
9. [License](#9-license)  
10. [Citation](#10-citation)

---

## 1. FinChronoDate Overview

**FinChronoDate** is a **sentence-level** dataset for **calendar date resolution** from natural-language time expressions in the domain of **financial and economic news**.

Each sample consists of:

- one sentence (`sentence`),
- a detected time expression substring (`time_expression_substring` + span),
- a **synthetic reference date** (`reference_year`, `reference_month`, `reference_day`, `reference_weekday`),
- and a **resolved calendar date label** (`time_standard_format`, `time_weekday`),
- optionally, a pattern-type tag (`time_expression_type`).

The core supervision signal is:

> Given the sentence, the time expression substring, and the reference date,  
> predict the correct **calendar date** (`YYYY-MM-DD`) referred to in the sentence  
> (or verify the correctness of `time_standard_format`).

The dataset covers expressions that refer to:

- **a single calendar date**, and  
- **a span of calendar dates**, for which a canonical date is annotated (e.g., start, end, or a salient date).

---

## 2. Data Source & Granularity

- **Underlying data source**  
  Sentences are extracted from publicly available **online news** and **reports**,  
  with a focus on:

  - financial news,
  - macroeconomic commentary,
  - market analysis and related content.

- **Granularity**  
  - The dataset is **sentence-level**.  
  - Each JSON line corresponds to **one sentence**, not a full document.

- **Temporal focus**  
  Sentences are selected and filtered based on whether they **mention or point to**:

  - a **specific calendar date**, or  
  - a **span of calendar dates** (e.g., “from Monday to Friday”, “in the week of November 10”).

Although the raw texts come from whole documents, this dataset only keeps the **sentence** and the necessary **temporal metadata**. Full documents are not included.

---

## 3. Data Schema

Each line in the dataset is a JSON object with the following fields:

    {
      "id": "800009f203e61ef0_2",
      "sentence": "...",
      "time_expression_substring": "...",
      "time_expression_substring_index": [start, end],
      "reference_year": 2002,
      "reference_month": 1,
      "reference_day": 28,
      "reference_weekday": "Monday",
      "time_standard_format": "2002-01-30",
      "time_weekday": "Wednesday",
      "time_expression_type": "pattern_4_3_on_weekday"
    }

### 3.1 `id` (string)

Example:

    "id": "80012869c8b43c51_1"

- The `id` has two parts separated by an underscore (`_`):

  - Prefix (e.g., `80012869c8b43c51`): identifies the **source document**.
  - Suffix (e.g., `1`): a **sentence index** within that document.

- Interpretation:

  - If two samples share the same prefix before `_`, they originate from the **same underlying document**.
  - The numeric suffix denotes the **position of the sentence** in that document  
    (e.g., `_1` = first sentence, `_2` = second sentence, etc.).

---

### 3.2 `sentence` (string)

- The **text sentence** that serves as the main model input.
- All character indices in `time_expression_substring_index` are defined relative to this string.

Example:

    "sentence": "On Wednesday, flash estimates for the inflation rate in Germany revealed a faster-than-expected decline."

---

### 3.3 `time_expression_substring` (string)

- The **exact substring** within `sentence` that expresses a date or time-related concept.

Example:

    "time_expression_substring": "On Wednesday"

The substring is expected to refer to **a calendar date** or **a span of dates**, possibly under-specified (e.g., missing year).

---

### 3.4 `time_expression_substring_index` (list[int])

- A list of two integers `[start, end]` giving the **character-level span** of  
  `time_expression_substring` in `sentence`.

- Indexing convention:

  - 0-based,
  - `end` is **exclusive** (Python-style slicing).

Example:

    "time_expression_substring_index": [0, 12]

This should satisfy:

    sentence[0:12] == time_expression_substring

---

### 3.5 `reference_year` (integer)

- A **randomly sampled year** in the range **[1900, 2050]**.
- It is **not** necessarily:
  - the true publication year of the article, nor
  - any explicit year mentioned in the sentence.

- Purpose:

  - When the sentence does **not explicitly specify the year** of the mentioned date(s),
    `reference_year` offers an **anchor year** for resolving the date.

Example:

    "reference_year": 2002

---

### 3.6 `reference_month` (integer)

- A **randomly sampled month** (1–12).
- Used together with `reference_year` and `reference_day` to construct a **reference date**.

- Purpose:

  - When the time expression lacks an obvious month (e.g., “on the 5th”),
    `reference_month` can be used as contextual information to infer the final date.

Example:

    "reference_month": 1

---

### 3.7 `reference_day` (integer)

- A **randomly sampled day of month**.

Combined with `reference_year` and `reference_month`, it defines the **reference date**:

    reference_date = reference_year - reference_month - reference_day

- Purpose:

  - For relative or under-specified expressions (e.g., “on Wednesday”, “two days later”, “next Friday”),
    `reference_day` helps determine the **most plausible calendar day**.

Example:

    "reference_day": 28

---

### 3.8 `reference_weekday` (string)

- The **weekday name** corresponding to the reference date:

    reference_weekday = weekday(reference_year, reference_month, reference_day)

- Typical values:

  - `"Monday"`, `"Tuesday"`, `"Wednesday"`, `"Thursday"`,
  - `"Friday"`, `"Saturday"`, `"Sunday"`.

- Purpose:

  - An additional discrete signal to assist reasoning.
  - For example, if the sentence says “On Wednesday” but the reference date is a Monday,
    models may infer that the target date is **the Wednesday of that week** (or a nearby week).

Example:

    "reference_weekday": "Monday"

---

### 3.9 `time_standard_format` (string)

- The **resolved calendar date label** associated with the time expression in `sentence`,
  represented in `YYYY-MM-DD` format.

Example:

    "time_standard_format": "2002-01-30"

- Semantics:

  - For **single-date** expressions (e.g., “November 26, 2024”), this is exactly that date.
  - For expressions that cover a **span of dates**, the dataset currently annotates a
    **single canonical date** (e.g., a start date, end date, or the most salient date)
    following internal heuristics.
  - This is the main **target label** that models should predict or verify.

---

### 3.10 `time_weekday` (string)

- The **weekday name** corresponding to `time_standard_format`.

    time_weekday = weekday(time_standard_format)

- Typical values:

  - `"Monday"`, `"Tuesday"`, `"Wednesday"`, `"Thursday"`,
  - `"Friday"`, `"Saturday"`, `"Sunday"`.

- Purpose:

  - Sanity check for the resolved date.
  - Additional supervision for models that may jointly reason over date and weekday.

Example:

    "time_weekday": "Wednesday"

---

### 3.11 `time_expression_type` (string, optional)

- A **categorical label** describing the **pattern family** of the time expression.

Example:

    "time_expression_type": "pattern_4_3_on_weekday"

- Usage:

  - Error analysis per pattern type,
  - Pattern-aware sampling or curriculum learning,
  - Pattern-specific baselines and ablations.

If this field is not present in some files, it can be treated as optional.

---

## 4. Example

A complete example from the dataset:

    {
      "id": "800009f203e61ef0_2",
      "sentence": "On Wednesday, flash estimates for the inflation rate in Germany revealed a faster-than-expected decline.",
      "time_expression_substring": "On Wednesday",
      "time_expression_substring_index": [0, 12],
      "reference_year": 2002,
      "reference_month": 1,
      "reference_day": 28,
      "reference_weekday": "Monday",
      "time_standard_format": "2002-01-30",
      "time_weekday": "Wednesday",
      "time_expression_type": "pattern_4_3_on_weekday"
    }

Interpretation:

- The sentence only mentions **“On Wednesday”** (no explicit date).
- The synthetic reference date is **2002-01-28 (Monday)**.
- The dataset resolves the time expression to **2002-01-30 (Wednesday)**.
- `time_weekday` confirms that the resolved date is indeed a Wednesday.

---

## 5. Suggested File Layout

The actual file layout may vary, but a typical structure could be:

    FinChrono/
    ├── data/
    │   ├── finchronodate_train.jsonl   # training split (sentence-level samples)
    │   ├── finchronodate_dev.jsonl     # development / validation split
    │   └── finchronodate_test.jsonl    # test split
    ├── scripts/
    │   └── examples/                   # example loading / evaluation scripts (optional)
    └── README.md                       # this file

Note: Please refer to the actual content of the `data/` directory in this repository  
for the exact filenames and splits that are currently provided.

---

## 6. Basic Usage

### 6.1 Loading the JSONL files (Python)

    import json

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    train_path = "data/finchronodate_train.jsonl"   # adjust to actual path
    for sample in load_jsonl(train_path):
        sentence = sample["sentence"]
        time_sub = sample["time_expression_substring"]
        label_date = sample["time_standard_format"]
        # your training / evaluation logic here

### 6.2 Typical modeling tasks

- **Calendar date prediction**  
  Input: `sentence`, `time_expression_substring`, `time_expression_substring_index`, and `reference_*`.  
  Output: predicted `YYYY-MM-DD` date, to be compared with `time_standard_format`.

- **Consistency checking**  
  Given a system prediction, check:

  - whether it matches `time_standard_format`, and
  - whether the weekday is consistent with `time_weekday`.

---

## 7. Intended Use

FinChronoDate is intended for research and development on:

- **Time expression parsing and normalization** in financial text,
- **Calendar date reasoning** from under-specified expressions (missing year, month, or day),
- **Temporal reasoning benchmarks** for:

  - large language models,
  - sequence labeling models,
  - multi-task architectures (classification + date prediction).

Possible applications include:

- financial news search and retrieval,
- temporal-aware question answering,
- event extraction pipelines,
- constructing time-aware financial knowledge graphs.

---

## 8. Limitations

- **Synthetic reference dates**

  - `reference_year`, `reference_month`, `reference_day`, and `reference_weekday`
    are **synthetically generated**.
  - They do not necessarily match actual publication dates or real-world timelines.
  - They are designed to serve as **contextual anchors** for temporal reasoning.

- **Single canonical date per sample**

  - Even when the original expression refers to a **span of dates**, the dataset currently
    annotates **one canonical date** in `time_standard_format`.
  - This simplifies the task to **one-date prediction**, but does not fully capture
    all multi-day semantics.

- **Sentence-only context**

  - The dataset only provides **sentence-level context**.
  - Document-level or multi-sentence temporal reasoning is out of scope for this release.

Users should take these limitations into account when designing experiments and interpreting results.

---

## 9. License

A common setup would be:

- **Code** (if scripts are included): MIT License or Apache-2.0.
- **Dataset** (`.jsonl` files):  
  - e.g., **CC BY 4.0** (allows commercial use with attribution), or  
  - **CC BY-NC 4.0** (non-commercial use only).

Please see the `LICENSE` and/or `DATA_LICENSE` files in this repository for the precise terms.

---

## 10. Citation

TBD.
