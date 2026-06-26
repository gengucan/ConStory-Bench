"""
compute_filtered_ced.py

Computes Consistency Error Density (CED) restricted to specific error
categories - by default, Factual & Detail Consistency and Timeline & Plot
Logic, since that's what most capstone RQs around memory-layer
interventions care about.

This works directly off the judge.py output CSV, because that CSV already
contains one column per subtype (19 total), each holding a JSON array
string of detected error instances. We don't need constory.metrics to
support category filtering - we can just count entries in the relevant
columns ourselves and apply the paper's CED formula:

    CED_i = errors_i / (words_i / 10000)
    CED_overall = mean(CED_i) across stories

Usage:
    python compute_filtered_ced.py output/judge_baseline_poc_0_end_<ts>.csv

Optional: restrict to specific categories (default: factual_detail + timeline_plot)
    python compute_filtered_ced.py <csv> --categories factual_detail timeline_plot
"""

import argparse
import json
import sys

import pandas as pd


# Mirrors the EVALUATION_CRITERIA structure from constory/judge.py
CATEGORY_SUBCRITERIA = {
    "characterization": [
        "memory_contradictions",
        "knowledge_contradictions",
        "skill_power_fluctuations",
        "forgotten_abilities",
    ],
    "factual_detail": [
        "appearance_mismatches",
        "nomenclature_confusions",
        "quantitative_mismatches",
    ],
    "narrative_style": [
        "perspective_confusions",
        "tone_inconsistencies",
        "style_shifts",
    ],
    "timeline_plot": [
        "absolute_time_contradictions",
        "duration_timeline_contradictions",
        "simultaneity_contradictions",
        "causeless_effects",
        "causal_logic_violations",
        "abandoned_plot_elements",
    ],
    "world_building": [
        "core_rules_violations",
        "social_norms_violations",
        "geographical_contradictions",
    ],
}


def count_errors_in_cell(cell_value: str) -> int:
    """
    Each subtype column holds either:
      - a JSON array string like '[{...}, {...}]'  -> count = len(array)
      - '[]'                                         -> count = 0
      - 'ERROR: <message>'                           -> treat as missing/unknown
    Returns the count, or None if the cell represents a failed evaluation
    (so the caller can decide how to handle missing data rather than
    silently treating it as zero).
    """
    if not isinstance(cell_value, str):
        return 0
    text = cell_value.strip()
    if text.startswith("ERROR"):
        return None  # signal: this subtype failed evaluation for this story
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return len(parsed)
        return 0
    except (json.JSONDecodeError, TypeError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to judge output CSV")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["factual_detail", "timeline_plot"],
        choices=list(CATEGORY_SUBCRITERIA.keys()),
        help="Which categories to include in the filtered CED (default: factual_detail timeline_plot)",
    )
    parser.add_argument(
        "--story-column",
        default="generated_story",
        help="Column containing the story text (used for word count)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    target_subcriteria_cols = []
    for cat in args.categories:
        for sc in CATEGORY_SUBCRITERIA[cat]:
            col = f"{cat}_{sc}"
            if col in df.columns:
                target_subcriteria_cols.append(col)
            else:
                print(f"  WARNING: expected column '{col}' not found in CSV - skipping it")

    if not target_subcriteria_cols:
        print("No matching columns found. Check that judge.py finished and wrote the expected columns.")
        sys.exit(1)

    print(f"\nUsing {len(target_subcriteria_cols)} subtype columns across categories: {args.categories}")

    per_story_rows = []
    any_missing_data = False

    for _, row in df.iterrows():
        story_id = row.get("id", row.get("original_id", "unknown"))
        story_text = row.get(args.story_column, "")
        word_count = len(str(story_text).split())

        total_errors = 0
        missing_subtypes = []

        for col in target_subcriteria_cols:
            count = count_errors_in_cell(row[col])
            if count is None:
                missing_subtypes.append(col)
                any_missing_data = True
            else:
                total_errors += count

        ced = (total_errors / (word_count / 10000)) if word_count > 0 else None

        per_story_rows.append({
            "id": story_id,
            "word_count": word_count,
            "filtered_error_count": total_errors,
            "missing_subtypes": ", ".join(missing_subtypes) if missing_subtypes else "",
            "CED": round(ced, 4) if ced is not None else None,
        })

    result_df = pd.DataFrame(per_story_rows)
    print("\n=== Per-story results ===")
    print(result_df.to_string(index=False))

    valid_ced = result_df["CED"].dropna()
    if len(valid_ced) > 0:
        overall_ced = valid_ced.mean()
        print(f"\n=== Overall filtered CED (categories: {', '.join(args.categories)}) ===")
        print(f"Mean CED across {len(valid_ced)} story/stories: {overall_ced:.4f}")
    else:
        print("\nNo valid CED could be computed (all stories had zero word count or all-missing data).")

    if any_missing_data:
        print(
            "\nNOTE: One or more subtype columns contained 'ERROR' entries for at least one story "
            "(likely due to a rate limit or judge failure). Those subtypes were excluded from that "
            "story's error count rather than silently treated as zero. Consider re-running judge.py "
            "on the affected story IDs to fill in missing data before drawing conclusions."
        )

    out_path = args.csv_path.replace(".csv", "_filtered_ced.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed breakdown to: {out_path}")


if __name__ == "__main__":
    main()