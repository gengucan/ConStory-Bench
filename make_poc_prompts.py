"""
make_poc_prompts.py

Downloads the full ConStory-Bench prompts.parquet from Hugging Face and
saves a tiny 1-2 row subsample to data/prompts_poc.parquet for local
pipeline testing.

Run this ONCE from inside your cloned ConStory-Bench repo root, e.g.:

    cd ConStory-Bench
    python make_poc_prompts.py --n 2

This will create: data/prompts_poc.parquet
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="Number of prompts to keep for the POC")
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Optional: filter to a single task type (generation/continuation/expansion/completion) "
             "for a more controlled POC. Leave blank to just take the first N rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out", type=str, default="data/prompts_poc.parquet")
    args = parser.parse_args()

    print("Downloading full prompt set from Hugging Face (jayden8888/ConStory-Bench)...")
    prompts = pd.read_parquet("hf://datasets/jayden8888/ConStory-Bench/prompts.parquet")
    print(f"Full prompt set shape: {prompts.shape}")
    print(f"Columns: {prompts.columns.tolist()}")

    if args.task_type:
        # NOTE: adjust this column name if it differs once you inspect the real schema
        task_col_candidates = [c for c in prompts.columns if "task" in c.lower()]
        if task_col_candidates:
            task_col = task_col_candidates[0]
            prompts = prompts[prompts[task_col].str.lower() == args.task_type.lower()]
            print(f"Filtered to task_type='{args.task_type}' using column '{task_col}': {prompts.shape}")
        else:
            print("WARNING: could not find a task-type column automatically; skipping filter.")

    poc = prompts.sample(n=min(args.n, len(prompts)), random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    poc.to_parquet(args.out)

    print(f"\nSaved {len(poc)} prompt(s) to {args.out}")
    print("\nPreview:")
    for i, row in poc.iterrows():
        print(f"\n--- Prompt {i} ---")
        for col in poc.columns:
            val = str(row[col])
            preview = val[:200] + ("..." if len(val) > 200 else "")
            print(f"  {col}: {preview}")


if __name__ == "__main__":
    main()