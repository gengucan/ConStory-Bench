"""
peek_parquet.py

Quick inspector for any parquet file - prints shape, columns, and a
truncated preview of the first row. No VS Code extension needed.

Usage:
    python peek_parquet.py path/to/file.parquet
"""

import sys
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python peek_parquet.py <path_to_parquet>")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_parquet(path)

    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n  " + "\n  ".join(df.columns.tolist()))

    print("\n--- Row 0 preview ---")
    for col in df.columns:
        val = str(df.iloc[0][col])
        preview = val[:300] + ("..." if len(val) > 300 else "")
        print(f"\n[{col}]")
        print(preview)


if __name__ == "__main__":
    main()