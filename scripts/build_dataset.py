#!/usr/bin/env python3
"""
build_dataset.py — Build processed_webdev_coding from real interview data.

Sources (in priority order):
  1. webdev-coding-interview-expanded.json  (4849 entries, all categories)
  2. webdev-coding-interview-real-examples-part1.json  (31 entries, high quality)
  3. webdev-coding-interview.json  (original, checked for non-duplicates)

Output format (Parquet, same as processed_webdev_coding):
  - prompt      : str  — full instruction for the model
  - completion  : str  — ideal response
  - category    : str  — e.g. Express, HTML, JavaScript
  - source      : str  — which source file contributed this entry

Usage:
  cd repositories/opedDev
  conda activate opedDev_py311
  python scripts/build_dataset.py [--val-split 0.1] [--seed 42] [--dry-run]

The script BACKS UP the existing processed_webdev_coding directory before replacing it.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
DATASETS   = REPO_ROOT / "datasets"
OUTPUT_DIR = DATASETS / "processed_webdev_coding"

SOURCES = [
    DATASETS / "webdev-coding-interview-expanded.json",
    DATASETS / "webdev-coding-interview-real-examples-part1.json",
    DATASETS / "webdev-coding-interview.json",
]

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert web development interviewer. "
    "Provide a clear, correct, and concise answer to the following coding question."
)

def build_prompt(question: str, category: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n[{category}] {question}\n"
        "<|assistant|>\n"
    )


def build_completion(entry: dict) -> str:
    """Build the expected model completion from an entry."""
    parts = [entry.get("example_answer", "").strip()]

    # Append correctness note if available (useful training signal)
    correctness = entry.get("correctness", "").strip()
    if correctness:
        parts.append(f"\n\n// Correctness: {correctness}")

    return "\n".join(parts)


def load_source(path: Path, source_name: str) -> list[dict]:
    """Load and normalize a source JSON file into a list of records."""
    if not path.exists():
        print(f"  ⚠️  Not found: {path.name} — skipping")
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  ⚠️  Unexpected format in {path.name} — skipping")
        return []

    records = []
    for entry in data:
        question = entry.get("question", "").strip()
        answer   = entry.get("example_answer", "").strip()
        category = entry.get("category", "General").strip()

        # Skip entries with missing or synthetic content
        if not question or not answer:
            continue
        if "[Synthetic]" in question or "[Answer" in answer:
            continue
        if len(answer) < 20:
            continue

        records.append({
            "prompt":     build_prompt(question, category),
            "completion": build_completion(entry),
            "category":   category,
            "source":     source_name,
        })

    print(f"  ✅ {path.name}: {len(records)} valid entries loaded")
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicates by question prompt (keep first occurrence)."""
    seen = set()
    deduped = []
    for r in records:
        key = r["prompt"][:200]  # first 200 chars as fingerprint
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    removed = len(records) - len(deduped)
    if removed:
        print(f"  🗑️  Removed {removed} duplicate entries")
    return deduped


def backup_existing(output_dir: Path) -> None:
    """Backup the existing output directory with a timestamp."""
    if output_dir.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup = output_dir.parent / f"{output_dir.name}.bak.{ts}"
        shutil.copytree(output_dir, backup)
        print(f"  📦 Backed up existing dataset to {backup.name}")


def save_split(df: pd.DataFrame, path: Path, split_name: str) -> None:
    """Save a DataFrame split as Parquet."""
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "data.parquet"
    df.to_parquet(out_file, index=False)
    print(f"  💾 {split_name}: {len(df)} samples → {out_file.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Build processed_webdev_coding from real data.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction (default: 0.1)")
    parser.add_argument("--seed",      type=int,   default=42,  help="Random seed")
    parser.add_argument("--dry-run",   action="store_true",     help="Print stats without writing files")
    args = parser.parse_args()

    print("\n🏗️  Building opedDev dataset from real content\n")

    # 1. Load all sources
    all_records = []
    for path in SOURCES:
        records = load_source(path, path.stem)
        all_records.extend(records)

    print(f"\n  Total raw records: {len(all_records)}")

    # 2. Deduplicate
    records = deduplicate(all_records)
    print(f"  Total after dedup: {len(records)}")

    # 3. Check category distribution
    df = pd.DataFrame(records)
    print("\n  Category distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat:30s}: {count}")

    if args.dry_run:
        print("\n  [dry-run] No files written.")
        return

    # 4. Backup existing and split
    backup_existing(OUTPUT_DIR)

    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed, stratify=df["category"]
    )

    print(f"\n  Train: {len(train_df)}  |  Val: {len(val_df)}")

    # 5. Save as HuggingFace DatasetDict (Arrow format, load_from_disk compatible)
    from datasets import Dataset, DatasetDict
    train_hf = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_hf   = Dataset.from_pandas(val_df.reset_index(drop=True))
    ds = DatasetDict({"train": train_hf, "test": val_hf})
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"  💾 train: {len(train_hf)} samples")
    print(f"  💾 test:  {len(val_hf)} samples")
    print(f"  📁 saved to {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # 6. Write dataset info
    info = {
        "description": "Web development coding interview dataset (real content)",
        "num_train": len(train_df),
        "num_test": len(val_df),
        "categories": df["category"].value_counts().to_dict(),
        "sources": df["source"].value_counts().to_dict(),
        "built_at": datetime.now().isoformat(),
        "prompt_format": "Qwen chat template (<|system|> / <|user|> / <|assistant|>)",
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Dataset saved to {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print("   Run training with: python scripts/train_model.py")


if __name__ == "__main__":
    main()
