"""
infer_gender.py — Rule-based gender inference for design_tags.json.

Adds a "gender" field ("Women", "Men", "Unisex") to every entry using
deterministic rules derived from category/product_type/design tags.
No API calls. Run once before deploying the search server.

Usage:
    python infer_gender.py
"""

import json
from pathlib import Path

DESIGN_TAGS_PATH = Path("design_tags.json")


def _fits_intersect(fit_value, target_fits: set) -> bool:
    """Check if a fit field (string or list) overlaps with target_fits."""
    if isinstance(fit_value, list):
        return bool(set(fit_value) & target_fits)
    return fit_value in target_fits


def _neckline_intersect(neckline_value, target: set) -> bool:
    if isinstance(neckline_value, list):
        return bool(set(neckline_value) & target)
    return neckline_value in target


def infer_gender(tags: dict) -> str:
    category     = tags.get("category", "")
    product_type = tags.get("product_type", "")
    design       = tags.get("design")

    # ── Women ──────────────────────────────────────────────────────────────────
    if category == "Bras":
        return "Women"

    if category == "Dresses":
        return "Women"

    if category == "Swimwear" and product_type in {"Bikini", "One-piece swimsuit"}:
        return "Women"

    if category == "Shoes" and product_type == "Heels":
        return "Women"

    if category == "Tops" and product_type in {"Camisole", "Blouse", "Bodysuit"}:
        return "Women"

    if category == "Tops" and isinstance(design, dict):
        neckline = design.get("neckline", [])
        if _neckline_intersect(neckline, {"Halter", "Tube", "Plunge", "Off-the-shoulder", "One shoulder", "Cami"}):
            return "Women"
        if design.get("length") == "Cropped":
            return "Women"

    if category == "Bottoms" and product_type == "Skirt":
        return "Women"

    if category == "Bottoms" and product_type in {"Pants", "Jeans"} and isinstance(design, dict):
        fit = design.get("fit", [])
        if _fits_intersect(fit, {"Mom fit", "Palazzo", "Flare", "Barrel"}):
            return "Women"

    if category == "Bags" and product_type == "Handbag":
        return "Women"

    # ── Men ────────────────────────────────────────────────────────────────────
    if category == "Ties":
        return "Men"

    if category == "Tops" and product_type == "Polo":
        return "Men"

    if category == "Bags" and product_type == "Briefcase":
        return "Men"

    if category == "Bottoms" and product_type in {"Pants", "Jeans"} and isinstance(design, dict):
        fit = design.get("fit", [])
        if _fits_intersect(fit, {"Dad fit", "Carpenter", "Cargo"}):
            return "Men"

    # ── Unisex (default) ───────────────────────────────────────────────────────
    return "Unisex"


def main():
    if not DESIGN_TAGS_PATH.exists():
        raise SystemExit(f"Error: {DESIGN_TAGS_PATH} not found. Run tag_images.py --full first.")

    print(f"Loading {DESIGN_TAGS_PATH} …")
    data = json.loads(DESIGN_TAGS_PATH.read_text(encoding="utf-8"))
    print(f"  {len(data)} entries loaded.")

    counts = {"Women": 0, "Men": 0, "Unisex": 0}
    skipped = 0
    for pid, tags in data.items():
        if not isinstance(tags, dict):
            skipped += 1
            data[pid] = {"gender": "Unisex"}
            counts["Unisex"] += 1
            continue
        gender = infer_gender(tags)
        tags["gender"] = gender
        counts[gender] += 1

    if skipped:
        print(f"  Skipped {skipped} malformed entries (not a dict).")

    DESIGN_TAGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(data)
    print("\nGender distribution:")
    for label, n in counts.items():
        print(f"  {label:<8} {n:>6}  ({n / total * 100:.1f}%)")
    print(f"\nDone. {DESIGN_TAGS_PATH} updated with gender field.")


if __name__ == "__main__":
    main()
