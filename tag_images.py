"""
tag_images.py — Batch-tag TCA stock images using Claude Haiku 4.5 + Anthropic Batch API.

Usage:
    python tag_images.py --test          # Tag 100 random images → test_tags_100.json
    python tag_images.py --retag-test    # Re-tag the SAME 100 images with updated schema
    python tag_images.py --full          # Tag all 40,000 images → design_tags.json
"""

import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np
import requests
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ── Config ─────────────────────────────────────────────────────────────────────
PATHS_PATH        = Path("fashion_clip_paths.npy")
TEST_OUTPUT_PATH  = Path("test_tags_100.json")
FULL_OUTPUT_PATH  = Path("design_tags.json")
R2_BASE_URL       = "https://pub-cac4bbcad35d42c6bdb038e52755c31c.r2.dev"
MODEL             = "claude-haiku-4-5-20251001"
BATCH_SIZE        = 1_000
POLL_INTERVAL     = 30
DOWNLOAD_WORKERS  = 20   # parallel HTTP downloads from R2 before building the batch
EXTENSIONS_TRIED  = (".jpg", ".jpeg", ".png", ".webp")  # fallback order if .jpg 404s

# ── Prompt ─────────────────────────────────────────────────────────────────────
PROMPT = """Look at this clothing item and reply with ONLY a valid JSON object — no explanation, no markdown.

Follow these rules exactly:

=== TAG 1: category ===
Pick exactly one: Bras, Underwear, Socks, Ties, Tops, Bottoms, Dresses, Outerwear, One-Piece, Swimwear, Shoes, Jewelry, Bags, Headwear, Other accessories, Cold weather accessories, Other

=== TAG 2: product_type ===
Only include if category is Tops, Bottoms, Outerwear, One-Piece, Swimwear, Shoes, Jewelry, Bags, Headwear, Other accessories, or Cold weather accessories.
Omit entirely for Bras, Underwear, Socks, Ties, Dresses, Other.

If Tops: T-shirt, Tank top, Camisole, Blouse, Shirt, Polo, Sweater, Cardigan, Hoodie, Sweatshirt, Bodysuit, Tunic, Vest, Other
If Bottoms: Pants, Jeans, Shorts, Skirt, Leggings, Joggers, Sweatpants, Capris
If Outerwear: Jacket, Coat, Blazer, Trench coat, Puffer, Windbreaker, Vest
If One-Piece: Jumpsuit, Romper, Overalls
If Swimwear: Bikini, One-piece swimsuit, Cover-up
If Shoes: Sneakers, Sandals, Boots, Heels, Flats, Loafers, Oxfords, Clogs, Slippers
If Jewelry: Necklaces, Earrings, Rings, Bracelets, Anklets, Body jewelry, Nose jewelry
If Bags: Handbag, Backpack, Clutch, Duffel bag, Messenger bag, Briefcase
If Headwear: Hair pins, Hair combs, Tiaras, Headbands, Hair clips, Scrunchies, Bandanas, Hats
If Other accessories: Watch, Belt, Suspenders, Sunglasses, Wallet
If Cold weather accessories: Scarf, Shawl, Gloves, Mittens, Earmuffs

=== TAG 3: design ===
Only include when specified below. Omit entirely otherwise.

If Tops → object with all four keys:
  fit (one): Fitted, Relaxed, Oversized
  length (one): Cropped, Waist length, Hip length, Tunic length
  sleeve_length (one): No sleeve, Short sleeve, Long sleeve, Elbow-length sleeve, Cap sleeve, Three-quarter sleeve
  neckline (one or more as array): Crewneck, V-neck, Henley, Cami, Halter, Tube, Button-down, Mock neck, Turtleneck, Square, Scoop, Boat, Plunge, Off-the-shoulder, One shoulder, Other

If Bottoms + Pants or Jeans → object with all three keys:
  fit (one or more as array): Skinny, Slim, Straight, Bootcut, Flare, Wide-leg, Barrel, Boyfriend, Mom fit, Dad fit, Baggy, Tapered, Cigarette, Carpenter, Cargo, Palazzo, Other
  length (one): Cropped, Ankle, Full length
  rise (one): Ultra low rise, Mid rise, High rise, Ultra high rise

If Bottoms + Skirt → array of one or more: A-line, Pencil, Circle, Pleated, Tiered, Tulip, Mermaid, Wrap, Bubble, Other

If Bottoms + Shorts, Leggings, Joggers, Sweatpants, or Capris → OMIT design

If Dresses → object with both keys:
  fit (one or more as array): Slip, Wrap, Bodycon, A-line, Mermaid, Other
  length (one): Micro, Mini, Midi, Maxi, Floor length

If Outerwear → OMIT design
If One-Piece → OMIT design

If Shoes + Sneakers → string (one): Low-top, Mid-top, High-top, Ankle
If Shoes + Boots → string (one): Low-top, Mid-top, High-top, Ankle, Mid-calf, Knee-high, Over-the-knee
If Shoes + Heels → object with both keys:
  heel_height (one): Flat, Low heel, Mid heel, High heel, Ultra-high heel
  heel_type (one): Stiletto, Block, Kitten, Wedge, Cone, Platform
If Shoes + Sandals, Flats, Loafers, Oxfords, Clogs, or Slippers → OMIT design

If Jewelry + Necklaces → array of one or more: Chain, Pendant, Choker, Layered
If Jewelry + Earrings → array of one or more: Stud, Hoop, Drop, Dangle, Huggie, Threader, Ear cuffs, Climbers/crawlers, Chandelier
If Jewelry + Bracelets → array of one or more: Chain, Tennis, Charm, Bangle, Cuff, Beaded, Link
If Jewelry + Anklets → array of one or more: Chain, Beaded, Charm
If Jewelry + Rings, Body jewelry, or Nose jewelry → OMIT design

If Bags + Handbag → string (one): Tote, Crossbody, Shoulder, Satchel, Hobo, Bucket
If Bags + Backpack, Clutch, Duffel bag, Messenger bag, or Briefcase → OMIT design

If Headwear + Hats → string (one): Beanie, Fedora, Baseball cap, Bucket hat
If Headwear + Hair pins, Hair combs, Tiaras, Headbands, Hair clips, Scrunchies, or Bandanas → OMIT design

If Other accessories → OMIT design
If Cold weather accessories → OMIT design

=== TAG 4: color ===
Array of one or more: Red, Orange, Yellow, Green, Blue, Purple, Pink, Black, White, Gray, Beige, Brown, Gold, Silver

=== TAG 5: pattern ===
Exactly one: Solid, Graphic, Floral, Plaid, Checkered, Striped, Polka dot, Animal print, Paisley, Abstract, Geometric, Camouflage

=== TAG 6: occasion ===
Array of one or more: Casual, Semi-Formal, Business Casual, Business Formal, Cocktail, Formal, Black Tie, White Tie, Activewear

=== TAG 7: material ===
Exactly one: linen, denim, knit, lace, leather, satin, sheer, cotton, crochet, velvet, silk, wool, cashmere, polyester, corduroy, other

---

Examples of valid output:

Tops example:
{"category":"Tops","product_type":"Blouse","design":{"fit":"Relaxed","length":"Hip length","sleeve_length":"Long sleeve","neckline":["V-neck"]},"color":["White"],"pattern":"Solid","occasion":["Casual","Semi-Formal"],"material":"linen"}

Bottoms (jeans) example:
{"category":"Bottoms","product_type":"Jeans","design":{"fit":["Straight"],"length":"Ankle","rise":"High rise"},"color":["Blue"],"pattern":"Solid","occasion":["Casual"],"material":"denim"}

Bottoms (skirt) example:
{"category":"Bottoms","product_type":"Skirt","design":["A-line","Pleated"],"color":["Pink","White"],"pattern":"Floral","occasion":["Casual","Semi-Formal"],"material":"cotton"}

Dress example:
{"category":"Dresses","design":{"fit":["A-line"],"length":"Midi"},"color":["Black"],"pattern":"Solid","occasion":["Cocktail","Formal"],"material":"satin"}

Outerwear example:
{"category":"Outerwear","product_type":"Blazer","color":["Gray"],"pattern":"Solid","occasion":["Business Casual","Semi-Formal"],"material":"wool"}

Shoes (heels) example:
{"category":"Shoes","product_type":"Heels","design":{"heel_height":"High heel","heel_type":"Stiletto"},"color":["Black"],"pattern":"Solid","occasion":["Cocktail","Formal"],"material":"leather"}

Bras example:
{"category":"Bras","color":["Beige"],"pattern":"Solid","occasion":["Casual"],"material":"lace"}

Jewelry (earrings) example:
{"category":"Jewelry","product_type":"Earrings","design":["Hoop","Dangle"],"color":["Gold"],"pattern":"Solid","occasion":["Casual","Semi-Formal"],"material":"other"}

Jewelry (rings) example:
{"category":"Jewelry","product_type":"Rings","color":["Silver"],"pattern":"Solid","occasion":["Casual"],"material":"other"}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_pid(raw_path: str) -> str:
    return Path(raw_path).stem


def load_all_pids() -> list[str]:
    paths = np.load(str(PATHS_PATH), allow_pickle=True)
    return [extract_pid(str(p)) for p in paths]


_EXT_TO_MEDIA_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def fetch_image(pid: str) -> tuple[str, str, str] | tuple[str, None, None]:
    """Download an image from R2 and base64-encode it.

    Tries .jpg first (the common case), then falls back through other
    extensions if R2 returns a 404 — covers the case where a file was
    uploaded under a different extension than fashion_clip_paths.npy assumes.

    Returns (pid, base64_data, media_type) on success, or (pid, None, None)
    if every extension 404s / errors.
    """
    for ext in EXTENSIONS_TRIED:
        url = f"{R2_BASE_URL}/{pid}{ext}"
        try:
            resp = requests.get(url, timeout=15)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.content:
            b64 = base64.standard_b64encode(resp.content).decode("utf-8")
            return pid, b64, _EXT_TO_MEDIA_TYPE[ext]
    return pid, None, None


def fetch_images_parallel(pids: list[str]) -> dict[str, tuple[str, str]]:
    """Download+encode many images concurrently. Returns {pid: (b64, media_type)}.

    PIDs that fail to download under any extension are omitted from the
    result and reported separately — these are genuinely missing/broken
    R2 objects, not rate-limit casualties.
    """
    images: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(fetch_image, pid): pid for pid in pids}
        for future in as_completed(futures):
            pid, b64, media_type = future.result()
            if b64 is None:
                missing.append(pid)
            else:
                images[pid] = (b64, media_type)
    if missing:
        print(f"  Warning: {len(missing)} images could not be downloaded from R2 (tried {EXTENSIONS_TRIED})")
        for m in missing[:5]:
            print(f"    missing: {m}")
    return images


def build_request(pid: str, b64_data: str, media_type: str) -> Request:
    return Request(
        custom_id=pid,
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }],
        ),
    )


def submit_batch(client: anthropic.Anthropic, pids: list[str]) -> str:
    print(f"  Downloading {len(pids)} images from R2 …")
    images = fetch_images_parallel(pids)
    requests_list = [
        build_request(pid, b64, media_type)
        for pid, (b64, media_type) in images.items()
    ]
    batch = client.messages.batches.create(requests=requests_list)
    print(f"  Submitted batch {batch.id} ({len(requests_list)} images)")
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch_id}] processing={counts.processing}  "
            f"succeeded={counts.succeeded}  errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            return
        time.sleep(POLL_INTERVAL)


def parse_result_text(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def collect_results(client: anthropic.Anthropic, batch_id: str) -> dict[str, dict]:
    tags: dict[str, dict] = {}
    errors = 0
    error_samples: list[str] = []
    for result in client.messages.batches.results(batch_id):
        pid = result.custom_id
        if result.result.type != "succeeded":
            errors += 1
            if len(error_samples) < 5:
                error_samples.append(f"  {pid}: {result.result.type} — {getattr(result.result, 'error', result.result)}")
            continue
        text = next(
            (b.text for b in result.result.message.content if b.type == "text"), ""
        )
        parsed = parse_result_text(text)
        if parsed is None:
            errors += 1
        else:
            tags[pid] = parsed
    if errors:
        print(f"  Warning: {errors} results skipped (API error or invalid JSON)")
        for s in error_samples:
            print(s)
    return tags


def run_batches(client: anthropic.Anthropic, pids: list[str]) -> dict[str, dict]:
    all_tags: dict[str, dict] = {}
    chunks = [pids[i : i + BATCH_SIZE] for i in range(0, len(pids), BATCH_SIZE)]
    print(f"Total images: {len(pids)} — {len(chunks)} batch(es) of ≤{BATCH_SIZE}")

    for idx, chunk in enumerate(chunks, 1):
        print(f"\nBatch {idx}/{len(chunks)} — {len(chunk)} images")
        batch_id = submit_batch(client, chunk)
        wait_for_batch(client, batch_id)
        chunk_tags = collect_results(client, batch_id)
        all_tags.update(chunk_tags)
        print(f"  Collected {len(chunk_tags)} tags (running total: {len(all_tags)})")

    return all_tags


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch-tag TCA images with Claude Haiku")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test",
        action="store_true",
        help="Tag 100 random images → test_tags_100.json",
    )
    group.add_argument(
        "--retag-test",
        action="store_true",
        dest="retag_test",
        help="Re-tag the SAME 100 images already in test_tags_100.json with updated schema",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Tag all 40,000 images → design_tags.json",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Error: ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    if args.test:
        print("Loading product IDs …")
        all_pids = load_all_pids()
        sample = random.sample(all_pids, 100)
        print(f"\n── TEST MODE: 100 random images ─────────────────────────────")
        tags = run_batches(client, sample)
        TEST_OUTPUT_PATH.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(tags)} tags → {TEST_OUTPUT_PATH}")
        print("Review test_tags_100.json. If ≥85% accurate, run:  python tag_images.py --full")

    elif args.retag_test:
        if not TEST_OUTPUT_PATH.exists():
            raise SystemExit(
                "Error: test_tags_100.json not found. Run --test first to generate it."
            )
        existing = json.loads(TEST_OUTPUT_PATH.read_text())
        pids = list(existing.keys())
        print(f"\n── RETAG TEST MODE: same {len(pids)} images, new schema ──────────")
        tags = run_batches(client, pids)
        TEST_OUTPUT_PATH.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(tags)} tags → {TEST_OUTPUT_PATH} (overwritten with new schema)")
        print("Review test_tags_100.json. If ≥85% accurate, run:  python tag_images.py --full")

    elif args.full:
        print("Loading product IDs …")
        all_pids = load_all_pids()
        existing: dict[str, dict] = {}
        if FULL_OUTPUT_PATH.exists():
            existing = json.loads(FULL_OUTPUT_PATH.read_text())
            print(f"Resuming: {len(existing)} already tagged, skipping those PIDs.")
        remaining = [p for p in all_pids if p not in existing]
        print(f"\n── FULL MODE: {len(remaining)} images remaining ──────────────────")
        new_tags = run_batches(client, remaining)
        all_tags = {**existing, **new_tags}
        FULL_OUTPUT_PATH.write_text(json.dumps(all_tags, indent=2, ensure_ascii=False))
        print(f"\nDone. Saved {len(all_tags)} tags → {FULL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
