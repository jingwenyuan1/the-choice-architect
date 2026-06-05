"""
Fashion semantic search server  (CLIP ViT-B/32 + FAISS).

Start with:
    python fashion_server.py

Endpoints:
    GET /search?q=<text>&k=<n>   — returns JSON list of matched product objects
    GET /image/<filename>         — serves an image (checks all image directories)
    GET /health                   — returns status, clip availability, and index size

If CLIP fails to load, the server falls back to simple keyword matching.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import faiss
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_DIRS   = [Path(__file__).parent.parent / "fashion-images"]
INDEX_PATH   = Path("fashion_clip.index")
PATHS_PATH   = Path("fashion_clip_paths.npy")
URL_MAP_PATH = Path("url_map.json")
MAX_K        = 500

# ── R2 public base URL ────────────────────────────────────────────────────────
R2_BASE_URL = "https://pub-cac4bbcad35d42c6bdb038e52755c31c.r2.dev"

# ── Load FAISS index ───────────────────────────────────────────────────────────
print("Loading FAISS index …")
index = faiss.read_index(str(INDEX_PATH))
paths = np.load(str(PATHS_PATH), allow_pickle=True)
print(f"Index loaded — {index.ntotal} vectors, {len(paths)} paths.")
# Warm up FAISS: first search triggers internal lazy initialisation.
index.search(np.zeros((1, index.d), dtype=np.float32), 1)
print("[warmup] FAISS ready.")

# ── Load product catalogue ─────────────────────────────────────────────────────
url_map: dict = {}
if URL_MAP_PATH.exists():
    with open(URL_MAP_PATH, "r", encoding="utf-8") as f:
        url_map = json.load(f)
    print(f"Loaded url_map.json — {len(url_map)} entries.")
else:
    print("No url_map.json found — metadata will fall back to filenames.")

# ── Load CLIP ─────────────────────────────────────────────────────────────────
CLIP_LOADED  = False
clip_model   = None
clip_device  = "cpu"

try:
    import torch
    import clip as openai_clip

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {clip_device} …")
    clip_model, _ = openai_clip.load("ViT-B/32", device=clip_device)
    clip_model.eval()
    CLIP_LOADED = True
    print("CLIP loaded successfully.")
    # Warm up CLIP: first encode_text triggers JIT / CUDA kernel compilation.
    # Running it now means the first real search pays no compilation penalty.
    with torch.no_grad():
        clip_model.encode_text(openai_clip.tokenize(["warmup"]).to(clip_device))
    print("[warmup] CLIP inference ready.")
except Exception as exc:
    print(f"CLIP unavailable — keyword fallback active. ({exc})")

print(f"\nServer ready — {index.ntotal} products indexed.\n")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")


# ── Color classifier ──────────────────────────────────────────────────────────

# Garment keywords → which vertical region to sample
_LOWER_TERMS = {"shorts", "pants", "jeans", "skirt", "trousers", "leggings",
                "joggers", "chinos", "culottes", "bermuda", "capris", "sweatpants"}
_UPPER_TERMS = {"top", "shirt", "jacket", "blouse", "sweater", "hoodie",
                "cardigan", "coat", "tee", "polo", "blazer", "vest",
                "tank", "pullover", "camisole", "bra", "crop"}

def _query_region(query: str) -> str:
    words = set(query.lower().split())
    if words & _LOWER_TERMS: return "lower"
    if words & _UPPER_TERMS: return "upper"
    return "full"


def _classify_pixel(r: int, g: int, b: int) -> str:
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    diff = mx - mn

    # Beige/cream/sand: checked before the grey branch so low-diff warm neutrals
    # are not swallowed by grey/white.
    if l > 200 and 10 <= r - b <= 30 and (mx == 0 or diff / mx < 0.15):
        return "beige"

    if diff < 30:
        if l < 60:  return "black"
        if l > 195: return "white"
        return "grey"

    if mx == r:   h = ((g - b) / diff % 6) * 60
    elif mx == g: h = ((b - r) / diff + 2) * 60
    else:         h = ((r - g) / diff + 4) * 60

    if h < 20 or h >= 340: return "red"
    if h < 45:  return "orange"
    if h < 70:  return "yellow"
    if h < 160: return "green"
    if h < 200: return "cyan"
    if h < 260: return "blue"
    if h < 290: return "purple"
    return "pink"


def _compute_color(filename: str, region: str) -> str | None:
    """Fetch image from R2 and compute its dominant garment colour."""
    import io
    import requests as req_lib
    pid = Path(filename).stem
    url = f"{R2_BASE_URL}/{pid}.jpg"
    try:
        resp = req_lib.get(url, timeout=5)
        resp.raise_for_status()
        W, H = 40, 60
        img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((W, H), Image.LANCZOS)
        all_px = list(img.getdata())

        corners = [all_px[0], all_px[W - 1], all_px[(H - 1) * W], all_px[H * W - 1]]
        bg = tuple(sum(c[ch] for c in corners) // 4 for ch in range(3))

        if region == "upper":
            row_pixels = all_px[: int(H * 0.7) * W]
        elif region == "lower":
            row_pixels = all_px[int(H * 0.3) * W :]
        else:
            row_pixels = all_px

        col_start = W // 5
        col_end   = W - W // 5
        h_rows    = len(row_pixels) // W
        pixels = [
            row_pixels[row * W + col]
            for row in range(h_rows)
            for col in range(col_start, col_end)
        ]

        counts: dict[str, int] = {}
        for r, g, b in pixels:
            if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) < 60:
                continue
            if r > 220 and g > 220 and b > 220:
                continue
            name = _classify_pixel(r, g, b)
            counts[name] = counts.get(name, 0) + 1

        if not counts:
            for r, g, b in pixels:
                name = _classify_pixel(r, g, b)
                counts[name] = counts.get(name, 0) + 1

        total_fg = sum(counts.values())
        best     = max(counts, key=lambda k: counts[k])
        return best
    except Exception:
        return None


# ── Persistent thread pool ────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pid(raw_path: str) -> str:
    """Extract a product ID string from a path or URL."""
    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        return raw_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return Path(raw_path).stem


def _image_url(pid: str, raw_path: str) -> str:
    """Resolve the best image URL for a product — always prefer R2."""
    # Always return R2 URL using the product ID as the filename
    return f"{R2_BASE_URL}/{pid}.jpg"


def _build_result(faiss_idx: int, color: str | None = None) -> dict:
    """Turn a FAISS index row into a product dict. Color is pre-computed."""
    raw_path = str(paths[faiss_idx])
    pid      = _pid(raw_path)
    meta     = url_map.get(pid, {})
    return {
        "id":        pid,
        "image_url": _image_url(pid, raw_path),
        "name":      meta.get("name") or f"Item {pid}",
        "price":     meta.get("price"),
        "link":      meta.get("link"),
        "color":     color or meta.get("color"),
    }


def _parallel_colors(indices: list[int], region: str) -> list[str | None]:
    """Detect dominant colour for a list of FAISS indices in parallel."""
    filenames = [Path(str(paths[i])).name for i in indices]
    return list(_executor.map(lambda f: _compute_color(f, region), filenames))


# ── Search strategies ─────────────────────────────────────────────────────────

def _semantic_search(query: str, k: int) -> list[dict]:
    """Encode the query with CLIP and return top-k FAISS nearest neighbours."""
    import clip as openai_clip
    region    = _query_region(query)
    tokens    = openai_clip.tokenize([query]).to(clip_device)
    with __import__("torch").no_grad():
        feat  = clip_model.encode_text(tokens)
    feat      = feat / feat.norm(dim=-1, keepdim=True)
    query_vec = feat.cpu().numpy().astype(np.float32)

    k_actual  = min(k, int(index.ntotal))
    D, I      = index.search(query_vec, k_actual)

    # Deduplicate while preserving FAISS rank order
    valid, seen = [], set()
    for idx in I[0]:
        if idx < 0 or idx >= len(paths): continue
        pid = _pid(str(paths[idx]))
        if pid not in seen:
            seen.add(pid)
            valid.append(int(idx))

    colors = _parallel_colors(valid, region)
    return [_build_result(idx, col) for idx, col in zip(valid, colors)]


def _image_search(image_url: str, k: int, exclude_pid: str | None = None) -> list[dict]:
    """Encode an image with CLIP and return top-k visually similar products."""
    import clip as openai_clip

    # Try to load image from disk first; fall back to R2 (client URL may be localhost)
    filename = image_url.rsplit("/", 1)[-1].split("?")[0]
    img = None
    for d in IMAGE_DIRS:
        p = d / filename
        if p.exists():
            img = Image.open(p).convert("RGB")
            break
    if img is None:
        import requests as _requests
        from io import BytesIO
        r2_url = f"{R2_BASE_URL}/{filename}"
        resp = _requests.get(r2_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")

    import torchvision.transforms as T
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(clip_device)

    with __import__("torch").no_grad():
        feat = clip_model.encode_image(img_tensor)
    feat      = feat / feat.norm(dim=-1, keepdim=True)
    query_vec = feat.cpu().numpy().astype(np.float32)

    k_actual = min(k + 1, int(index.ntotal))
    D, I     = index.search(query_vec, k_actual)

    valid, seen = [], set()
    if exclude_pid:
        seen.add(exclude_pid)
    for idx in I[0]:
        if idx < 0 or idx >= len(paths): continue
        pid = _pid(str(paths[idx]))
        if pid not in seen:
            seen.add(pid)
            valid.append(int(idx))
        if len(valid) >= k:
            break

    colors = _parallel_colors(valid, "full")
    return [_build_result(idx, col) for idx, col in zip(valid, colors)]


def _keyword_search(query: str, k: int) -> list[dict]:
    """Simple term-matching fallback (used when CLIP is unavailable)."""
    region = _query_region(query)
    terms  = query.lower().split()
    valid  = []
    for i, raw_path in enumerate(paths):
        if len(valid) >= k: break
        pid  = _pid(str(raw_path))
        meta = url_map.get(pid, {})
        haystack = " ".join(filter(None, [
            pid, meta.get("name", ""), meta.get("category", ""), meta.get("color", ""),
        ])).lower()
        if all(t in haystack for t in terms):
            valid.append(i)
    colors = _parallel_colors(valid, region)
    return [_build_result(idx, col) for idx, col in zip(valid, colors)]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "indexed": int(index.ntotal),
        "clip":    CLIP_LOADED,
    })


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    k     = min(int(request.args.get("k", 100)), MAX_K)

    if not query:
        return jsonify([])

    if CLIP_LOADED:
        results = _semantic_search(query, k)
    else:
        results = _keyword_search(query.lower(), k)

    return jsonify(results)


@app.route("/search_by_image")
def search_by_image():
    image_url   = request.args.get("url", "").strip()
    k           = min(int(request.args.get("k", 100)), MAX_K)
    exclude_pid = request.args.get("exclude", None) or None

    print(f"\n[search_by_image] HIT")
    print(f"  url     : {image_url!r}")
    print(f"  k       : {k}")
    print(f"  exclude : {exclude_pid!r}")

    if not image_url:
        print("  → rejected: no url")
        return jsonify([])

    if not CLIP_LOADED:
        print("  → rejected: CLIP not loaded")
        return jsonify({"error": "CLIP not available"}), 503

    results = _image_search(image_url, k, exclude_pid)
    print(f"  → returned {len(results)} results")
    if results:
        print(f"  → top 3 ids: {[r['id'] for r in results[:3]]}")
    return jsonify(results)


@app.route("/image/<path:filename>")
def serve_image(filename):
    for d in IMAGE_DIRS:
        img_path = d / filename
        if img_path.exists():
            return send_file(str(img_path))
    return "Not found", 404


# ── Warmup ────────────────────────────────────────────────────────────────────
if CLIP_LOADED and len(paths) > 0:
    print("[warmup] Warming up colour classifier …")
    _warmup_files = [Path(str(p)).name for p in paths[:32]]
    list(_executor.map(lambda f: _compute_color(f, "full"), _warmup_files))
    print("[warmup] Colour classifier ready.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
