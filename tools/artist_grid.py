"""Queue an artist-tag comparison grid on a running ComfyUI server.

Every image in a round shares the same prompt, seed and settings; only the
artist line changes. Results are saved to ComfyUI's output folder under
artist_grid/<round>/ and combined into one labeled contact sheet.

Usage (from E:\\ComfyUI\\ComfyUI_windows_portable):
    python_embeded\\python.exe -I ..\\tools\\artist_grid.py example

Add a new entry to ROUNDS to set up your own comparison. Pass
--checkpoint <file name> to run the same round on another checkpoint.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ComfyUI_windows_portable", "ComfyUI", "output"))
CHECKPOINT = "WAI-Nsfw-Illustrious-17.safetensors"

ULRICH_BASE = "ulrich von hutten \\(azur lane\\), azur lane, 1girl, solo,\n"
ULRICH_OUTFIT = (
    "short hair, black hair, multicolored hair, two-tone hair, white hair, hair over one eye,\n"
    "red horns, mechanical horns, yellow eyes, slit pupils, piercing, ear piercing,\n"
    "black jacket, cross, choker, black thighhighs,\n"
)
SCENE = (
    "masterpiece, best quality, amazing quality, general,\n"
    "upper body, looking at viewer, from below, dutch angle,\n"
    "rooftop, railing, blurry foreground, city, sunset, wind, hair blowing,\n"
    "backlighting, rim lighting, shadow, high contrast, cinematic lighting,\n"
    "depth of field, lens flare, light particles"
)
NEGATIVE = "bad quality, worst quality, worst detail, sketch, censor, nsfw, bad hands, watermark, signature"


ROUNDS = {
    # Replace "artist a" / "artist b" with Danbooru artist tags you want to compare.
    "example": {
        "seed": 123456789,
        "size": (832, 1216),
        "prompt": ULRICH_BASE + ULRICH_OUTFIT + "{artists}" + SCENE,
        "negative": NEGATIVE + ", logo, artist name, text",
        "artists": [
            ("no artist", ""),
            ("artist a", "artist a"),
            ("artist b", "artist b"),
            ("a 0.7 + b 0.5", "(artist a:0.7), (artist b:0.5)"),
        ],
    },
}


def build_prompt(cfg: dict, artists: str, prefix: str, checkpoint: str) -> dict:
    text = cfg["prompt"].format(artists=f"{artists},\n" if artists else "")
    width, height = cfg["size"]
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": cfg["negative"], "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": cfg["seed"], "steps": 25, "cfg": 5.5, "sampler_name": "euler_ancestral",
            "scheduler": "normal", "denoise": 1.0,
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["5", 0], "vae": ["1", 2], "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def wait_for(prompt_id: str) -> str:
    while True:
        hist = json.load(urllib.request.urlopen(f"{SERVER}/history/{prompt_id}"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            if entry["status"]["status_str"] != "success":
                raise RuntimeError(json.dumps(entry["status"])[:1000])
            for out in entry["outputs"].values():
                for img in out.get("images", []):
                    return os.path.join(OUTPUT_DIR, img["subfolder"], img["filename"])
        time.sleep(3)


def contact_sheet(items, path, cols=6, cell_h=640):
    cells = []
    for label, file in items:
        im = Image.open(file).convert("RGB")
        im = im.resize((int(im.width * cell_h / im.height), cell_h), Image.LANCZOS)
        cells.append((label, im))
    cw = cells[0][1].width
    rows = (len(cells) + cols - 1) // cols
    pad, label_h = 8, 30
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (cell_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, im) in enumerate(cells):
        x = pad + (i % cols) * (cw + pad)
        y = pad + (i // cols) * (cell_h + label_h + pad)
        draw.text((x + 4, y + 6), f"{i + 1}. {label}", fill="black", font=font)
        sheet.paste(im, (x, y + label_h))
    sheet.save(path, quality=90)


def main():
    parser = argparse.ArgumentParser(description="Queue an artist-tag comparison grid.")
    parser.add_argument("round", nargs="?", default="round1", help="entry in ROUNDS")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="checkpoint file name; results go to a folder named after it")
    args = parser.parse_args()
    if args.round not in ROUNDS:
        sys.exit(f"Unknown round '{args.round}'. Available: {', '.join(ROUNDS)}")
    cfg = ROUNDS[args.round]
    name = args.round
    if args.checkpoint != CHECKPOINT:
        name += "_" + os.path.splitext(args.checkpoint)[0]

    queued = []
    for i, (label, artists) in enumerate(cfg["artists"], 1):
        prefix = f"artist_grid/{name}/{i:02d}"
        pid = post("/prompt", {"prompt": build_prompt(cfg, artists, prefix, args.checkpoint)})["prompt_id"]
        queued.append((label, pid))
    print(f"{name}: queued {len(queued)} images", flush=True)

    start = time.time()
    results = []
    for label, pid in queued:
        file = wait_for(pid)
        results.append((label, file))
        print(f"done {len(results)}/{len(queued)} {label} ({time.time() - start:.0f}s)", flush=True)

    sheet_path = os.path.join(OUTPUT_DIR, "artist_grid", name, f"{name}_sheet.jpg")
    contact_sheet(results, sheet_path)
    print("sheet", sheet_path, flush=True)


if __name__ == "__main__":
    main()
