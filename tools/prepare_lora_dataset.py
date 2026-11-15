r"""Build a kohya-style LoRA dataset from a folder of images.

Crops a strip from the bottom of each image (to remove signatures), resizes
so the long side is at most MAX_SIDE, and writes one caption .txt per image
using tags from a tag_folder.py report, prefixed with a trigger word.

Usage:
    python_embeded\python.exe -I ..\tools\prepare_lora_dataset.py <images> <tags_per_image.txt> <out_dir> <trigger> [repeats]
"""

import os
import sys
import zipfile

from PIL import Image

CROP_BOTTOM = 0.09
MAX_SIDE = 2048
DROP_TAGS = {"signature", "artist name", "dated", "watermark", "twitter username",
             "web address", "english text", "copyright name", "logo"}


def main():
    src, tag_file, out_root, trigger = sys.argv[1:5]
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    tags = {}
    with open(tag_file, encoding="utf-8") as fh:
        for line in fh:
            name, _, t = line.rstrip("\n").partition("\t")
            tags[name] = [x.strip() for x in t.split(",") if x.strip() and x.strip() not in DROP_TAGS]

    folder = os.path.join(out_root, "train", f"{repeats}_{trigger.replace(' ', '_')}")
    os.makedirs(folder, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(src)):
        if f not in tags:
            continue
        im = Image.open(os.path.join(src, f)).convert("RGB")
        w, h = im.size
        im = im.crop((0, 0, w, int(h * (1 - CROP_BOTTOM))))
        scale = MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        base = os.path.splitext(f)[0]
        im.save(os.path.join(folder, base + ".jpg"), quality=95)
        with open(os.path.join(folder, base + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(", ".join([trigger] + tags[f]))
        count += 1

    zip_path = os.path.join(out_root, "dataset.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(os.path.join(out_root, "train")):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, out_root))
    print(f"{count} images -> {folder}")
    print(f"zip: {zip_path} ({os.path.getsize(zip_path) / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
