"""Tag every image in a folder with a WD tagger and summarize the results.

Uses the same preprocessing as the ComfyUI WD14 Tagger node. Writes one
line of tags per image to tags_per_image.txt and a frequency table of the
most common tags to tag_summary.txt, both in the chosen output folder.

Usage (from E:\\ComfyUI\\ComfyUI_windows_portable):
    python_embeded\\python.exe -I ..\\tools\\tag_folder.py <image_folder> <output_folder> [threshold]
"""

import collections
import csv
import os
import sys

import numpy as np
from onnxruntime import InferenceSession
from PIL import Image

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ComfyUI_windows_portable",
                                         "ComfyUI", "custom_nodes", "ComfyUI-WD14-Tagger", "models"))
MODEL = "wd-eva02-large-tagger-v3"


def load_tags():
    names, categories = [], []
    with open(os.path.join(MODEL_DIR, MODEL + ".csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names.append(row[1].replace("_", " "))
            categories.append(row[2])
    return names, categories


def prepare(path, size):
    image = Image.open(path).convert("RGB")
    ratio = float(size) / max(image.size)
    new_size = tuple(int(x * ratio) for x in image.size)
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    arr = np.array(square).astype(np.float32)[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(arr, 0)


def main():
    folder, out_dir = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    os.makedirs(out_dir, exist_ok=True)

    session = InferenceSession(os.path.join(MODEL_DIR, MODEL + ".onnx"), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    size = inp.shape[1]
    out_name = session.get_outputs()[0].name
    names, categories = load_tags()

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    general_counts, character_counts = collections.Counter(), collections.Counter()
    lines = []
    for i, f in enumerate(files, 1):
        probs = session.run([out_name], {inp.name: prepare(os.path.join(folder, f), size)})[0][0]
        general = [names[j] for j, p in enumerate(probs) if categories[j] == "0" and p > threshold]
        characters = [names[j] for j, p in enumerate(probs) if categories[j] == "4" and p > 0.85]
        general_counts.update(general)
        character_counts.update(characters)
        lines.append(f"{f}\t{', '.join(characters + general)}")
        if i % 10 == 0 or i == len(files):
            print(f"tagged {i}/{len(files)}", flush=True)

    with open(os.path.join(out_dir, "tags_per_image.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    total = len(files)
    with open(os.path.join(out_dir, "tag_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{total} images, threshold {threshold}\n\nGeneral tags (share of images)\n")
        for tag, n in general_counts.most_common(150):
            fh.write(f"{n / total:6.1%}  {tag}\n")
        fh.write("\nCharacters\n")
        for tag, n in character_counts.most_common(40):
            fh.write(f"{n:4d}  {tag}\n")
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
