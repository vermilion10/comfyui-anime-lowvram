# ComfyUI Anime Low-VRAM Workflows

ComfyUI workflows, a post-processing node and helper scripts for anime image generation with SDXL / Illustrious models on GPUs with 4 GB of VRAM.

Tested on an RTX 2050 Laptop (4 GB VRAM) with 24 GB of system RAM, where a 832x1216 image with a 1.5x hires pass and face detailing takes about 3 minutes.

## Features

- **Low-VRAM hires workflow**: base image, 1.5x upscale and second pass, tiled VAE encode and decode so SDXL fits in 4 GB
- **Face detailer workflow**: redraws small faces at high resolution, fixing broken eyes in full-body shots
- **Fix faces workflow**: repairs the face of an existing image without changing anything else
- **Style reference workflow**: IP-Adapter colour and lighting reference from up to three images
- **Image to tags workflow**: WD EVA02 tagger that turns any image into Danbooru tags
- **Anime Post FX node**: bloom, chromatic aberration, vignette and film grain, processed on the CPU
- **Helper scripts**: artist-tag comparison grids, batch image tagging with tag statistics, and LoRA dataset preparation

## Requirements

- Windows with an NVIDIA GPU (4 GB VRAM or more)
- 16 GB of system RAM or more (24 GB recommended for SDXL)
- About 25 GB of free disk space
- [ComfyUI Windows portable](https://github.com/Comfy-Org/ComfyUI/releases) (NVIDIA build)

## Installation

1. Download and extract the ComfyUI Windows portable build. You get a folder named `ComfyUI_windows_portable`.
2. Copy the contents of this repository next to it:

   ```
   your-folder\
   |-- ComfyUI_windows_portable\
   |-- Start ComfyUI.bat
   `-- tools\
   ```

3. Copy `custom_nodes\anime_postfx` into `ComfyUI_windows_portable\ComfyUI\custom_nodes\`.
4. Copy the files in `workflows\` into `ComfyUI_windows_portable\ComfyUI\user\default\workflows\`.
5. Install the custom node packs used by the workflows, either with ComfyUI-Manager or by cloning them into `custom_nodes`:

   | Node pack | Used by |
   |---|---|
   | [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | Face detailer, fix faces |
   | [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) | Face detector loader |
   | [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | Style reference |
   | [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) | Image to tags |

6. Download the models into `ComfyUI_windows_portable\ComfyUI\models\`:

   | Model | Folder | Source |
   |---|---|---|
   | WAI-Illustrious-SDXL v17 | `checkpoints` | [Civitai](https://civitai.com/models/827184) |
   | ip-adapter-plus_sdxl_vit-h | `ipadapter` | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models) |
   | CLIP-ViT-H-14 image encoder | `clip_vision` | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder), saved as `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` |
   | face_yolov8m.pt | `ultralytics\bbox` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) |
   | wd-eva02-large-tagger-v3 | `custom_nodes\ComfyUI-WD14-Tagger\models` | [SmilingWolf](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) (downloaded by the node on first use) |

   The workflows reference the checkpoint by its published file name. If you use another SDXL anime checkpoint, select it in the Load Checkpoint node.

7. Start ComfyUI with `Start ComfyUI.bat`.

### About the launcher

`Start ComfyUI.bat` starts the portable build with ComfyUI-Manager and live previews enabled, and redirects every cache (Hugging Face, PyTorch, pip, CUDA, Ultralytics) and the temp folder into a `_cache` folder next to it, so nothing is written to the system drive. Extra arguments are passed through, for example `"Start ComfyUI.bat" --lowvram`.

## Workflows

| Workflow | Purpose | Time on 4 GB |
|---|---|---|
| `sdxl_anime_lowvram` | Minimal base image with tiled VAE decode | about 40 s |
| `sdxl_anime_hires_postfx` | Base, 1.5x hires pass, Post FX | about 150 s |
| `sdxl_anime_hires_face` | Hires workflow plus FaceDetailer | about 200 s |
| `fix_faces` | Redraw the face of an existing image | about 40 s |
| `sdxl_anime_style_ref` | Hires workflow plus IP-Adapter style reference | about 170 s |
| `image_to_tags` | Danbooru tags for any image | a few seconds |

To explore prompts quickly, select the hires and finish groups and press **Ctrl+B** to bypass them. Once a base image looks right, set its seed to fixed and enable them again.

## Prompting tips

- Illustrious models understand Danbooru tags better than sentences: `1girl, solo, short hair, from below, backlighting, rim lighting`.
- Escape parentheses in tags: `character \(series\)`.
- Artist tags control the drawing style. Mix two or three with weights, for example `(artist a:0.7), (artist b:0.5)`.
- Negative prompts apply to the whole image. Put colours in the positive prompt instead of negating them.
- Keep `logo, artist name, text` in the negative prompt; some artist tags make the model draw signatures.

## Anime Post FX node

Found under `image/postprocessing`. All effects run on the CPU.

| Input | Effect |
|---|---|
| `bloom_strength`, `bloom_radius`, `bloom_threshold` | Glow around bright areas |
| `chromatic_aberration` | Red and blue lens fringing, in pixels at the image edge |
| `vignette` | Darkening toward the corners |
| `grain` | Film grain intensity |

Set any effect to 0 to disable it.

## Helper scripts

Run from the `ComfyUI_windows_portable` folder with the bundled Python.

```bat
python_embeded\python.exe -I ..\tools\artist_grid.py example
python_embeded\python.exe -I ..\tools\tag_folder.py "D:\images" "D:\tag_report" 0.35
python_embeded\python.exe -I ..\tools\prepare_lora_dataset.py <images> <tags_per_image.txt> <out_dir> "<trigger>" 2
```

- `artist_grid.py` queues images that share a prompt and seed and differ only in artist tags, then builds a labelled contact sheet. Edit the `ROUNDS` dictionary to define comparisons. ComfyUI must be running.
- `tag_folder.py` tags every image in a folder and writes per-image tags plus a frequency summary. Requires the WD EVA02 tagger model.
- `prepare_lora_dataset.py` crops, resizes and captions images into a kohya-style LoRA training folder.

## License

MIT. See [LICENSE](LICENSE). Models and third-party node packs are covered by their own licenses.
