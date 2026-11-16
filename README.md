# ComfyUI Anime Low-VRAM Workflows

ComfyUI workflows, a post-processing node and helper scripts for anime image generation with SDXL / Illustrious models on GPUs with 4 GB of VRAM.

Tested on an RTX 2050 Laptop (4 GB VRAM) with 24 GB of system RAM, where a 832x1216 image with a 1.5x hires pass and face detailing takes about 3 minutes.

## Features

- **Low-VRAM hires workflow**: base image, 1.5x upscale and second pass, tiled VAE encode and decode so SDXL fits in 4 GB
- **Face and hand detailer workflow**: redraws small faces and hands at high resolution, fixing broken eyes and fingers in full-body shots
- **Fix faces workflow**: repairs the face and hands of an existing image without changing anything else
- **Fix region workflow**: paint a mask over any broken area (feet, fabric, a missed hand) and redraw only that area
- **Expand image workflow**: outpaints the background to a new aspect ratio, for example a portrait image into a 16:9 wallpaper, keeping the original pixels
- **Pose / composition workflows (ControlNet)**: copy the layout of a reference image, for one character or two, with a depth or pose guide
- **Close-up / wide workflow**: landscape or square portraits and upper-body shots at about 1 megapixel
- **Two-character workflow**: regional prompting with one prompt per character, so features do not bleed between them
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
   | [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Pose workflows (depth and DWPose preprocessors) |

   For comfyui_controlnet_aux, install only the dependencies the pose workflows need (`omegaconf addict yacs` on top of the portable build) rather than its full `requirements.txt`. Its `onnxruntime-gpu` requirement replaces the CPU `onnxruntime` used by the tagger, and mediapipe/trimesh are only needed for preprocessors these workflows do not use.

6. Download the models into `ComfyUI_windows_portable\ComfyUI\models\`:

   | Model | Folder | Source |
   |---|---|---|
   | WAI-Illustrious-SDXL v17 | `checkpoints` | [Civitai](https://civitai.com/models/827184) |
   | ip-adapter-plus_sdxl_vit-h | `ipadapter` | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models) |
   | CLIP-ViT-H-14 image encoder | `clip_vision` | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder), saved as `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` |
   | face_yolov8m.pt | `ultralytics\bbox` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) |
| hand_yolov8s.pt | `ultralytics\bbox` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) |
| ControlNet Union ProMax (SDXL) | `controlnet` | [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), `diffusion_pytorch_model_promax.safetensors` saved as `controlnet-union-sdxl-promax.safetensors` |
| Depth Anything V2 (base) | downloaded by comfyui_controlnet_aux on first use | [depth-anything](https://huggingface.co/depth-anything); the base model is licensed for non-commercial use |
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
| `sdxl_anime_hires_face` | Hires workflow plus face and hand detailers | about 240 s |
| `sdxl_anime_closeup_wide` | Landscape or square close-ups, hires pass, Post FX | about 150 s |
| `sdxl_anime_two_characters` | Two characters with separate regional prompts | about 150 s |
| `sdxl_anime_pose` | Hires face workflow guided by the layout of a reference image | about 240 s |
| `sdxl_anime_pose_two_characters` | Two-character workflow guided by a reference image | about 330 s |
| `fix_faces` | Redraw the face and hands of an existing image | about 80 to 120 s |
| `fix_region` | Redraw an area you paint in the MaskEditor | about 30 s |
| `expand_image` | Extend the background to a new aspect ratio | about 80 s |
| `sdxl_anime_style_ref` | Hires workflow plus IP-Adapter style reference | about 170 s |
| `image_to_tags` | Danbooru tags for any image | a few seconds |

To explore prompts quickly, select the hires and finish groups and press **Ctrl+B** to bypass them. Once a base image looks right, set its seed to fixed and enable them again.

### Two characters

With a single prompt, SDXL models apply every tag to everyone in the image, so one character's hair colour, horns or outfit leaks onto the other. `sdxl_anime_two_characters` splits the prompt:

- **Shared prompt**: scene, style, framing and the count tags (`1girl, 1boy`).
- **Left / right character prompts**: each applies to 65% of the width, overlapping by 30% in the middle so the characters can touch.
- The character prompts switch on at 20% of sampling, after the shared prompt has laid out the composition. Raise this to 0.3 if the characters drift apart; lower it to 0.1 if features still bleed.

Keep each character's distinctive tags only in their own prompt, and use a landscape size. Characters standing directly in front of each other can still mix, since the regions are rectangles.

### Pose and composition reference (ControlNet)

`sdxl_anime_pose` and `sdxl_anime_pose_two_characters` add a reference image to the base pass. Drag any image onto **Pose reference** (a photo, a screenshot, another artwork or one of your renders) and set the latent size to the same aspect ratio. The prompts still decide the characters, outfits, colours and style; the reference decides the layout.

- The default guide is a **depth map** (Depth Anything V2). It captures body shapes, overlaps and who is in front, and works on anime images.
- **DWPose** (stick-figure pose) is included but unconnected. It works on photos of real people and often fails on anime images, where dark clothes and overlapping bodies confuse it. To use it, connect its output to Apply ControlNet and set the union type to `openpose`.
- `strength` 0.7 and `end_percent` 0.7 follow the layout closely while leaving the final steps free. Lower strength to 0.5 for looser results.
- For two characters, keep the left/right order of the reference the same as the Left and Right character prompts. Distinctive features can still bleed where the regions overlap; keep them out of the shared prompt.

ControlNet works best at generation time. Repairing a single hand in a finished image with a hand skeleton is possible, but the redrawn hand tends to be sharper and larger than its surroundings.

### Fix region

For problems no detector finds, such as feet. Right-click the **Image to fix** node, choose **Open in MaskEditor**, paint over the area and save. In the masked area prompt, describe only what belongs there (`barefoot, foot, toes`) plus the original artist tags; the full character prompt can make the model draw a face in the mask. Denoise 0.55 keeps the original shape, 0.65 redraws it with more detail but can add stray pieces.

To remove an object, mask it, describe the background that should replace it (`flower field, no humans`), put the object's name in the negative prompt and use denoise 0.75 to 0.9.

### Expand image

Set the pixels to add on each side in **Pad image**. For a 1248x1824 image, `left 1000, right 1000` gives a 3248x1824 (16:9) wallpaper. Describe only the scenery in the background prompt, without character tags. The new area is generated at about 1.3 megapixels and blended under the untouched original, with `feathering` 200 to hide the seam. The workflow first removes the Post FX vignette (otherwise the dark edges continue into the new area) and re-applies it to the whole new canvas at the end; set both to 0 for images without a vignette.

## Editing tips

- Save a copy before each editing round and branch from it; repairing a repair accumulates artifacts.
- For small exact details (a ring on the wrong finger), move existing pixels and blend them with a light `fix_region` pass (denoise 0.3) instead of asking the model to draw them.
- In removal prompts, describe what should be there, never what you are removing, and put likely substitutes (jewelry, bows, hands) in the negative.
- To make existing structures taller without inventing new ones, copy their own pixels upward as a guide and run `fix_region` over the new area at denoise 0.35.
- Repair upside-down subjects after rotating the image 180 degrees; the model draws anatomy best the right way up.

## Prompting tips

- Illustrious models understand Danbooru tags better than sentences: `1girl, solo, short hair, from below, backlighting, rim lighting`.
- Escape parentheses in tags: `character \(series\)`.
- Artist tags control the drawing style. Mix two or three with weights, for example `(artist a:0.7), (artist b:0.5)`.
- Negative prompts apply to the whole image. Put colours in the positive prompt instead of negating them.
- Keep artist weights at 1.3 or lower. Higher weights start to override character details such as eye shape. To favour one artist, lower the others instead.
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

The pack also contains **Anime Remove Vignette**, the exact inverse of the vignette effect. Use it with the same value before editing or expanding an image that already has a vignette.

## Helper scripts

Run from the `ComfyUI_windows_portable` folder with the bundled Python.

```bat
python_embeded\python.exe -I ..\tools\artist_grid.py example
python_embeded\python.exe -I ..\tools\tag_folder.py "D:\images" "D:\tag_report" 0.35
python_embeded\python.exe -I ..\tools\prepare_lora_dataset.py <images> <tags_per_image.txt> <out_dir> "<trigger>" 2
```

- `artist_grid.py` queues images that share a prompt and seed and differ only in artist tags, then builds a labelled contact sheet. Edit the `ROUNDS` dictionary to define comparisons, and add `--checkpoint <file name>` to run the same round on another checkpoint. ComfyUI must be running.
- `tag_folder.py` tags every image in a folder and writes per-image tags plus a frequency summary. Requires the WD EVA02 tagger model.
- `prepare_lora_dataset.py` crops, resizes and captions images into a kohya-style LoRA training folder.

## License

MIT. See [LICENSE](LICENSE). Models and third-party node packs are covered by their own licenses.
