<a id="readme-top"></a>
<div align="center">
<h1 align="center">ComfyUI-KEEP</h1>

<img  alt="ComfyUI-KEEP workflow preview" src="https://github.com/user-attachments/assets/bc30d6a3-a1ea-41b4-8278-614070a4c500" />

<p align="center">
ComfyUI custom node pack for <strong>video face super-resolution</strong> — the
<strong>KEEP</strong> algorithm (Kalman-Inspired Feature Propagation) on single images and on frame sequences.
<br />

[![Report Bug][bug-shield]][bug-url] [![Request Feature][feature-shield]][feature-url]

</p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Telegram][telegram-shield]][telegram-url]
[![X][x-shield]][x-url]

</div>

<br>

## ▷ About

**KEEP** (*Kalman-Inspired FEaturE Propagation for Video Face Super-Resolution*, ECCV 2024) restores faces in video frames with a Kalman-style propagation of the features along the **time axis**: the identity of a face stays stable over the whole sequence instead of being detected and synthesized again in every single frame.

This pack ports the algorithm to ComfyUI as three pure **V3** nodes — load the model, restore an image, restore a sequence.

### ❖ Highlights

* **Video-aware** — the sequence node propagates the features over the frames (optical flow), so the faces stay consistent.
* **Any upscaler** — connect any ComfyUI `UPSCALE_MODEL` for the background and a second one for the restored face patch.
* **Four detectors** — RetinaFace (ResNet50 / MobileNet 0.25) and YOLOv5 (n / l).
* **Automatic weights** — the checkpoints are downloaded on the first run and verified against their sha256.
* **Clean V3** — `comfy_api.latest` nodes only, no legacy mappings, no import-time installs, no `sys.path` hacks.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<div align="center">
<img alt="ComfyUI-KEEP example workflow" width="70%" src="https://github.com/user-attachments/assets/62eaf7ea-0ab5-447c-85f7-9952c32b0c47"  />
<p><sub><i>The shipped example workflow: a single image and a video sequence, both restored with KEEP.</i></sub></p>
</div>


<a id="keep-nodes"></a>
## ▓ Nodes

| Node | What it does |
|:---|:---|
| **❖ [Load KEEP Models](#keep-model-loader)** | Loads the KEEP checkpoint, the face detector / parser and the optional upscalers into one model pack. |
| **❖ [KEEP Single Image](#keep-single-image)** | Restores the faces of a single image. |
| **❖ [KEEP Image Sequence](#keep-image-sequence)** | Restores the faces of a frame sequence with the optical-flow propagation. |

<a id="keep-model-loader"></a>
### ❖ Load KEEP Models

Loads the KEEP checkpoint, the face detection / parsing models and the two optional upscale models. The result is the **model pack** that the other two nodes consume.

**Node id:** `KEEP_ModelLoader` · **Output:** `keep_model_pack` (`KEEP_MODEL_PACK`)

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `model` | `KEEP` | The checkpoint: `KEEP` (the general model) or `Asian`. The `Asian` checkpoint is **not published upstream** — see the [weights section](#keep-weights). |
| `detection_model` | `retinaface_resnet50` | The face detector: `retinaface_resnet50`, `retinaface_mobile0.25`, `YOLOv5l`, `YOLOv5n`. |
| `bg_upscale_model` | *(optional)* | A `UPSCALE_MODEL` (from the core *Load Upscale Model* node) for the background of the image. |
| `face_upscale_model` | *(optional)* | A `UPSCALE_MODEL` for the restored face patch. |

</details>

> [!NOTE]
> The pack is loaded **once** and reused by every node of the workflow. It is moved to the GPU only while a node runs and released afterwards.

<a id="keep-single-image"></a>
### ❖ KEEP Single Image

Restores and enhances the faces of a single image.

**Node id:** `KEEP_FaceUpscaleImage` · **Input:** `image` (`IMAGE`) · **Output:** `IMAGE`

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `final_upscale_factor` | `1.0` (0.5 – 4.0, step 0.1) | The final scale of the output image. If your background upscaler is 4× but you want a 2× result, set this to `2.0`; for the original size set it to `1.0`. |
| `has_aligned_face` | `false` | Check it when the input is an already aligned 512×512 face crop. |
| `only_center_face` | `true` | If several faces are detected, restore only the one closest to the center of the image. |
| `draw_bounding_box` | `false` | Draw a box around the detected face on the output image. |

</details>

<a id="keep-image-sequence"></a>
### ❖ KEEP Image Sequence

Restores the faces of a sequence of frames (the frames of a video, for example) — the path that uses the optical flow of KEEP.

**Node id:** `KEEP_ProcessImageSequence` · **Input:** `images` (`IMAGE`) · **Output:** `processed_images` (`IMAGE`)

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `final_upscale_factor` | `1.0` (0.5 – 4.0, step 0.1) | The final scale of the output frames. |
| `has_aligned_frames` | `false` | Check it when the input frames are already aligned 512×512 face crops. |
| `only_center_face` | `true` | If several faces are detected per frame, restore only the one closest to the center. |
| `draw_bounding_box` | `false` | Draw a box around the detected face on the output frames. |
| `max_clip_length` | `20` (1 – 100) | The number of frames the KEEP network processes in one batch — the knob for the VRAM usage of long sequences. |

</details>

> [!TIP]
> The optical flow is only needed for **more than one frame** — a single-frame run never touches the flow weights.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="keep-node-reference"></a>
## ▓ Node Reference

All nodes registered by this pack (V3 schema ids):

| Node id | Display name | Category | Purpose |
|:---|:---|:---|:---|
| `KEEP_ModelLoader` | Load KEEP Models | `ComfyUI-KEEP` | Loads the KEEP checkpoint, the face helper and the optional upscalers into a model pack. |
| `KEEP_FaceUpscaleImage` | KEEP Single Image | `ComfyUI-KEEP` | Restores the faces of a single image. |
| `KEEP_ProcessImageSequence` | KEEP Image Sequence | `ComfyUI-KEEP` | Restores the faces of a frame sequence with the optical-flow propagation. |

The node ids, the display names, the category, the socket types (`IMAGE`, `KEEP_MODEL_PACK`, `UPSCALE_MODEL`) and every widget (name, default, range and order) are compatibility guarantees: workflows saved with the V1 nodes keep working unchanged.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="keep-getting-started"></a>
## ▓ Getting Started

**Via ComfyUI Manager:** Search `ComfyUI-KEEP` → Install.

**Manual install:**

```sh
cd ComfyUI/custom_nodes/
git clone https://github.com/wildminder/ComfyUI-KEEP.git
cd ComfyUI-KEEP
pip install -r requirements.txt
```

Restart ComfyUI. The weights are downloaded automatically on the first run — no manual download step is needed.

### Example workflows

| Workflow | What it shows |
|:---|:---|
| [`example_workflows/keep-workflow.json`](example_workflows/keep-workflow.json) | The shipped example: a single image (with two upscalers) **and** a video sequence. |

> [!IMPORTANT]
> **The shipped example needs the [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) pack.** It uses the node types `VHS_LoadVideo` (reads the video) and `VHS_VideoCombine` (writes the restored video). VideoHelperSuite is an **optional dependency of the example only** — the pack itself does not import it, and the two other example nodes (and every node of this pack) work without it.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="keep-tips"></a>
## ▓ Tips & Best Practices

* **`max_clip_length` is the VRAM knob.** Long sequences are processed in clips of `max_clip_length` frames; lower it (e.g. `8`) on a small GPU, raise it (e.g. `50`) when you have headroom.
* **`final_upscale_factor` is the *final* size.** The upscaler models define their own factor (4×, 2×, …); this value scales the result afterwards — `1.0` keeps the upscaled size.
* **Aligned crops are faster and sharper.** For an already aligned 512×512 face, set `has_aligned_face` / `has_aligned_frames` — the detection step is skipped.
* **`only_center_face` keeps the subject.** With several faces in the frame, the one closest to the center is restored.
* **A failed node fails loudly.** A missing weight or a broken dependency raises an error that names the file and the path — the pack never returns a silently empty image.

<a id="keep-license"></a>
## ▓ License

The **node code** of this pack is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).

> [!WARNING]
> **The KEEP weights and the vendored KEEP code are non-commercial.** The algorithm, its checkpoints and the vendored `basicsr` / `facelib` tree come from the [KEEP project](https://github.com/jnjaby/KEEP) and are distributed under the **S-Lab License 1.0**, which permits redistribution and use **for non-commercial purposes only**. For a commercial use, contact the copyright holders (S-Lab). The node code of this pack carries no such restriction.

The full attribution — the S-Lab License 1.0, the Apache-2.0 `diffusers` compatibility classes and the Apache-2.0 GMFlow package — is in [LICENSE-THIRD-PARTY.md](LICENSE-THIRD-PARTY.md).

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="keep-changelog"></a>
## ▓ Changelog

### v2.0.0 — 2026-10-22

- **Migrated to the pure V3 API.** All three nodes are `comfy_api.latest` nodes now; the legacy `NODE_CLASS_MAPPINGS` are gone. The node ids, the display names, the category, the socket types and every widget (name, default, range and order) are **unchanged** — old workflows keep working.
- **The vendored dependencies were repaired.** The import-time `pip install` of the ffmpeg shim is gone, the broken stubs that silently disabled the KEEP architecture were deleted (a broken dependency now fails loudly instead of producing garbage), and the tree was pruned to the modules the pack really uses.
- **The weights are downloaded from a manifest and verified.** Every checkpoint has a URL, a sha256 and a destination in `src/keep_model_configs.py`; the downloader verifies the hash, re-downloads a corrupted file once and never leaves a partial file behind. The `Asian` checkpoint (not published upstream) is reported as such instead of failing with a download error.
- **The optical flow of the sequence path is wired.** The flownet is restored from the checkpoint when it carries `flownet.*` weights, and resolved from the optional GMFlow checkpoint when it does not.
- **The failures are actionable.** A missing weight names the file, the path and the URL; a foreign model pack is refused with a message; a failed processing raises instead of returning an empty image.
- **The documentation and the examples were rewritten.** This README (node table, weights table, license notice, changelog), the third-party license file and two validated example workflows.

### Before v2.0.0

- **2025-07-03** — Standard ComfyUI `Upscale Model` inputs instead of a hardcoded Real-ESRGAN; added `final_upscale_factor` for precise output scaling.
- **2025-05-20** — Initial commit.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="keep-acknowledgments"></a>
## ▓ Acknowledgments

* **Ruicheng Feng, Chongyi Li and Chen Change Loy** — [KEEP](https://github.com/jnjaby/KEEP) ([paper](https://arxiv.org/abs/2407.21532)) — the algorithm and the checkpoints
* **S-Lab** — the [S-Lab License 1.0](https://github.com/jnjaby/KEEP/blob/main/LICENSE) of the KEEP lineage
* **The HuggingFace team** — [diffusers](https://github.com/huggingface/diffusers) (Apache-2.0), the two compatibility classes the pack vendors
* **Haofei Xu et al.** — [GMFlow](https://github.com/haofeixu/GMFlow) ([paper](https://arxiv.org/abs/2111.13680)) (Apache-2.0), the optical flow
* **The VideoHelperSuite team** — [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), the video nodes of the shipped example
* **The ComfyUI team** — for the platform

### Citation for the original KEEP

If you use the core KEEP algorithm or models in your research, please cite the original paper:

```bibtex
@InProceedings{feng2024keep,
      title     = {Kalman-Inspired FEaturE Propagation for Video Face Super-Resolution},
      author    = {Feng, Ruicheng and Li, Chongyi and Loy, Chen Change},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year      = {2024}
}
```

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<p align="center">══════════════════════════════════</p>

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-KEEP.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-KEEP/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-KEEP.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-KEEP/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-KEEP.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-KEEP/network/members
[bug-shield]: https://img.shields.io/badge/Report-Bug-red?style=flat-square&logo=github
[bug-url]: https://github.com/wildminder/ComfyUI-KEEP/issues/new?labels=bug&template=bug-report---.md
[feature-shield]: https://img.shields.io/badge/Request-Feature-blue?style=flat-square&logo=github
[feature-url]: https://github.com/wildminder/ComfyUI-KEEP/issues/new?labels=enhancement&template=feature-request---.md
[telegram-shield]: https://img.shields.io/badge/Telegram-TokenDiff-26A5E4?style=for-the-badge&logo=telegram&logoColor=white
[telegram-url]: https://t.me/TokenDiff
[x-shield]: https://img.shields.io/badge/X-@wildmindai-000000?style=for-the-badge&logo=x&logoColor=white
[x-url]: https://x.com/wildmindai