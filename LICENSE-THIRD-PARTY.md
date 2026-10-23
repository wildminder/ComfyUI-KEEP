# Third-party licenses

The **node code** of ComfyUI-KEEP is licensed under the Apache License 2.0 — see
[LICENSE](LICENSE). This file documents the **third-party code** the pack
redistributes and the **weights** it downloads.

| Component | License | Where |
|:---|:---|:---|
| KEEP algorithm, checkpoints and the vendored `basicsr` / `facelib` tree | **S-Lab License 1.0 — non-commercial** | `modules/deps/basicsr/`, `modules/deps/facelib/`, the `models/keep_models/KEEP` weights |
| `diffusers` compatibility classes (`FeedForward`, `AdaLayerNorm`) | Apache-2.0 | `modules/deps/basicsr/archs/_diffusers_compat.py` |
| GMFlow (optical flow) | Apache-2.0 | `modules/deps/basicsr/archs/gmflow/` |
| The face detection / parsing weights of the facelib helper | S-Lab License 1.0 — non-commercial | `models/facedetection/` |
| The GMFlow checkpoint `gmflow_sintel-0c07dcb3.pth` | Apache-2.0 | `models/gmflow/` |

---

## 1. KEEP — S-Lab License 1.0 (non-commercial)

The KEEP algorithm, its checkpoints and the vendored `basicsr` / `facelib`
tree come from the [KEEP project](https://github.com/jnjaby/KEEP)
(ECCV 2024, *Kalman-Inspired FEaturE Propagation for Video Face
Super-Resolution*) and are distributed under the **S-Lab License 1.0**.

> **Non-commercial notice.** The S-Lab License 1.0 permits redistribution and
> use **for non-commercial purposes only**. The KEEP weights, the face
> detection / parsing weights and the vendored KEEP code in this repository
> inherit that restriction. For a commercial use, contact the copyright
> holders (S-Lab) — the upstream license asks for exactly that.
>
> The **node code** of this pack (everything outside `modules/deps/`) is
> Apache-2.0 and carries no such restriction.

The full text of the license is the `LICENSE` file of the upstream project:
<https://github.com/jnjaby/KEEP/blob/main/LICENSE>. In short, it allows
redistribution and use in source and binary form, with or without
modification, **for non-commercial purposes**, provided the copyright notice,
the conditions and the disclaimer are retained — and asks to contact the
contributors when a commercial use is required.

The weights listed in the weights table of the [README](README.md) are
downloaded from the [KEEP v1.0.0 release](https://github.com/jnjaby/KEEP/releases/tag/v1.0.0)
and keep the license of their origin.

## 2. `diffusers` compatibility classes — Apache-2.0

`modules/deps/basicsr/archs/_diffusers_compat.py` contains **verbatim copies**
of two classes of [huggingface/diffusers](https://github.com/huggingface/diffusers)
version **0.11.0** — `FeedForward` and `AdaLayerNorm` (the two symbols the
KEEP architecture uses from `diffusers.models.attention`).

> Copyright 2022 The HuggingFace Team. All rights reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy
> of the License at <http://www.apache.org/licenses/LICENSE-2.0>
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations
> under the License.

The copies are needed because `diffusers` cannot be imported in the supported
environments (Python 3.13 / torch 2.11); the equivalence with the upstream
0.11.0 classes is recorded in
`docs/audit/2026-10-22-diffusers-equivalence.md`.

## 3. GMFlow — Apache-2.0

`modules/deps/basicsr/archs/gmflow/` is the vendored
[GMFlow](https://github.com/haofeixu/GMFlow) package (*Learning Optical Flow
via Global Matching*, CVPR 2022,
[arXiv:2111.13680](https://arxiv.org/abs/2111.13680)). The upstream license
travels with the code — the full Apache-2.0 text is in
`modules/deps/basicsr/archs/gmflow/LICENSE`:

> Copyright 2022, Haofei Xu
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy
> of the License at <http://www.apache.org/licenses/LICENSE-2.0>
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations
> under the License.

The GMFlow checkpoint (`gmflow_sintel-0c07dcb3.pth`, listed in the weights
table of the [README](README.md)) is the Sintel model of the same project and
is under the same Apache-2.0 license. It is **defensive**: the pack only
resolves it when a loaded checkpoint does not carry its own `flownet.*`
weights.

## 4. The upscalers and the video helper

The pack does **not** redistribute any upscaler model: the `UPSCALE_MODEL`
inputs are fed by the user from the ComfyUI core *Load Upscale Model* node
(the models in `models/upscale_models/` keep their own licenses). The shipped
example workflow uses the nodes of the
[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
pack, which is installed separately and keeps its own license.

---

*This file is a summary, not a replacement: the license texts of the upstream
projects are the binding ones and travel with the code and the weights.*
