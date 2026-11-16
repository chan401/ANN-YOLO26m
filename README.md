# ANN-YOLO26m

ANN-YOLO26m picks single-particle centers from cryo-EM MRC micrographs using robust floating-point grayscale normalization, overlapping native-scale tiles, multi-scale grayscale input and coordinate recovery.

## Pretrained model

Use **ANN-YOLO26m-LowSNR.pt** with `inference/v3_low_snr/model_configuration.json`. HCN1 and TRPV1 use the **same weight and global threshold 0.22**.

[Download the pretrained weight](https://github.com/chan401/ANN-YOLO26m/releases/download/paper-submission-v1/ANN-YOLO26m-LowSNR.pt)

SHA-256:

```text
429e2fac85ea0140adce1325ae638b3d16142f192e63418a69aa6177de75f1bd
```

| Dataset | ANN-YOLO26m F1 | crYOLO F1 | Topaz F1 |
| --- | --- | --- | --- |
| HCN1 | 0.8212 | 0.7756 | 0.5407 |
| TRPV1 | 0.5208 | 0.5151 | 0.1750 |

The table reports the paper evaluation. The separate 2026-11-16 verification gives ANN F1=0.8214 and 0.5209; [verification records](results/verification_20261116/README.md) retain those counts separately. These are known benchmark results; the small TRPV1 difference does not establish statistically significant superiority or guarantee performance on other samples.

## Install

Use Linux, Python 3.12 and a CUDA-capable NVIDIA GPU. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m zipfile -e runtime/ultralytics_runtime_source.zip runtime/extracted
export PYTHONPATH="$PWD/runtime/extracted${PYTHONPATH:+:$PYTHONPATH}"
```

Use the archived runtime with this checkpoint; newer Ultralytics defaults are not equivalent to the recorded configuration.

## Run particle picking

Save the downloaded weight in the repository root:

```bash
python inference/v3_low_snr/predict_particles.py \
  --config inference/v3_low_snr/model_configuration.json \
  --weights ANN-YOLO26m-LowSNR.pt \
  --input /path/to/micrographs \
  --output picks --device 0
```

For the EMPIAR-10059/TRPV1 mode-0 micrographs used in the evaluation, add `--unsigned-mode0`, following their storage interpretation. This is a data-decoding option, not a separate weight or threshold. Do not apply it indiscriminately to other datasets.

Input may be an MRC file or a directory. Output contains per-image TXT coordinates and a combined CSV. TXT coordinates are zero-based in the original MRC array: x is column, y is row. CSV also includes CryoPPP coordinates and scores. The output directory must be empty. Inference requires no GT and verifies the weight SHA-256 before loading.

See the [model card](MODEL_CARD.md) for input settings and evaluation scope, [Reproducibility](REPRODUCIBILITY.md) for experiment configurations, and the [paper release](https://github.com/chan401/ANN-YOLO26m/releases/tag/paper-submission-v1) for downloads. The EMPIAR-10028 comparison uses its own dataset-specific checkpoint and [100-image test set](manifests/empiar10028/test_100_v1/test.txt). The [runtime license](runtime/LICENSE) applies to the archived Ultralytics source.
