# ANN-YOLO26m

ANN-YOLO26m picks single-particle centers from cryo-EM MRC micrographs. It applies robust grayscale normalization, predicts on overlapping tiles, and merges detections in the original image coordinate system.

## 1. Install

Use Linux, Python 3.12, and a CUDA-capable NVIDIA GPU. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m zipfile -e runtime/ultralytics_runtime_source.zip runtime/extracted
export PYTHONPATH="$PWD/runtime/extracted${PYTHONPATH:+:$PYTHONPATH}"
```

## 2. Download the verified single-checkpoint model

[Download ANN-YOLO26m-LowSNR.pt](https://github.com/chan401/ANN-YOLO26m/releases/download/paper-submission-v1.1/ANN-YOLO26m-LowSNR.pt). This is the historical v3 LowSNR model verified on **both HCN1 and TRPV1 using the same checkpoint and global threshold 0.22**. SHA-256:

```text
429e2fac85ea0140adce1325ae638b3d16142f192e63418a69aa6177de75f1bd
```

| Record | HCN1 F1 | TRPV1 F1 |
| --- | --- | --- |
| Historical LowSNR predictions | 0.8212 | 0.5208 |
| Fresh forward verification, 2026-11-16 | 0.8214 | 0.5209 |
| crYOLO pretrained baseline, same micrographs | 0.7756 | 0.5151 |
| Topaz pretrained baseline, same micrographs | 0.5407 | 0.1750 |

The fresh run is a separate verification of the historical model, not a newly trained improvement. It differs by one matched particle per dataset; historical paper counts remain unchanged. These are known benchmarks, not a new domain-blind evaluation. The small TRPV1 difference does not establish statistically significant superiority. See [model provenance](MODEL_PROVENANCE.md) and [verification records](results/verification_20261116/README.md).

## 3. Run particle picking

Save the downloaded checkpoint in the repository root or update the path passed to `--weights`.

```bash
python inference/v3_low_snr/predict_particles.py \
  --config inference/v3_low_snr/model_configuration.json \
  --weights ANN-YOLO26m-LowSNR.pt \
  --input /path/to/micrographs \
  --output picks --device 0
```

For the EMPIAR-10059/TRPV1 mode-0 micrographs used here, add `--unsigned-mode0`, following their recorded storage interpretation. This is a data-decoding option, not a target-specific weight or threshold; do not apply it indiscriminately to other datasets.

The input can be an MRC file or a directory of MRC micrographs. Output includes per-image TXT coordinates and a combined predictions CSV. TXT coordinates are zero-based in the original MRC array (x is column, y is row). The CSV also includes CryoPPP coordinates and scores. The output directory must be empty. Inference requires no GT and verifies the checkpoint SHA-256 before loading. Use the archived runtime; newer Ultralytics defaults are not equivalent to this frozen setup.

## 4. Other checkpoints and paper reproduction

`ANN-YOLO26m-General.pt` remains the distinct historical v2 checkpoint, paired with `inference/v2/model_configuration.json` and SHA-256 `92776d3c11fd1157a012fac6f3c38ba280c8e077b22a18c9c67176cd31703679`. Do not substitute it for LowSNR when attributing the two results above.

The [original paper-submission-v1 snapshot](https://github.com/chan401/ANN-YOLO26m/releases/tag/paper-submission-v1) is preserved. Use the [corrected paper-submission-v1.1 release](https://github.com/chan401/ANN-YOLO26m/releases/tag/paper-submission-v1.1) for the corrected model mapping and explicit inference flags. The paper's EMPIAR-10028 comparison uses a separate dataset-specific checkpoint and the [frozen 100-micrograph test set](manifests/empiar10028/test_100_v1/test.txt). See [Reproducibility](REPRODUCIBILITY.md). The [runtime license](runtime/LICENSE) applies to the archived Ultralytics runtime.
