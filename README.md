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

## 2. Download the pretrained model

[Download ANN-YOLO26m-General.pt](https://github.com/chan401/ANN-YOLO26m/releases/download/paper-submission-v1/ANN-YOLO26m-General.pt), the recommended checkpoint for general particle picking. It was trained on seven source datasets. The inference script checks its SHA-256 against the included configuration.

## 3. Run particle picking

Save the downloaded checkpoint in the repository root or update the path passed to `--weights`.

```bash
python inference/v2/predict_particles.py \
  --config inference/v2/model_configuration.json \
  --weights ANN-YOLO26m-General.pt \
  --input /path/to/micrographs \
  --output picks --device 0
```

The input can be an MRC file or a directory of MRC micrographs. Output includes per-image TXT coordinates and a combined predictions CSV. TXT coordinates are zero-based in the original MRC array (x is column, y is row). The CSV also includes CryoPPP coordinates and scores.

For citation, use the [versioned paper submission release](https://github.com/chan401/ANN-YOLO26m/releases/tag/paper-submission-v1). The paper's EMPIAR-10028 comparison uses the [frozen 100-micrograph test set](manifests/empiar10028/test_100_v1/test.txt). See [Reproducibility](REPRODUCIBILITY.md) for the experiment settings and evaluation workflow. The [runtime license](runtime/LICENSE) applies to the archived Ultralytics runtime.
