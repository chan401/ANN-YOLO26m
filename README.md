# ANN-YOLO26m

Research code and reproducibility materials for single-particle picking in cryo-electron microscopy micrographs. The workflow reads MRC images, normalizes grayscale intensities in float32, detects particles in overlapping tiles, and maps merged particle centers back to the original image coordinates.

| Component | Location |
| --- | --- |
| EMPIAR-10028 training, validation and frozen 100-image test | [experiments/empiar10028](experiments/empiar10028) |
| Seven-source generalization and low-SNR continuation | [experiments/generalization](experiments/generalization) |
| Standalone MRC inference and checkpoint identities | [inference](inference) |
| EMPIAR-10017 historical training and coordinate export | [experiments/empiar10017](experiments/empiar10017) |
| Dataset membership and input fingerprints | [manifests](manifests) |

## Inference

Use Python 3.12 with a CUDA-capable GPU. The benchmark runtime was PyTorch 2.5.1 with CUDA 12.4 and Ultralytics 8.4.8. The archived Ultralytics source and its upstream license are under [runtime](runtime). Run the following from the repository root on Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m zipfile -e runtime/ultralytics_runtime_source.zip runtime/extracted
export PYTHONPATH="$PWD/runtime/extracted${PYTHONPATH:+:$PYTHONPATH}"

python inference/v2/predict_particles.py \
  --config inference/v2/model_configuration.json \
  --weights /path/to/v2/best.pt \
  --input /path/to/micrographs \
  --output picks-v2 --device 0
```

The low-SNR continuation uses the corresponding files under `inference/v3_low_snr/`. The scripts require the checkpoint that matches the selected configuration's SHA-256. TXT results use zero-based MRC array coordinates (x is the column, y is the row); CSV results include both MRC and CryoPPP coordinates.

## Reproducing the experiments

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for input sources, folder layout, commands and experiment boundaries. This release contains code, configurations and manifests. Obtain micrographs and annotations from their archives and provide the appropriate initialization or fitted checkpoint separately. Exact frozen benchmark inference requires the original fitted checkpoints. `SOURCE_SNAPSHOT.json` records the SHA-256 of each file collected from the research server.

The Ultralytics runtime retains its upstream AGPL-3.0 license in [runtime/LICENSE](runtime/LICENSE).
