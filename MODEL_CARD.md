# ANN-YOLO26m LowSNR model card

## Checkpoint and training

The deployed checkpoint is `ANN-YOLO26m-LowSNR.pt`, SHA-256 `429e2fac85ea0140adce1325ae638b3d16142f192e63418a69aa6177de75f1bd`. Both HCN1 and TRPV1 use this same checkpoint and global threshold 0.22.

The model continues training from the selected seven-source checkpoint using source-only low-SNR augmentation. The continuation runs 20 epochs at learning rate 5e-5. The augmentation reduces contrast and adds noise before regenerating the multi-scale channels. Its source-validation macro F1 is 0.733290859969671. See the [protocol](experiments/generalization/v3_low_snr/protocol.json), [training configuration](experiments/generalization/v3_low_snr/low_snr20_training_config.json) and [freeze](experiments/generalization/v3_low_snr/final_frozen_config.json).

## Inference

Use `inference/v3_low_snr/predict_particles.py` and its matching configuration. Input is normalized per micrograph; each native-scale tile supplies LP4/LP2/raw grayscale channels. Settings are tile size 1536, overlap 256, padding 128, center suppression radius 28 native pixels, `max_det=2000`, candidate score floor `1e-5`, `end2end=True`, `iou=0.7` and FP32 inference. Inference requires no particle labels.

EMPIAR-10059 mode-0 data uses the recorded unsigned-byte interpretation via `--unsigned-mode0`. Other inputs must follow their actual storage interpretation.

## Evaluation

HCN1 contains 20 evaluation micrographs and 2328 GT particles; TRPV1 contains 20 and 13082 GT particles. The v3 [image registry](experiments/generalization/v3_low_snr/image_registry.json) records HCN1 as `regression_check` and TRPV1 as `final_check`. Center matching is maximum-cardinality one-to-one with native-pixel radii 38.5 and 33 respectively.

| Dataset | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| HCN1 | 2103 | 691 | 225 | 0.7527 | 0.9034 | 0.8212 |
| TRPV1 | 4971 | 1038 | 8111 | 0.8273 | 0.3800 | 0.5208 |

The same-v3 HCN1 evaluation is documented by its [regression protocol](experiments/generalization/v3_low_snr/secondary_HCN1_protocol.json). The source-selected checkpoint and threshold are shared across both target sets.

The separate 2026-11-16 forward verification gives TP/FP/FN=2104/691/224 on HCN1 and 4972/1037/8110 on TRPV1, with F1=0.8214 and 0.5209. These one-match differences are retained separately; their cause has not been isolated. See [verification records](results/verification_20261116/README.md).

## Scope

Earlier target-domain observations informed development; these are known benchmark checks rather than a pristine unknown-domain blind test. The historical paired bootstrap interval for TRPV1 against crYOLO includes zero. A higher F1 point estimate therefore does not establish stable or statistically significant superiority. TRPV1 recall remains limited. Additional independent data are needed to characterize broader use.
