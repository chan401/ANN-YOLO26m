# Model identity and correction of result attribution

## Correction dated 2026-11-16

The earlier guide directed readers to General/v2 for HCN1 and LowSNR/v3 for TRPV1. This omitted the **HCN1 regression check using the same LowSNR checkpoint**. Both historical F1 values, **0.8212 on HCN1 and 0.5208 on TRPV1**, can be recomputed from one LowSNR weight and one global threshold, **0.22**.

Use `ANN-YOLO26m-LowSNR.pt`, SHA-256:

```text
429e2fac85ea0140adce1325ae638b3d16142f192e63418a69aa6177de75f1bd
```

This is a historical trained model, not the later A3 retraining. It initializes from selected v2 weights and continues training on seven source datasets with source-only low-SNR augmentation. Its historical source-validation macro F1 is **0.733290859969671**; **0.7341986413334569** belongs to the preceding v2 selection. See the preserved [protocol](experiments/generalization/v3_low_snr/protocol.json), [training configuration](experiments/generalization/v3_low_snr/low_snr20_training_config.json), [freeze](experiments/generalization/v3_low_snr/final_frozen_config.json) and [HCN1 regression protocol](experiments/generalization/v3_low_snr/secondary_HCN1_protocol.json).

## Historical prediction evidence

Both completion records identify checkpoint `429e2fac...`:

| Dataset / role | Historical archive relative to server experiment root | Prediction SHA-256 | TP / FP / FN | F1 |
| --- | --- | --- | --- | --- |
| HCN1 / secondary regression | `v3_low_snr/results/ANN_v3_HCN1_regression/predictions.csv` | `fedacfaca3d45e5eda73f8de2aba9d69b888c2f6a1d1b558401867a21e624136` | 2103 / 691 / 225 | 0.8211636079656385 |
| TRPV1 / final check | `v3_low_snr/results/ANN_v3/predictions.csv` | `7bbc5a80cf0e1882d4fa3004231b9d0193b433203b2a8891ce2376f9c7b4a9c8` | 4971 / 1038 / 8111 | 0.5207689487192918 |

Archive paths describe server records, not files downloadable from this checkout. The [audit](results/verification_20261116/historical_prediction_audit.json) verifies archive and weight hashes and recomputes matching metrics. That audit is distinct from a new forward pass.

## Fresh forward verification on 2026-11-16

A separate forward pass used the same LowSNR checkpoint and 0.22 threshold on the same 20 HCN1 and 20 TRPV1 micrographs, without retraining or target-threshold scanning. All actual MRC and GT hashes were checked. Both use native LP4/LP2/raw input, 1536-pixel tiles, 256-pixel overlap, 128-pixel padding, 28-pixel center suppression, `max_det=2000`, score floor `1e-5`, `end2end=True`, `iou=0.7` and FP32 inference.

| Dataset | GT | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HCN1 | 2328 | 2104 | 691 | 224 | 0.7527728085867621 | 0.9037800687285223 | 0.8213937146203397 |
| TRPV1 | 13082 | 4972 | 1037 | 8110 | 0.8274255283741055 | 0.38006421036538757 | 0.5208737101251899 |

The one-match differences from historical counts have not been attributed to a specific cause. Records are kept separately; verification does not overwrite paper counts. See [full metrics and checks](results/verification_20261116/README.md).

## Scope and retained alternatives

The LowSNR F1 point estimates exceed both pretrained baselines on these two fixed sets. The historical paired bootstrap interval for TRPV1 against crYOLO includes zero, so the point estimate alone does not establish stable or statistically significant superiority. Prior target-domain observations informed development; HCN1 is a regression check. Neither this recheck nor the historical experiment is a pristine unknown-domain blind test.

The later A3 retraining produced another checkpoint, `f65067ce2a1f7cef0fc44175c7ab259c39a0fe8ac41e575d9142f54f0c6855b1`, with source-selected threshold 0.20. Its HCN1/TRPV1 F1 scores were 0.7689169895982052 and 0.40437031440692744, below crYOLO. It is not renamed or substituted for LowSNR, and its outcome remains documented.

General/v2 (`92776d3c...`) remains a distinct archived model with its original configuration. This correction does not rewrite weight bytes, old tags, source snapshot hashes or frozen historical experiment files.
