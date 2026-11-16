# LowSNR forward verification and historical audit

These records distinguish a historical prediction audit from a new forward pass on known benchmark micrographs. They do not report a newly trained model or a new domain-blind evaluation.

| File | Meaning |
| --- | --- |
| `historical_prediction_audit.json` | Matching metrics recomputed from historical predictions; prediction hashes and the shared weight hash were checked. |
| `forward_verification.json` | New 40-image forward result using the historical checkpoint and threshold 0.22; includes the inference contract. |
| `metrics.csv` | Fresh ANN counts and historical pretrained-baseline counts on the same images, GT and baseline settings. |
| `standalone_check.json` | Independent entry point checked on one image per target; retained counts match the full forward run. All 40 actual MRC and GT hashes were checked. |

The [v3 registry](../../experiments/generalization/v3_low_snr/image_registry.json) contains TRPV1 `final_check` and HCN1 `regression_check` membership: 20 images per target, 13082 and 2328 GT particles respectively. Matching is maximum-cardinality one-to-one center matching at 33 and 38.5 native pixels. Baseline prediction and model hashes were verified and baseline counts reused; baseline forward inference was not rerun on 2026-11-16.

Historical ANN counts remain HCN1 2103/691/225 and TRPV1 4971/1038/8111 (TP/FP/FN). Fresh counts are 2104/691/224 and 4972/1037/8110. The causes of these one-match differences have not been isolated.

The separate A3 retraining (`f65067ce2a1f7cef0fc44175c7ab259c39a0fe8ac41e575d9142f54f0c6855b1`) did not exceed crYOLO: F1=0.7689169895982052 on HCN1 and 0.40437031440692744 on TRPV1 at its source-selected threshold 0.20. This is not the LowSNR checkpoint verified here. See [model provenance](../../MODEL_PROVENANCE.md).
