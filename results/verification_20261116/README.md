# Forward verification on 2026-11-16

This evaluation applies the single LowSNR checkpoint and threshold 0.22 to the same 20 HCN1 and 20 TRPV1 micrographs. No weight updates or threshold scans are performed for the verification.

| File | Content |
| --- | --- |
| `metrics.csv` | Fresh ANN counts and same-batch pretrained-baseline counts. |
| `forward_verification.json` | Weight identity, inference settings and per-target comparisons. |
| `historical_prediction_audit.json` | Integrity checks and metric recomputation for the paper-evaluation prediction archives. |
| `standalone_check.json` | Independent entry-point checks on one image per target and verification of all 40 actual MRC and GT hashes. |
| `repository_manifest.json` | Repository and tested-package entry-point fingerprints. |

The [v3 registry](../../experiments/generalization/v3_low_snr/image_registry.json) records HCN1 `regression_check` and TRPV1 `final_check` membership. Matching uses maximum-cardinality one-to-one centers with 38.5 / 33 native-pixel radii. Baseline prediction and model hashes were verified and counts reused; baseline forward inference was not rerun on this date.

Paper-evaluation ANN counts are HCN1 2103/691/225 and TRPV1 4971/1038/8111 (TP/FP/FN). Fresh counts are 2104/691/224 and 4972/1037/8110. The one-match differences have not been attributed to a specific cause; both records are retained.

These are known benchmark checks. See the [model card](../../MODEL_CARD.md) for interpretation and scope. Separate development-run metadata is stored in [development_A3_20261116.json](../development_A3_20261116.json); it identifies a different checkpoint.
