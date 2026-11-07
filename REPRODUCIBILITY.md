# Reproducibility

## Data and checkpoints

- Micrographs are available from [EMPIAR](https://www.ebi.ac.uk/empiar/). Public annotation archives are available from [CryoPPP on Zenodo](https://zenodo.org/records/7934683).
- Use the filenames, membership and checksums in `manifests/` and the generalization image registries. Preserve the original pixel grids and use the code's stated coordinate conversion.
- Obtain the official YOLO26m initialization from [Ultralytics](https://docs.ultralytics.com/models/yolo26/). The original initialization SHA-256 is in `experiments/empiar10028/config/standard_m_p3p5_config.json`. The benchmark freeze files record the fitted checkpoint identities and decision thresholds. A newly trained checkpoint is a new run and must be selected and calibrated on validation data.

## EMPIAR-10028

**The reported EMPIAR-10028 benchmark uses the frozen test set `test_100_v1` with 100 micrographs and 8,830 annotated particles.** Training and validation use 132 and 44 micrographs, respectively. The three methods use the same test set and ground truth, with one-to-one matching at 56 pixels in the native image grid.

- [Final test membership](manifests/empiar10028/test_100_v1/test.txt) lists the exact 100 micrographs used for the reported comparison.
- [Test manifest](manifests/empiar10028/test_100_v1/test_manifest.csv) records per-image annotation counts and file checksums.
- [Frozen evaluation configuration](experiments/empiar10028/config/frozen_config.json) records the validation-selected thresholds and points to the final test manifest.

The snapshots preserve the original Linux directory layout. On a dedicated reproduction machine, place the dataset under `/root/autodl-tmp/cryoem_datasets/EMPIAR-10028` or adapt the path constants in a working copy. Within that root:

1. Place MRCs in `micrographs/` and copy both included split folders into `manifests/`.
2. Extract the CryoPPP annotation archive so CSVs are under `annotations_original/cryoppp_10028/10028/ground_truth/particle_coordinates/`. For each train/validation member, write the CSV's `X-Coordinate` and `Y-Coordinate` as two space-separated columns to `annotations_standardized/split_v3_paper60_20_20/ann_yolo26/coords/{train,val}/<stem>.coord`. Keep the original coordinates because `prepare.py` applies the MRC orientation transform internally. `data/create_split_v3.py` records the original split and conversion procedure and expects the earlier project layout.
3. Extract the archived Ultralytics runtime to `/root/autodl-tmp/cryoem_annyolo26/SNNYOLO26/ultralytics-snn-yolo26` and place the initialization there. The upstream AMP check uses the official `yolo26n.pt` when applicable.
4. Copy the evaluator to `metadata/three_method_evaluation/`. Run the following from the release root using the environment in the README:

```bash
python experiments/empiar10028/train/prepare.py
python experiments/empiar10028/train/train.py --epochs 80 --batch 4 --lr 0.002
python experiments/empiar10028/train/select_checkpoints.py --run standard_m_p3p5
```

The selector checks checkpoints on the 44 validation micrographs. Its selection rule and candidate paths are recorded in the script. The frozen 100-image run uses `test/ann_test_inference.py` and `test/evaluate_test100.py` from the original experiment directory `experiments/test_100_v1_three_methods/code/` under the dataset root. `data/prepare_test_100_v1.py` records the preparation of the 100-image test set and shared ground truth. Install crYOLO and Topaz separately if reproducing the full three-method comparison. Use the versions, checkpoints, adapters and per-method thresholds in `config/frozen_config.json`. The test scripts verify the frozen seal and hashes. The frozen EMPIAR-10028 comparison requires the dataset-specific ANN checkpoint and the baseline weights listed in that configuration. The public General checkpoint is the model used for the HCN1 generalization run. Recreate the recorded layout and inputs to replay the original run. For a separate retrained run, create a separate validation-derived freeze with `test/freeze_test.py` before opening the test set.

## Related datasets: EMPIAR-10017 and EMPIAR-10291

| Dataset | Role and records |
| --- | --- |
| EMPIAR-10017 | The later 14-micrograph comparison uses the ANN-YOLO26m method developed for EMPIAR-10028. Its evaluation inputs are listed in the [14-image manifest](manifests/empiar10017/test14.txt). For the supplied coordinate export, the comparison uses 27-pixel one-to-one center matching. The export does not establish the checkpoint or threshold provenance. EMPIAR-10017 also appears as a source dataset in the generalization protocol. |
| EMPIAR-10291 | Included as a source dataset for generalization. See the [source and target protocol](experiments/generalization/v2/protocol.json) and [image registry](experiments/generalization/v2/image_registry.json) for its documented role and split membership. |

## Cross-dataset generalization

The research snapshots in `experiments/generalization/` map to `/root/autodl-tmp/cryoem_generalization/{v1,v2,v3_low_snr}`. Use each variant's protocol and image registry to preserve source membership and target batches. The v2 code includes the canonical-scale candidate and the native multiscale branch. The standalone v2 configuration identifies the selected native multiscale checkpoint. The low-SNR continuation initializes from the selected v2 checkpoint and applies source-side contrast and noise augmentation. The standalone v3 configuration identifies the separately fitted low-SNR checkpoint.

- Source download and preparation are recorded in `v1/code/bootstrap_catalogs.py`, `v1/code/prepare_external.py`, `v2/code/prepare_data.py` and `v2/code/prepare_training.py`.
- For v2, see `v2/code/workflow.py`, `v2/code/source_warmup.py` and `v2/code/native_wrapper.py` for the original staged runs and prerequisite checkpoints. Run each stage in the order documented in the workflow after its inputs are present.
- For the low-SNR continuation, see `v3_low_snr/code/initialize.py`, `train_low_snr.py` and `select_evaluate.py`. The scripts depend on the earlier source registry, prepared data and v2 checkpoint. The protocol and freeze files identify the target batch and source-side threshold selection.
- Dataset-specific radii and target memberships are recorded in the corresponding freeze files. HCN1 and TRPV1 experiments must be interpreted using their own frozen variant and batch.

Historical workflow scripts use original paths and some require intermediate outputs of earlier stages. Standalone inference is the direct entry point when the matching fitted checkpoint is available. Use [ANN-YOLO26m-General.pt](https://github.com/chan401/ANN-YOLO26m/releases/download/paper-submission-v1/ANN-YOLO26m-General.pt) with the v2 configuration to reproduce the reported HCN1 run. Use [ANN-YOLO26m-LowSNR.pt](https://github.com/chan401/ANN-YOLO26m/releases/download/paper-inference-v1/ANN-YOLO26m-LowSNR.pt) with the v3 low-SNR configuration to reproduce the reported TRPV1 run. The experiment snapshots and other checkpoint requirements are documented above.
