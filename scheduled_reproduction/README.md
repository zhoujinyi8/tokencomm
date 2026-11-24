# Scheduled TokCom reproduction

This folder preserves the recurring reproduction work separately from the repository's earlier experiments. Stage numbering and results here belong to this folder's own sequence.

## Current status

Author VQGAN/MaskGIT weights and demo PHY have been exercised on ten development images across five receptions. Exact token counts, original-image PSNR, LPIPS and CLIP are recorded with explicit input conventions. CRC recovery can reduce exact token accuracy; soft-channel selection gives small development-image quality changes that require new-data validation.

A project-disjoint 100-image holdout and its protocol were frozen before selection/inference. Stage 45 caches author token IDs and five PHY receptions per image, with per-file identity locks and resumable atomic writes. See evidence/stage45_cache_audit.json and the separate recovery/metric coverage below. The holdout is being evaluated under the frozen protocol and must not be used for further strategy tuning.

## Frozen evaluation coverage

- Recovery: 500/500 receptions; 100/100 images have five complete trials.
- Image metrics: 500/500 receptions, six paths per reception; 100/100 images have five metric trials.
- See `evidence/stage46_output_audit.json` and `evidence/stage47_metric_file_audit.json` for actual audited coverage. Recovery completion alone does not establish metric completion or paper-wide performance. The complete summary entry point requires every image and trial.

## Files and execution layout

- `reports/` contains Chinese progress reports; `evidence/` contains machine-readable summaries and frozen identities.
- `experiment/tokencom_inspect/tokencom_reproduction/` contains experiment scripts and local dependencies.
- Obtain author code at commit `0422ed32dcde3347e0bb6697f343f7a9ff7b3214` from https://github.com/liqiao19/TokenCom_Code and place it in `experiment/tokencom_official_reference/`.
- Restore the frozen image list and local image files under the code directory's `data/imagenette_stage44_holdout/`. Restore protocol/lock JSON files into its `runs/` directory. Raw images, large weights and generated caches are excluded from Git.
- Author weight identities and settings are in `evidence/stage44_frozen_protocol.json`. Obtain the weights from the author's referenced public source; do not substitute community token caches.
- Token encoding uses the main PyTorch environment. PHY caching uses TensorFlow 2.15.1 / Sionna 0.19.2 in a separate environment. Run `cache_stage45_holdout_tokens.py`, then `cache_stage45_holdout_phy.py`, then `audit_stage45_caches.py` from the code directory. Locks reject changed scripts on resume.

## Continuing the frozen evaluation

Restore `evidence/stage44_holdout/{manifest.json,test.jsonl}` and the protocol into the local paths above. `restore_stage44_images.py --parquet PATH` reconstructs exactly the selected images from the source parquet pinned by SHA-256; it does not resample images.

After the stage45 cache audit is complete, run `run_stage46_frozen_recovery.py`. Its fixed computation window saves complete receptions and resumes missing cases under the same script lock. `audit_stage46_outputs.py` validates saved input identity, preservation of unmasked positions and exact gate definitions without performance aggregation.

`evaluate_stage47_frozen_metrics.py` writes per-reception metrics under a separate execution lock; outcomes are not used to select settings. `summarize_stage47_complete_holdout.py` refuses an incomplete evaluation: it requires all 100 images and five receptions for all six paths. Preserve source and execution locks between sessions.

## Complete frozen summary

All 100 images and five receptions are represented in `evidence/stage47_complete_holdout_summary.json`. It contains six-path means, paired per-image differences and all per-image values. Trial means are computed within image before averaging images. No exploratory gate fraction is selected. This completes the frozen 8 dB evaluation, while paper TCE/baseline, CMI ablation and multi-SNR reproduction remain open. Engineering feedback protocols are evaluated separately.

## Interpretation

The holdout is disjoint from prior project image files and decoded pixels, not proven disjoint from public ImageNet pretraining. Author oracle masks require sender truth; CRC masks use receiver checks. Paper TCE/retransmission baselines, CMI ablation, resource-equivalent comparisons and multi-SNR reproduction remain incomplete. Engineering feedback/control protocols are separate extensions. Development results are not paper-wide performance claims. Three exploratory gate fractions are all reported; none is selected from holdout outcomes.
