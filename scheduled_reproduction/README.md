# Scheduled TokCom reproduction

This folder preserves the recurring reproduction work separately from the repository's earlier experiments. Stage numbering and results here belong to this folder's own sequence.

## Current status

Author VQGAN/MaskGIT weights and demo PHY have been exercised on ten development images across five receptions. Exact token counts, original-image PSNR, LPIPS and CLIP are recorded with explicit input conventions. CRC recovery can reduce exact token accuracy; soft-channel selection gives small development-image quality changes that require new-data validation.

A project-disjoint 100-image holdout and its protocol were frozen before selection/inference. Stage 45 caches author token IDs and five PHY receptions per image, with per-file identity locks and resumable atomic writes. See evidence/stage45_cache_audit.json and the separate recovery/metric coverage below. The frozen 8 dB holdout evaluation is complete and must not be used for further strategy tuning.

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

All 100 images and five receptions are represented in `evidence/stage47_complete_holdout_summary.json`. It contains six-path means, paired per-image differences and all per-image values. Trial means are computed within image before averaging images. No exploratory gate fraction is selected. This completes the frozen 8 dB evaluation. Stage 53 adds an analytic payload TCE reference, an exploratory development CMI comparison and ideal correct-token VQGAN references. Resource-equivalent baseline experiments, broader CMI validation and paper-wide multi-SNR reproduction remain open. Engineering feedback protocols are evaluated separately.

## Interpretation

The holdout is disjoint from prior project image files and decoded pixels, not proven disjoint from public ImageNet pretraining. Author oracle masks require sender truth; CRC masks use receiver checks. Paper TCE/retransmission baselines, CMI ablation, resource-equivalent comparisons and multi-SNR reproduction remain incomplete. Engineering feedback/control protocols are separate extensions. Development results are not paper-wide performance claims. Three exploratory gate fractions are all reported; none is selected from holdout outcomes.

## Separate multi-SNR development pilot (stage 54)

The completed primary 100-image holdout stays frozen. A separate preplanned pilot uses the old ten development images at 6, 6.5, 7, 7.5, 8 and 9 dB, five receptions each (300 total). Current audited coverage: PHY 300/300; four-path paired generation/metrics 72/300. Complete PHY SNR groups: 6/6. Modes w=0 and w=1 use the same sender-truth oracle mask and fixed generation seed. The paths are direct decode, without CMI, with CMI, and ideal correct-token decode.

Only complete 50-reception SNR groups receive outcome summaries. Current complete groups: 0/6. Incomplete groups report coverage only. This author-demo fixed interleaver and ten-image pilot do not establish paper-wide multi-SNR reproduction. Ideal correct-token decode does not simulate physical retransmissions.

Resume `cache_stage54_development_phy.py` in the separate Sionna environment, audit with `audit_stage54_development_phy.py`, then run `run_stage54_development_multisnr.py` and `audit_stage54_development_recovery.py` in the PyTorch environment. Preserve plan and execution identities; restore local inputs named by the plan. `sync_github_worktree_from_api.py` verifies a published unsigned commit and full Git tree when native fetch is unavailable.

## Convention alignment audit (stage 55)

`evidence/stage55_input_convention_audit.json` verifies four pinned author files against remote Git blob identities. The published training code maps input x to 2*x-1 before VQ encoding, while the demo notebook directly encodes ToTensor output. The demo LPIPS call also differs from the explicit normalization convention in this frozen evaluation. This is a source-path discrepancy, not proof of checkpoint training provenance or a causal explanation for quality.

`evidence/stage55_range_diagnostic_plan.json` preregisters a separate four-condition noiseless development diagnostic. Stage 55 saves this plan without executing it. No primary or active multi-SNR settings change. Complete PHY statistics and CRC/payload masking exposure are separately recorded in `evidence/stage55_complete_phy_mask_exposure.json`; they do not measure generation damage.

## Executed convention diagnostic (stage 56)

The preregistered four-condition noiseless VQGAN diagnostic now has 10/10 audited development images. Complete results report all input/output mappings without choosing a winner. See `evidence/stage56_range_audit.json`. Token-ID differences between encodings are not channel TER. These results do not change the primary holdout or active multi-SNR settings.

The separate fixed-50%-mask token-prior diagnostic covers 10/10 audited images. It compares both tokenizations with and without class conditioning using saved logits. Cross entropy is recomputed over the author's full 1025-class output; top-1 accuracy is measured only at 128 masked positions per image. This is not iterative generation or channel recovery. See `evidence/stage56_prior_probe_audit.json`.
