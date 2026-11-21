# Scheduled TokCom reproduction

This folder preserves the recurring reproduction work separately from the repository's earlier experiments. Stage numbering and results here belong to this folder's own sequence.

## Current status

Author VQGAN/MaskGIT weights and demo PHY have been exercised on ten development images across five receptions. Exact token counts, original-image PSNR, LPIPS and CLIP are recorded with explicit input conventions. CRC recovery can reduce exact token accuracy; soft-channel selection gives small development-image quality changes that require new-data validation.

A project-disjoint 100-image holdout and its protocol were frozen before selection/inference. Stage 45 caches author token IDs and five PHY receptions per image, with per-file identity locks and resumable atomic writes. See evidence/stage45_cache_audit.json for actual completion; absence of final recovery metrics is intentional. The holdout is now being accessed under the frozen protocol and must not be used for further strategy tuning.

## Files and execution layout

- `reports/` contains Chinese progress reports; `evidence/` contains machine-readable summaries and frozen identities.
- `experiment/tokencom_inspect/tokencom_reproduction/` contains experiment scripts and local dependencies.
- Obtain author code at commit `0422ed32dcde3347e0bb6697f343f7a9ff7b3214` from https://github.com/liqiao19/TokenCom_Code and place it in `experiment/tokencom_official_reference/`.
- Restore the frozen image list and local image files under the code directory's `data/imagenette_stage44_holdout/`. Restore protocol/lock JSON files into its `runs/` directory. Raw images, large weights and generated caches are excluded from Git.
- Author weight identities and settings are in `evidence/stage44_frozen_protocol.json`. Obtain the weights from the author's referenced public source; do not substitute community token caches.
- Token encoding uses the main PyTorch environment. PHY caching uses TensorFlow 2.15.1 / Sionna 0.19.2 in a separate environment. Run `cache_stage45_holdout_tokens.py`, then `cache_stage45_holdout_phy.py`, then `audit_stage45_caches.py` from the code directory. Locks reject changed scripts on resume.

## Interpretation

The holdout is disjoint from prior project image files and decoded pixels, not proven disjoint from public ImageNet pretraining. Author oracle masks require sender truth; CRC masks use receiver checks. Full paper TCE, feedback protocol, resource-equivalent comparisons and a multi-SNR reproduction remain incomplete. Development results are not paper-wide performance claims. Three exploratory gate fractions are all reported; none is selected from holdout outcomes.
