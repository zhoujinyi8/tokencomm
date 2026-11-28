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

The completed primary 100-image holdout stays frozen. A separate preplanned pilot uses the old ten development images at 6, 6.5, 7, 7.5, 8 and 9 dB, five receptions each (300 total). Current audited coverage: PHY 300/300; four-path paired generation/metrics 300/300. Complete PHY SNR groups: 6/6. Modes w=0 and w=1 use the same sender-truth oracle mask and fixed generation seed. The paths are direct decode, without CMI, with CMI, and ideal correct-token decode.

Only complete 50-reception SNR groups receive outcome summaries. Current complete groups: 6/6. Incomplete groups report coverage only. This author-demo fixed interleaver and ten-image pilot do not establish paper-wide multi-SNR reproduction. Ideal correct-token decode does not simulate physical retransmissions.

Resume `cache_stage54_development_phy.py` in the separate Sionna environment, audit with `audit_stage54_development_phy.py`, then run `run_stage54_development_multisnr.py` and `audit_stage54_development_recovery.py` in the PyTorch environment. Preserve plan and execution identities; restore local inputs named by the plan. `sync_github_worktree_from_api.py` verifies a published unsigned commit and full Git tree when native fetch is unavailable.

## Convention alignment audit (stage 55)

`evidence/stage55_input_convention_audit.json` verifies four pinned author files against remote Git blob identities. The published training code maps input x to 2*x-1 before VQ encoding, while the demo notebook directly encodes ToTensor output. The demo LPIPS call also differs from the explicit normalization convention in this frozen evaluation. This is a source-path discrepancy, not proof of checkpoint training provenance or a causal explanation for quality.

`evidence/stage55_range_diagnostic_plan.json` preregisters a separate four-condition noiseless development diagnostic. Stage 55 saves this plan without executing it. No primary or active multi-SNR settings change. Complete PHY statistics and CRC/payload masking exposure are separately recorded in `evidence/stage55_complete_phy_mask_exposure.json`; they do not measure generation damage.

## Executed convention diagnostic (stage 56)

The preregistered four-condition noiseless VQGAN diagnostic now has 10/10 audited development images. Complete results report all input/output mappings without choosing a winner. See `evidence/stage56_range_audit.json`. Token-ID differences between encodings are not channel TER. These results do not change the primary holdout or active multi-SNR settings.

The separate fixed-50%-mask token-prior diagnostic covers 10/10 audited images. It compares both tokenizations with and without class conditioning using saved logits. Cross entropy is recomputed over the author's full 1025-class output; top-1 accuracy is measured only at 128 masked positions per image. This is not iterative generation or channel recovery. See `evidence/stage56_prior_probe_audit.json`.

## Separate matched end-to-end conventions (stage 57)

The preregistered old-development comparison now has 50/50 demo-pipeline cases and 50/50 training-code-convention cases, representing 50/50 complete image/trial pairs. Four paths are recorded per case. 12 demo cases are reused only after exact identity checks; 88 cases are computed by this study. Both PHY streams have 50 audited receptions; 50 new simulations and 50 inherited demo PHY references are distinguished.

The two conventions are x -> raw decode y, and 2*x-1 -> (y+1)/2 for every metric path. Same images, 8 dB, noise seeds, code/modulation, class modes and generation parameters are retained. Each stream uses its own true-error oracle mask; both class modes share that mask within the stream. No cross-pipeline quality means are released until all 50 pairs and all four paths are complete. See `evidence/stage57_comparison_audit.json`; this independent namespace does not alter the frozen primary or the original six-SNR grid.

Resume `run_stage57_convention_comparison.py`, audit with `audit_stage57_convention_comparison.py`, then render a session with `report_convention_comparison.py --session runs/SESSION.json`. The frozen reuse-source manifest prevents later original-grid progress from changing this study's initial identities.

## Resumed six-SNR model study (stage 59)

The matched convention study is complete at 100 pipeline receptions and 50 pairs; its frozen audit and all 100 case files remain unchanged. Stage 59 resumes the original independently locked six-SNR model study from 72/300. Current model coverage: 300/300; new model cases this window: 228. PHY remains 300/300, with 0 new physical simulations this window. No matched-convention case is transferred into the original grid. Incomplete SNR groups receive coverage only, without quality means.

Publication follows draft PR #2 on `reproduction/automation-30-stage58`. Use `sync_github_worktree_from_api.py --branch BRANCH --commit SHA --expected-parent PARENT --receipt runs/RECEIPT.json` for exact tree/commit synchronization on the selected existing branch. The explicit branch is checked before staging or ref mutation; isolated Git contract checks cover rejection and idempotence.

## Independent randomized packet assignment (stage 60)

The prior fixed-grid study stays complete at 300/300, the primary holdout at 500/500 and the convention study at 100/100. A separately preregistered study now has randomized PHY 300/300 and four-path models 217/300. The 4 diagnostic transmissions are separate from formal receptions. The fixed complete baseline audit was pinned before first new inference.

Each image has a fixed PCG64 permutation shared across five trials and six SNRs. Packet errors are defined in transmitted packet coordinates; received IDs and both masks are inverse-permuted before model processing. All other model/channel/metric identities are retained. Completed SNR groups alone receive paired quality means; current complete paired groups: 0/6. This is an old-Imagenette sensitivity study, not paper-wide ImageNet100 replication.

Resume `run_stage60_randomized_recovery.py`, then `audit_stage60_randomized_recovery.py` and `audit_stage60_paired_packetization.py`. Render `report_randomized_session.py --session runs/SESSION.json`. Keep the execution spec, permutation helper, scripts and locks unchanged. The pure permutation/mask checks, actual auditor fixtures and paired-aggregation gates have separate bounded receipts. Noisy high-SNR and exact noiseless diagnostic checks are labeled separately.

## Completed pairing and actual packet ARQ (stage 61)

The randomized four-path model study is complete at 300/300, including 83 new cases this round and all six complete fixed/random pairs. All four paths and per-image differences are retained without choosing a winning permutation.

The separate padded-packet experiment completes 300/300 frames and 900 VQGAN endpoint metric rows across single pass, receiver CRC ARQ and sender-oracle ARQ. Attempts are simulated with independent packet/attempt noise, capped at 32, with wrong CRC acceptances and censoring retained. Each first pass is 1664 QAM symbols (40 payload + 11 CRC bits, half-rate convolution, 2 padding bits per packet). Forward data only: feedback/header/CMI airtime is not modeled. The earlier 1632-symbol MaskGIT results are not a matching-framing baseline.

Metric accounting records 298 actual decoder forwards and 612 rows using exact same-image clean-ID references. No MaskGIT inference is claimed for ARQ endpoints. Two noiseless packet diagnostics and 14 pure accounting/metric checks are separate from 300 formal frames. The initial Mapper shape preflight correction and its replaced lock are preserved.

Next: `stage61_matched_padded_recovery_plan.json` and 300 audited canonical first-pass inputs prepare an independent six-path neural comparison under the same 1664-symbol framing. Implement and lock its runner/auditor before new inference. This round performs zero Transformer recovery cases in that new study. Keep all prior frozen studies unchanged; ImageNet100, Cheng and complete resource-equivalent paper replication remain open.

## Matched padded neural recovery (stage 62)

Implemented the separately locked six-path runner and auditor. Audited coverage: 82/300 receptions, six paths each. Uses the 300 existing 1664-symbol first-pass inputs, separate oracle/CRC masks, and frozen model/metric conventions. No new PHY and no copying old unpadded generated endpoints. Incomplete SNR groups receive coverage only. Resume `run_stage62_matched_recovery.py`, then `audit_stage62_matched_recovery.py` and `report_stage62_matched.py`. The six pure mask/metric validation checks are independent of real model evaluations.

## Matched padded continuation (stage 63)

Continued from 82 cases with the identical runner, inputs and execution lock. Current independently audited coverage: 194/300 receptions; new this round: 112. The first 82 case hashes remain unchanged. Stage62 session receipts are frozen separately. Resume the same locked runner for missing cases. Quality means require 50 receptions per SNR; no incomplete-SNR means are published.

## Matched padded completion check (stage 64)

Continued from 194 unchanged cases with the identical execution lock. Audited coverage is 300/300 with 6 complete SNR groups. Each complete group compares all six neural paths with all three same-framing ARQ endpoint policies, including per-image paired differences. Direct decoder endpoints are checked against the identical first-pass ARQ endpoints before comparison. Feedback/header/CMI airtime is excluded; ideal correct tokens remain a quality reference, not a realizable retransmission policy. Full paper replication remains incomplete. See the stage64 report and comparison audit.

## Cheng byte-stream preparation (stage 65)

Audited 20/20 clean stock-codec cases across anchor/attention quality 1 and the same ten old development images, with two fresh-process decoding checks. Actual entropy bpp and custom 34-byte CH65 header costs are distinct from likelihood estimates. Stock rates are about 3.2 times the paper's stated ~0.04 bpp target; these are not paper-rate-matched baselines. Native upstream rANS/CDF extensions were built in a separate Windows environment from hash-pinned source/toolchain archives. No native binaries, checkpoints, image arrays or compressed image streams are exported in this Git snapshot.

A separately planned one-update training smoke and fixed 100-update balanced proxy pilot exercise optimization and checkpoint/RNG state. They use existing training images with frozen evaluation identities excluded, and do not establish paper checkpoint/data provenance. See the pilot audit for actual update/calibration coverage. `verify_imagenet100_manifest.py` checks a supplied 100-class manifest and file hashes; structural validity is explicitly separate from author provenance. Full paper reproduction remains incomplete.

## Fixed-budget Cheng calibration and frozen endpoint (stage 66)

All three independent 100-update Anchor arms complete on the same disjoint training images/order/seed. The preregistered rate-only selector chooses `lambda_00015` from all ten development images; all 30 fresh-decoder pixel hashes agree. It is frozen before the independent existing ten-class 100-image clean holdout. Held-out actual entropy rate is 0.042954 bpp, container rate 0.047104, PSNR 23.673 dB, LPIPS 0.452458, CLIP 0.643420. Five preregistered independent held-out decodes reproduce pixels exactly. No held-out training/reselection occurs.

Actual custom-container first-pass cost is 2017.34 QAM symbols versus the earlier 1664-symbol TokCom framing reference, so this is not complete resource matching. A separate physical variable-packet byte study covers 300/300 frames with CRC, single-pass and oracle stopping traces; 6/6 SNR groups are complete. Engineering CRC endpoint quality is complete for all 300 frames: 299 exact-byte clean references and one actually decoded corrupt frame (PSNR 8.612 dB). All outcomes are retained. Corrupt single-pass endpoint quality remains incomplete, with no failure imputation. This is not a complete noisy Cheng paper benchmark.

The arXiv v4/v5 case-study source is identical; v5 adds a code URL in the abstract and build metadata. Focused TER/config facts also match the original journal text cache, without re-verifying the journal PDF. Exact paper ImageNet100 identities, Cheng recipe/checkpoint and complete feedback/control/CMI airtime remain open. 17 new pure checks bring the CI protocol total to 60. See stage66 session and source receipts. Full paper reproduction is incomplete.
