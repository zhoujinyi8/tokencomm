"""Copy explicit source/report/evidence allowlists into the isolated Git worktree."""
import hashlib
import json
import shutil
from pathlib import Path

root=Path(__file__).resolve().parent
checkout=root.parents[1]/'tokencomm_github_sync'
assert (checkout/'.git').is_file()
(checkout/'.flake8').write_text('[flake8]\nextend-exclude = scheduled_reproduction\n',encoding='utf-8')
(checkout/'pytest.ini').write_text('[pytest]\npython_files = test_*.py\nmarkers =\n    integration: requires local experiment data, checkpoints, or frozen artifact identities\nnorecursedirs = .* build dist CVS _darcs {arch} *.egg venv scheduled_reproduction\n',encoding='utf-8')
target=checkout/'scheduled_reproduction'
target.mkdir(parents=True,exist_ok=True)
(target/'.gitattributes').write_text('* -text\n',encoding='utf-8')
(target/'.gitignore').write_text('experiment/tokencom_official_reference/\n**/__pycache__/\n',encoding='utf-8')
code=target/'experiment/tokencom_inspect/tokencom_reproduction'
code.mkdir(parents=True,exist_ok=True)
for path in root.glob('*.py'):shutil.copy2(path,code/path.name)
for name in ('requirements.txt','paper_reference.json'):
    if (root/name).exists():shutil.copy2(root/name,code/name)
for path in (root/'third_party').rglob('*'):
    if path.is_file() and (path.suffix.lower() in ('.py','.md','.txt','.json','.yaml','.yml') or 'LICENSE' in path.name.upper()):
        out=code/path.relative_to(root);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
reports=target/'reports';reports.mkdir(exist_ok=True)
for path in (root.parents[2]/'outputs').glob('TokCom_*.md'):shutil.copy2(path,reports/path.name)
for path in (root.parents[2]/'outputs').glob('TokCom_阶段*多SNR续跑_*_PHY.svg'):shutil.copy2(path,reports/path.name)
for path in (root.parents[2]/'outputs').glob('TokCom_阶段*多SNR续跑_*_模型质量.svg'):shutil.copy2(path,reports/path.name)
for path in (root.parents[2]/'outputs').glob('TokCom_阶段*随机交织对照_*_PHY.svg'):shutil.copy2(path,reports/path.name)
for path in (root.parents[2]/'outputs').glob('TokCom_阶段*随机交织对照_*_模型配对差值.svg'):shutil.copy2(path,reports/path.name)
for path in (root.parents[2]/'outputs').glob('TokCom_阶段*随机交织对照_*_填充包ARQ.svg'):shutil.copy2(path,reports/path.name)
for path in (root.parents[2]/'outputs').glob('TokCom_阶段*完整链路输入约定对照_*_配对差值.svg'):shutil.copy2(path,reports/path.name)
evidence=target/'evidence';evidence.mkdir(exist_ok=True)
for name in ['stage30_multitrial_progress.json','stage33_author_psnr_10images_5trials.json',
             'stage34_lpips_five_trials_summary.json','stage35_clip_summary.json',
             'stage36_joint_metrics.json','stage37_direct_comparison.json','stage39_transition_attribution.json',
             'stage40_gate_summary.json','stage41_soft_channel_diagnostics.json','stage43_soft_gate_summary.json',
             'stage44_frozen_protocol.json','stage44_holdout_audit.json','stage45_encoder_lock.json',
             'stage45_phy_lock.json','stage45_cache_audit.json',
             'stage46_recovery_lock.json','stage46_recovery_progress.json',
             'stage46_output_audit.json','stage46_environment.json',
             'stage47_metrics_lock.json','stage47_metric_file_audit.json',
             'stage48_ci_verification.json','stage48_session_progress.json',
             'stage49_session_progress.json','stage50_session_progress.json',
             'stage51_window_start.json','stage51_session_progress.json',
             'stage52_window_start.json','stage52_session_progress.json',
             'stage47_complete_holdout_summary.json','stage52_tce_reference_check.json',
             'stage52_paper_scope_review.json','stage53_window_start.json',
             'stage53_session_progress.json','stage53_cmi_interface_audit.json',
             'stage53_payload_tce_plan.json','stage53_payload_tce_result.json',
             'stage53_complete_summary_audit.json','stage53_development_cmi_plan.json',
             'stage53_development_cmi_result.json','stage53_ter_definition_audit.json',
             'stage53_development_cmi_grid_plan.json','stage53_development_cmi_grid_summary.json',
             'stage53_development_cmi_grid_audit.json','stage53_ideal_vq_plan.json',
             'stage53_ideal_vq_summary.json','stage53_ideal_vq_audit.json',
             'stage54_window_start.json','stage54_local_sync_recovery.json',
             'stage54_api_sync_idempotence.json','stage54_development_multisnr_plan.json',
             'stage54_development_phy_progress.json','stage54_development_phy_audit.json',
             'stage54_development_recovery_lock.json','stage54_development_recovery_progress.json',
             'stage54_development_recovery_audit.json','stage54_session_progress.json',
             'stage54_audit_contract_verification.json','stage55_window_start.json',
             'stage55_session_progress.json','stage55_development_phy_audit.json',
             'stage55_development_recovery_audit.json','stage55_complete_phy_mask_exposure.json',
             'stage55_input_convention_audit.json','stage55_range_diagnostic_plan.json',
             'stage56_window_start.json','stage56_session_progress.json',
             'stage56_development_phy_audit.json','stage56_development_recovery_audit.json',
             'stage56_range_execution_lock.json','stage56_range_progress.json','stage56_range_audit.json',
             'stage56_prior_probe_plan.json','stage56_prior_probe_lock.json',
             'stage56_prior_probe_progress.json','stage56_prior_probe_audit.json',
             'stage56_end_to_end_convention_plan.json','stage57_window_start.json',
             'stage57_session_progress.json','stage57_comparison_phy_lock.json',
             'stage57_comparison_phy_progress.json','stage57_comparison_phy_audit.json',
             'stage57_reuse_source_audit.json','stage57_comparison_execution_lock.json',
             'stage57_comparison_progress.json','stage57_comparison_audit.json',
             'stage57_aggregation_contract_verification.json','stage57_session_comparison_audit.json',
             'stage58_window_start.json','stage58_session_progress.json','stage58_session_comparison_audit.json',
             'stage58_plot_smoke_verification.json','stage58_checkpoint_metadata_audit.json',
             'stage58_upstream_weight_source_audit.json',
             'stage59_window_start.json','stage59_session_progress.json',
             'stage59_development_phy_audit.json','stage59_development_recovery_audit.json',
             'stage59_publication_branch_contract.json','stage59_runtime_identity.json',
             'stage59_paper_alignment_review.json','stage59_randomized_packetization_plan.json',
             'stage59_token_transitions.json','stage59_token_transition_contract.json',
             'stage60_window_start.json','stage60_execution_spec.json','stage60_runtime_identity.json','stage60_phy_runtime_identity.json',
             'stage60_packetization_contract.json','stage60_phy_roundtrip_smoke.json',
             'stage60_noiseless_phy_verification.json','stage60_randomized_phy_lock.json',
             'stage60_randomized_phy_progress.json','stage60_randomized_phy_audit.json',
             'stage60_randomized_recovery_lock.json','stage60_randomized_recovery_progress.json',
             'stage60_randomized_recovery_audit.json','stage60_model_audit_contract.json',
             'stage60_paired_aggregation_contract.json','stage60_paired_packetization_audit.json',
             'stage60_session_progress.json','stage60_session_phy_audit.json',
             'stage60_session_model_audit.json','stage60_session_paired_audit.json',
             'stage61_window_start.json','stage61_runtime_identity.json','stage61_session_progress.json',
             'stage61_session_phy_audit.json','stage61_session_model_audit.json','stage61_session_paired_audit.json',
             'stage61_packet_arq_plan.json','stage61_packet_arq_lock.json','stage61_packet_arq_lock_v2.json',
             'stage61_arq_contract.json','stage61_arq_noiseless_verification.json','stage61_arq_preflight_correction.json',
             'stage61_packet_arq_pilot_audit.json','stage61_packet_arq_progress.json','stage61_packet_arq_audit.json',
             'stage61_session_packet_arq_audit.json','stage61_arq_metric_plan.json','stage61_arq_metric_lock.json',
             'stage61_arq_metric_contract.json','stage61_arq_metric_progress.json','stage61_arq_metric_audit.json',
             'stage61_session_arq_metric_audit.json','stage61_matched_padded_recovery_plan.json',
             'stage61_matched_padded_input_lock.json','stage61_matched_padded_input_audit.json',
             'stage62_matched_recovery_lock.json','stage62_matched_recovery_progress.json','stage62_matched_recovery_audit.json',
             'stage62_session_audit.json','stage62_session_progress.json','stage63_window_start.json','stage63_session_audit.json']:
    if (root/'runs'/name).exists():shutil.copy2(root/'runs'/name,evidence/name)
holdout=evidence/'stage44_holdout';holdout.mkdir(exist_ok=True)
for name in ('manifest.json','test.jsonl'):shutil.copy2(root/'data/imagenette_stage44_holdout'/name,holdout/name)
readme='''# Scheduled TokCom reproduction

This folder preserves the recurring reproduction work separately from the repository's earlier experiments. Stage numbering and results here belong to this folder's own sequence.

## Current status

Author VQGAN/MaskGIT weights and demo PHY have been exercised on ten development images across five receptions. Exact token counts, original-image PSNR, LPIPS and CLIP are recorded with explicit input conventions. CRC recovery can reduce exact token accuracy; soft-channel selection gives small development-image quality changes that require new-data validation.

A project-disjoint 100-image holdout and its protocol were frozen before selection/inference. Stage 45 caches author token IDs and five PHY receptions per image, with per-file identity locks and resumable atomic writes. See evidence/stage45_cache_audit.json and the separate recovery/metric coverage below. The holdout is being evaluated under the frozen protocol and must not be used for further strategy tuning.

## Files and execution layout

- `reports/` contains Chinese progress reports; `evidence/` contains machine-readable summaries and frozen identities.
- `experiment/tokencom_inspect/tokencom_reproduction/` contains experiment scripts and local dependencies.
- Obtain author code at commit `0422ed32dcde3347e0bb6697f343f7a9ff7b3214` from https://github.com/liqiao19/TokenCom_Code and place it in `experiment/tokencom_official_reference/`.
- Restore the frozen image list and local image files under the code directory's `data/imagenette_stage44_holdout/`. Restore protocol/lock JSON files into its `runs/` directory. Raw images, large weights and generated caches are excluded from Git.
- Author weight identities and settings are in `evidence/stage44_frozen_protocol.json`. Obtain the weights from the author's referenced public source; do not substitute community token caches.
- Token encoding uses the main PyTorch environment. PHY caching uses TensorFlow 2.15.1 / Sionna 0.19.2 in a separate environment. Run `cache_stage45_holdout_tokens.py`, then `cache_stage45_holdout_phy.py`, then `audit_stage45_caches.py` from the code directory. Locks reject changed scripts on resume.

## Interpretation

The holdout is disjoint from prior project image files and decoded pixels, not proven disjoint from public ImageNet pretraining. Author oracle masks require sender truth; CRC masks use receiver checks. Paper TCE/retransmission baselines, CMI ablation, resource-equivalent comparisons and multi-SNR reproduction remain incomplete. Engineering feedback/control protocols are separate extensions. Development results are not paper-wide performance claims. Three exploratory gate fractions are all reported; none is selected from holdout outcomes.
'''
readme=readme.replace('## Interpretation', '''## Continuing the frozen evaluation

Restore `evidence/stage44_holdout/{manifest.json,test.jsonl}` and the protocol into the local paths above. `restore_stage44_images.py --parquet PATH` reconstructs exactly the selected images from the source parquet pinned by SHA-256; it does not resample images.

After the stage45 cache audit is complete, run `run_stage46_frozen_recovery.py`. Its fixed computation window saves complete receptions and resumes missing cases under the same script lock. `audit_stage46_outputs.py` validates saved input identity, preservation of unmasked positions and exact gate definitions without performance aggregation.

`evaluate_stage47_frozen_metrics.py` writes per-reception metrics under a separate execution lock; outcomes are not used to select settings. `summarize_stage47_complete_holdout.py` refuses an incomplete evaluation: it requires all 100 images and five receptions for all six paths. Preserve source and execution locks between sessions.

## Interpretation''')
progress_path=root/'runs/stage46_recovery_progress.json'
metric_audit_path=root/'runs/stage47_metric_file_audit.json'
if progress_path.exists() and metric_audit_path.exists():
    progress=json.loads(progress_path.read_text(encoding='utf-8'))
    metrics=json.loads(metric_audit_path.read_text(encoding='utf-8'))
    readme=readme.replace('## Files and execution layout',f'''## Frozen evaluation coverage

- Recovery: {progress['completed_recovery_cases']}/500 receptions; {progress['complete_five_trial_images']}/100 images have five complete trials.
- Image metrics: {metrics['metric_receptions_saved']}/500 receptions, six paths per reception; {metrics['complete_five_trial_images']}/100 images have five metric trials.
- See `evidence/stage46_output_audit.json` and `evidence/stage47_metric_file_audit.json` for actual audited coverage. Recovery completion alone does not establish metric completion or paper-wide performance. The complete summary entry point requires every image and trial.

## Files and execution layout''')
if (root/'runs/stage47_complete_holdout_summary.json').exists():
    readme=readme.replace('The holdout is being evaluated under the frozen protocol and must not be used for further strategy tuning.',
                          'The frozen 8 dB holdout evaluation is complete and must not be used for further strategy tuning.')
    readme=readme.replace('## Interpretation','''## Complete frozen summary

All 100 images and five receptions are represented in `evidence/stage47_complete_holdout_summary.json`. It contains six-path means, paired per-image differences and all per-image values. Trial means are computed within image before averaging images. No exploratory gate fraction is selected. This completes the frozen 8 dB evaluation. Stage 53 adds an analytic payload TCE reference, an exploratory development CMI comparison and ideal correct-token VQGAN references. Resource-equivalent baseline experiments, broader CMI validation and paper-wide multi-SNR reproduction remain open. Engineering feedback protocols are evaluated separately.

## Interpretation''')
pilot_path=root/'runs/stage54_development_recovery_audit.json'
if pilot_path.exists():
    pilot=json.loads(pilot_path.read_text(encoding='utf-8'))
    phy=json.loads((root/'runs/stage54_development_phy_audit.json').read_text(encoding='utf-8'))
    assert pilot['all_checks_passed'] and phy['all_checks_passed']
    readme+=f'''\n## Separate multi-SNR development pilot (stage 54)\n
The completed primary 100-image holdout stays frozen. A separate preplanned pilot uses the old ten development images at 6, 6.5, 7, 7.5, 8 and 9 dB, five receptions each (300 total). Current audited coverage: PHY {phy['completed_cases']}/300; four-path paired generation/metrics {pilot['completed_cases']}/300. Complete PHY SNR groups: {len(phy['complete_snr_groups'])}/6. Modes w=0 and w=1 use the same sender-truth oracle mask and fixed generation seed. The paths are direct decode, without CMI, with CMI, and ideal correct-token decode.\n
Only complete 50-reception SNR groups receive outcome summaries. Current complete groups: {len(pilot['complete_snr_groups'])}/6. Incomplete groups report coverage only. This author-demo fixed interleaver and ten-image pilot do not establish paper-wide multi-SNR reproduction. Ideal correct-token decode does not simulate physical retransmissions.\n
Resume `cache_stage54_development_phy.py` in the separate Sionna environment, audit with `audit_stage54_development_phy.py`, then run `run_stage54_development_multisnr.py` and `audit_stage54_development_recovery.py` in the PyTorch environment. Preserve plan and execution identities; restore local inputs named by the plan. `sync_github_worktree_from_api.py` verifies a published unsigned commit and full Git tree when native fetch is unavailable.\n'''
if (root/'runs/stage55_input_convention_audit.json').exists():
    readme+='''\n## Convention alignment audit (stage 55)\n
`evidence/stage55_input_convention_audit.json` verifies four pinned author files against remote Git blob identities. The published training code maps input x to 2*x-1 before VQ encoding, while the demo notebook directly encodes ToTensor output. The demo LPIPS call also differs from the explicit normalization convention in this frozen evaluation. This is a source-path discrepancy, not proof of checkpoint training provenance or a causal explanation for quality.\n
`evidence/stage55_range_diagnostic_plan.json` preregisters a separate four-condition noiseless development diagnostic. Stage 55 saves this plan without executing it. No primary or active multi-SNR settings change. Complete PHY statistics and CRC/payload masking exposure are separately recorded in `evidence/stage55_complete_phy_mask_exposure.json`; they do not measure generation damage.\n'''
if (root/'runs/stage56_range_audit.json').exists():
    range_audit=json.loads((root/'runs/stage56_range_audit.json').read_text(encoding='utf-8'))
    assert range_audit['all_checks_passed']
    readme+=f'''\n## Executed convention diagnostic (stage 56)\n
The preregistered four-condition noiseless VQGAN diagnostic now has {range_audit['complete_images']}/10 audited development images. Complete results report all input/output mappings without choosing a winner. See `evidence/stage56_range_audit.json`. Token-ID differences between encodings are not channel TER. These results do not change the primary holdout or active multi-SNR settings.\n'''
if (root/'runs/stage56_prior_probe_audit.json').exists():
    prior=json.loads((root/'runs/stage56_prior_probe_audit.json').read_text(encoding='utf-8'))
    assert prior['all_checks_passed']
    readme+=f'''\nThe separate fixed-50%-mask token-prior diagnostic covers {prior['complete_images']}/10 audited images. It compares both tokenizations with and without class conditioning using saved logits. Cross entropy is recomputed over the author's full 1025-class output; top-1 accuracy is measured only at 128 masked positions per image. This is not iterative generation or channel recovery. See `evidence/stage56_prior_probe_audit.json`.\n'''
if (root/'runs/stage57_comparison_audit.json').exists():
    matched=json.loads((root/'runs/stage57_comparison_audit.json').read_text(encoding='utf-8'))
    assert matched['all_checks_passed'] and not matched.get('snapshot_only',False)
    readme+=f'''\n## Separate matched end-to-end conventions (stage 57)\n
The preregistered old-development comparison now has {matched['counts']['demo']}/50 demo-pipeline cases and {matched['counts']['training_code_convention']}/50 training-code-convention cases, representing {matched['complete_pairs']}/50 complete image/trial pairs. Four paths are recorded per case. {matched['reused_demo_cases']} demo cases are reused only after exact identity checks; {matched['computed_cases']} cases are computed by this study. Both PHY streams have 50 audited receptions; 50 new simulations and 50 inherited demo PHY references are distinguished.\n
The two conventions are x -> raw decode y, and 2*x-1 -> (y+1)/2 for every metric path. Same images, 8 dB, noise seeds, code/modulation, class modes and generation parameters are retained. Each stream uses its own true-error oracle mask; both class modes share that mask within the stream. No cross-pipeline quality means are released until all 50 pairs and all four paths are complete. See `evidence/stage57_comparison_audit.json`; this independent namespace does not alter the frozen primary or the original six-SNR grid.\n
Resume `run_stage57_convention_comparison.py`, audit with `audit_stage57_convention_comparison.py`, then render a session with `report_convention_comparison.py --session runs/SESSION.json`. The frozen reuse-source manifest prevents later original-grid progress from changing this study's initial identities.\n'''
if (root/'runs/stage59_session_progress.json').exists():
    continuation=json.loads((root/'runs/stage59_session_progress.json').read_text(encoding='utf-8'))
    readme+=f'''\n## Resumed six-SNR model study (stage 59)\n
The matched convention study is complete at 100 pipeline receptions and 50 pairs; its frozen audit and all 100 case files remain unchanged. Stage 59 resumes the original independently locked six-SNR model study from 72/300. Current model coverage: {continuation['paired_model_cases']}/300; new model cases this window: {continuation['new_paired_model_cases']}. PHY remains 300/300, with {continuation['new_phy_cases']} new physical simulations this window. No matched-convention case is transferred into the original grid. Incomplete SNR groups receive coverage only, without quality means.\n
Publication follows draft PR #{continuation.get('github_pull_request',2)} on `{continuation.get('github_branch','reproduction/automation-30-stage58')}`. Use `sync_github_worktree_from_api.py --branch BRANCH --commit SHA --expected-parent PARENT --receipt runs/RECEIPT.json` for exact tree/commit synchronization on the selected existing branch. The explicit branch is checked before staging or ref mutation; isolated Git contract checks cover rejection and idempotence.\n'''
if (root/'runs/stage60_session_progress.json').exists():
    randomized=json.loads((root/'runs/stage60_session_progress.json').read_text(encoding='utf-8'))
    readme+=f'''\n## Independent randomized packet assignment (stage 60)\n
The prior fixed-grid study stays complete at 300/300, the primary holdout at 500/500 and the convention study at 100/100. A separately preregistered study now has randomized PHY {randomized['randomized_phy_cases']}/300 and four-path models {randomized['randomized_model_cases']}/300. The {randomized['diagnostic_phy_transmissions_total']} diagnostic transmissions are separate from formal receptions. The fixed complete baseline audit was pinned before first new inference.\n
Each image has a fixed PCG64 permutation shared across five trials and six SNRs. Packet errors are defined in transmitted packet coordinates; received IDs and both masks are inverse-permuted before model processing. All other model/channel/metric identities are retained. Completed SNR groups alone receive paired quality means; current complete paired groups: {randomized['complete_paired_quality_snr_groups']}/6. This is an old-Imagenette sensitivity study, not paper-wide ImageNet100 replication.\n
Resume `run_stage60_randomized_recovery.py`, then `audit_stage60_randomized_recovery.py` and `audit_stage60_paired_packetization.py`. Render `report_randomized_session.py --session runs/SESSION.json`. Keep the execution spec, permutation helper, scripts and locks unchanged. The pure permutation/mask checks, actual auditor fixtures and paired-aggregation gates have separate bounded receipts. Noisy high-SNR and exact noiseless diagnostic checks are labeled separately.\n'''
if (root/'runs/stage61_session_progress.json').exists():
    current=json.loads((root/'runs/stage61_session_progress.json').read_text(encoding='utf-8'))
    readme+=f'''\n## Completed pairing and actual packet ARQ (stage 61)\n
The randomized four-path model study is complete at {current['randomized_model_cases']}/300, including {current['new_randomized_model_cases']} new cases this round and all six complete fixed/random pairs. All four paths and per-image differences are retained without choosing a winning permutation.\n
The separate padded-packet experiment completes {current['packet_arq_frame_experiments']}/300 frames and {current['packet_arq_metric_rows']} VQGAN endpoint metric rows across single pass, receiver CRC ARQ and sender-oracle ARQ. Attempts are simulated with independent packet/attempt noise, capped at 32, with wrong CRC acceptances and censoring retained. Each first pass is 1664 QAM symbols (40 payload + 11 CRC bits, half-rate convolution, 2 padding bits per packet). Forward data only: feedback/header/CMI airtime is not modeled. The earlier 1632-symbol MaskGIT results are not a matching-framing baseline.\n
Metric accounting records {current['arq_decoder_forwards_this_invocation']} actual decoder forwards and {current['arq_exact_clean_reference_rows']} rows using exact same-image clean-ID references. No MaskGIT inference is claimed for ARQ endpoints. Two noiseless packet diagnostics and 14 pure accounting/metric checks are separate from 300 formal frames. The initial Mapper shape preflight correction and its replaced lock are preserved.\n
Next: `stage61_matched_padded_recovery_plan.json` and {current['matched_padded_inputs']} audited canonical first-pass inputs prepare an independent six-path neural comparison under the same 1664-symbol framing. Implement and lock its runner/auditor before new inference. This round performs zero Transformer recovery cases in that new study. Keep all prior frozen studies unchanged; ImageNet100, Cheng and complete resource-equivalent paper replication remain open.\n'''
if (root/'runs/stage62_matched_recovery_audit.json').exists():
    frozen=root/'runs/stage62_session_audit.json'
    active=json.loads((frozen if frozen.exists() else root/'runs/stage62_matched_recovery_audit.json').read_text(encoding='utf-8'))
    readme+=f"\n## Matched padded neural recovery (stage 62)\n\nImplemented the separately locked six-path runner and auditor. Audited coverage: {active['completed_cases']}/300 receptions, six paths each. Uses the 300 existing 1664-symbol first-pass inputs, separate oracle/CRC masks, and frozen model/metric conventions. No new PHY and no copying old unpadded generated endpoints. Incomplete SNR groups receive coverage only. Resume `run_stage62_matched_recovery.py`, then `audit_stage62_matched_recovery.py` and `report_stage62_matched.py`. The six pure mask/metric validation checks are independent of real model evaluations.\n"
if (root/'runs/stage63_session_audit.json').exists():
    active=json.loads((root/'runs/stage63_session_audit.json').read_text(encoding='utf-8'))
    readme+=f"\n## Matched padded continuation (stage 63)\n\nContinued from 82 cases with the identical runner, inputs and execution lock. Current independently audited coverage: {active['completed_cases']}/300 receptions; new this round: {active['completed_cases']-82}. The first 82 case hashes remain unchanged. Stage62 session receipts are frozen separately. Resume the same locked runner for missing cases. Quality means require 50 receptions per SNR; no incomplete-SNR means are published.\n"
(target/'README.md').write_text(readme,encoding='utf-8')
files={}
for path in target.rglob('*'):
    if path.is_file() and path.name!='SNAPSHOT_SHA256.json':
        assert path.stat().st_size<5_000_000
        files[path.relative_to(target).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
(target/'SNAPSHOT_SHA256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print('snapshot files',len(files),'bytes',sum((target/p).stat().st_size for p in files))
