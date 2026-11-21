"""Copy explicit source/report/evidence allowlists into the isolated Git worktree."""
import hashlib
import json
import shutil
from pathlib import Path

root=Path(__file__).resolve().parent
checkout=root.parents[1]/'tokencomm_github_sync'
assert (checkout/'.git').is_file()
target=checkout/'scheduled_reproduction'
target.mkdir(parents=True,exist_ok=True)
(target/'.gitattributes').write_text('* -text\n',encoding='utf-8')
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
evidence=target/'evidence';evidence.mkdir(exist_ok=True)
for name in ['stage30_multitrial_progress.json','stage33_author_psnr_10images_5trials.json',
             'stage34_lpips_five_trials_summary.json','stage35_clip_summary.json',
             'stage36_joint_metrics.json','stage37_direct_comparison.json','stage39_transition_attribution.json',
             'stage40_gate_summary.json','stage41_soft_channel_diagnostics.json','stage43_soft_gate_summary.json',
             'stage44_frozen_protocol.json','stage44_holdout_audit.json','stage45_encoder_lock.json',
             'stage45_phy_lock.json','stage45_cache_audit.json',
             'stage46_recovery_lock.json','stage46_recovery_progress.json']:
    if (root/'runs'/name).exists():shutil.copy2(root/'runs'/name,evidence/name)
holdout=evidence/'stage44_holdout';holdout.mkdir(exist_ok=True)
for name in ('manifest.json','test.jsonl'):shutil.copy2(root/'data/imagenette_stage44_holdout'/name,holdout/name)
readme='''# Scheduled TokCom reproduction

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
'''
(target/'README.md').write_text(readme,encoding='utf-8')
files={}
for path in target.rglob('*'):
    if path.is_file() and path.name!='SNAPSHOT_SHA256.json':
        assert path.stat().st_size<5_000_000
        files[path.relative_to(target).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
(target/'SNAPSHOT_SHA256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print('snapshot files',len(files),'bytes',sum((target/p).stat().st_size for p in files))
