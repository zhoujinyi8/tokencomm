"""Exercise actual audit entry point with small isolated, non-model fixtures."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_root(tmp_path, receptions):
    script = Path(__file__).with_name('audit_stage54_development_recovery.py')
    shutil.copy2(script, tmp_path / script.name)
    (tmp_path / 'run_stage54_development_multisnr.py').write_text('# fixture\n', encoding='utf-8')
    run = tmp_path / 'runs'
    run.mkdir()
    phy_dir = run / 'stage54_development_phy'
    model_dir = run / 'stage54_development_recovery'
    phy_dir.mkdir()
    model_dir.mkdir()
    paths = ['direct', 'without_cmi', 'with_cmi', 'ideal_correct_tokens']
    plan = {'scope': 'isolated audit contract fixture', 'snr_db': [6., 6.5, 7., 7.5, 8., 9.], 'paths': paths}
    write_json(run / 'stage54_development_multisnr_plan.json', plan)
    plan_sha = digest(run / 'stage54_development_multisnr_plan.json')
    lock = {'script_sha256': digest(tmp_path / 'run_stage54_development_multisnr.py'), 'plan_sha256': plan_sha}
    write_json(run / 'stage54_development_recovery_lock.json', lock)
    lock_sha = digest(run / 'stage54_development_recovery_lock.json')
    write_json(run / 'stage47_complete_holdout_summary.json', {'fixture': True})
    sent = np.zeros((10, 256), dtype=np.int64)
    np.savez(run / 'stage26_official_token_link_snr8_5trials.npz', transmitted=sent)
    rows = [{'image_sha256': str(i)} for i in range(10)]
    np.savez(run / 'stage26_official_demo_tokens.npz', metadata=json.dumps({'rows': rows}))
    phy_hashes, case_hashes = {}, {}
    for index in range(receptions):
        image, trial = divmod(index, 5)
        stem = f'snr0_image{image:02d}_trial{trial}'
        phy_path = phy_dir / f'{stem}.npz'
        latent_path = model_dir / f'{stem}.npz'
        mask = np.zeros(256, dtype=bool)
        mask[:4] = True
        rx = sent[image].copy()
        rx[0] = 1
        np.savez(phy_path, oracle_token_mask=mask, received=rx)
        np.savez(latent_path, direct=rx, without_cmi=rx, with_cmi=sent[image], ideal_correct_tokens=sent[image])
        values = []
        for name in paths:
            correct = 255 if name in ('direct', 'without_cmi') else 256
            values.append({'path': name, 'correct_tokens': correct, 'whole_image_ter': 1-correct/256,
                           'steps': 0, 'psnr_raw': 20.+image, 'lpips_normalized': .2, 'clip_clipped': .8})
        case = {'snr_index': 0, 'snr_db': 6., 'image_index': image, 'trial': trial,
                'source_sha256': str(image), 'source_phy_sha256': digest(phy_path),
                'lock_sha256': lock_sha, 'latent_sha256': digest(latent_path), 'masked_tokens': 4, 'rows': values}
        path = model_dir / f'{stem}.json'
        write_json(path, case)
        phy_hashes[phy_path.name] = digest(phy_path)
        case_hashes[path.name] = digest(path)
    write_json(run / 'stage54_development_recovery_progress.json',
               {'plan_sha256': plan_sha, 'lock_sha256': lock_sha, 'completed_cases': receptions, 'case_sha256': case_hashes})
    write_json(run / 'stage54_development_phy_audit.json',
               {'plan_sha256': plan_sha, 'all_checks_passed': True, 'case_sha256': phy_hashes,
                'complete_snr_groups': {'6.0': {'fixture_analytic_reference': True}} if receptions == 50 else {}})
    return run


def execute(tmp_path):
    return subprocess.run([sys.executable, str(tmp_path / 'audit_stage54_development_recovery.py')],
                          capture_output=True, text=True, timeout=30)


def test_incomplete_snr_reports_coverage_without_means(tmp_path):
    run = fixture_root(tmp_path, 1)
    result = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    audit = json.loads((run / 'stage54_development_recovery_audit.json').read_text())
    assert audit['counts_by_snr']['6.0'] == 1
    assert audit['complete_snr_groups'] == {}
    assert audit['complete'] is False


def test_complete_snr_aggregates_all_images_and_trials(tmp_path):
    run = fixture_root(tmp_path, 50)
    result = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    audit = json.loads((run / 'stage54_development_recovery_audit.json').read_text())
    assert set(audit['complete_snr_groups']) == {'6.0'}
    group = audit['complete_snr_groups']['6.0']
    assert group['means_over_images']['direct']['psnr_raw'] == 24.5
    assert group['difference_with_minus_without_cmi']['correct_tokens'] == 1
    assert audit['complete'] is False


def check_corrupted_output(tmp_path, corruption):
    run = fixture_root(tmp_path, 1)
    case_path = run / 'stage54_development_recovery/snr0_image00_trial0.json'
    case = json.loads(case_path.read_text())
    if corruption == 'metric_count':
        case['rows'][0]['correct_tokens'] = 254
    else:
        path = case_path.with_suffix('.npz')
        with np.load(path) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        arrays['with_cmi'][10] = 1
        np.savez(path, **arrays)
        case['latent_sha256'] = digest(path)
        case['rows'][2]['correct_tokens'] = 255
        case['rows'][2]['whole_image_ter'] = 1/256
    write_json(case_path, case)
    progress_path = run / 'stage54_development_recovery_progress.json'
    progress = json.loads(progress_path.read_text())
    progress['case_sha256'][case_path.name] = digest(case_path)
    write_json(progress_path, progress)
    result = execute(tmp_path)
    assert result.returncode != 0
    assert 'AssertionError' in result.stderr
    assert not (run / 'stage54_development_recovery_audit.json').exists()


def test_audit_rejects_wrong_token_count(tmp_path):
    check_corrupted_output(tmp_path, 'metric_count')


def test_audit_rejects_unmasked_token_change(tmp_path):
    check_corrupted_output(tmp_path, 'unmasked_token')


if __name__ == '__main__':
    checks = [test_incomplete_snr_reports_coverage_without_means,
              test_complete_snr_aggregates_all_images_and_trials,
              test_audit_rejects_wrong_token_count, test_audit_rejects_unmasked_token_change]
    for check in checks:
        with tempfile.TemporaryDirectory(prefix='stage54_audit_') as directory:
            check(Path(directory))
        print('PASS', check.__name__)
    print('4 audit contract checks passed; isolated fixtures, no model inference')
