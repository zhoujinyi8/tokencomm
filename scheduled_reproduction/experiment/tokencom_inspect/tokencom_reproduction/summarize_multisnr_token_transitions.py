"""Describe exact-ID repair and collateral damage only for a complete six-SNR study."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def count_transitions(sent, received, final, mask):
    sent, received, final, mask = map(np.asarray, (sent, received, final, mask))
    assert sent.ndim == 1 and sent.shape == received.shape == final.shape == mask.shape
    assert mask.dtype == np.bool_
    assert np.array_equal(final[~mask], received[~mask]), 'Unmasked token changed'
    before = received == sent; after = final == sent
    assert before[~mask].all(), 'Oracle mask omitted an original error'
    repair = int((~before & after).sum())
    damage = int((before & ~after).sum())
    result = dict(tokens=sent.size, masked_tokens=int(mask.sum()),
                  wrong_before=int((~before).sum()), wrong_after=int((~after).sum()),
                  repaired=repair, damaged=damage,
                  masked_correct_preserved=int((mask & before & after).sum()),
                  wrong_still_wrong=int((~before & ~after).sum()),
                  wrong_to_different_wrong=int((~before & ~after & (final != received)).sum()),
                  net_correct_gain=repair-damage)
    assert result['wrong_after'] == result['wrong_before']-repair+damage
    assert result['wrong_before'] == repair+result['wrong_still_wrong']
    assert result['masked_tokens'] == (result['wrong_before']+damage+result['masked_correct_preserved'])
    return result


def summarize(audit_path, output_path):
    root = Path(__file__).resolve().parent; run = root/'runs'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert audit_path.resolve().parent == output_path.resolve().parent == run.resolve()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    audit = read(audit_path); plan_path = run/'stage54_development_multisnr_plan.json'
    plan = read(plan_path)
    assert audit['all_checks_passed'] and audit['plan_sha256'] == sha(plan_path)
    result = dict(analysis='descriptive exact-ID transitions; no strategy selection',
                  source_audit_sha256=sha(audit_path), script_sha256=sha(Path(__file__)),
                  plan_sha256=sha(plan_path), complete=audit['complete'],
                  completed_cases=audit['completed_cases'], new_model_inference_performed=False,
                  primary_holdout_changed=False, outcome_counts_aggregated=False)
    if audit['complete']:
        assert audit['completed_cases'] == 300 and len(audit['case_sha256']) == 300
        assert len(audit['complete_snr_groups']) == 6
        phy_audit = read(run/'stage54_development_phy_audit.json')
        source = root/'runs/stage26_official_token_link_snr8_5trials.npz'
        assert sha(source) == plan['input_sha256'][source.relative_to(root).as_posix()]
        with np.load(source) as saved: transmitted = saved['transmitted'].copy()
        per_case = []
        for name, digest in sorted(audit['case_sha256'].items()):
            path = run/'stage54_development_recovery'/name
            assert sha(path) == digest
            case = read(path); latent = path.with_suffix('.npz')
            phy = run/'stage54_development_phy'/latent.name
            assert sha(latent) == case['latent_sha256']
            assert sha(phy) == case['source_phy_sha256'] == phy_audit['case_sha256'][phy.name]
            sent = transmitted[case['image_index']].reshape(256)
            with np.load(phy) as original, np.load(latent) as final:
                rx = original['received'].reshape(256); mask = original['oracle_token_mask'].reshape(256).astype(bool)
                assert np.array_equal(final['direct'], rx) and np.array_equal(final['ideal_correct_tokens'], sent)
                rows = {}
                for mode in plan['paths']:
                    values = count_transitions(sent, rx, final[mode], mask)
                    metric = next(row for row in case['rows'] if row['path'] == mode)
                    assert metric['correct_tokens'] == 256-values['wrong_after']
                    rows[mode] = values
            per_case.append(dict(snr_db=case['snr_db'], image_index=case['image_index'], trial=case['trial'], paths=rows))
        groups = {}
        for snr in plan['snr_db']:
            subset = [row for row in per_case if row['snr_db'] == snr]
            assert {(c['image_index'],c['trial']) for c in subset} == {(i,t) for i in range(10) for t in range(5)}
            totals = {mode:{key:sum(row['paths'][mode][key] for row in subset)
                            for key in subset[0]['paths'][mode]} for mode in plan['paths']}
            for mode, values in totals.items():
                expected_ter = audit['complete_snr_groups'][str(snr)]['means_over_images'][mode]['whole_image_ter']
                assert np.isclose(values['wrong_after']/12800, expected_ter, atol=1e-12, rtol=0)
                assert values['tokens'] == 12800
            per_image = [{mode:{key:sum(row['paths'][mode][key] for row in subset if row['image_index'] == i)/5
                                for key in subset[0]['paths'][mode]} for mode in plan['paths']} for i in range(10)]
            groups[str(snr)] = dict(receptions=50, totals=totals, per_image_trial_means=per_image)
        result.update(outcome_counts_aggregated=True, groups=groups, receptions_audited=300,
                      path_rows_audited=1200, all_checks_passed=True)
    else:
        result.update(all_checks_passed=True, groups={}, reason='Incomplete cohort: outcome counts withheld')
    output_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',default='runs/stage59_development_recovery_audit.json',type=Path)
    parser.add_argument('--output',default='runs/stage59_token_transitions.json',type=Path)
    args=parser.parse_args()
    result=summarize(args.audit,args.output)
    print(json.dumps({key:value for key,value in result.items() if key != 'groups'},indent=2))
