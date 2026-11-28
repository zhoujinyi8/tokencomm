"""Fail-closed, development-only selection from a complete fixed-budget grid."""
import math


def select_candidate(arms, expected_ids, target=.04, relative_tolerance=.1, cases_per_arm=10):
    if not (math.isfinite(target) and target > 0 and math.isfinite(relative_tolerance) and 0 <= relative_tolerance < 1):
        raise ValueError('invalid target or tolerance')
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('invalid frozen grid')
    if len(arms) != len(expected_ids) or {a['arm_id'] for a in arms} != set(expected_ids):
        raise ValueError('complete frozen grid required')
    rows=[]
    for arm in arms:
        cases=arm['calibration_cases']
        if not arm['complete'] or arm['updates_completed'] != arm['expected_updates'] or arm['expected_updates'] <= 0:
            raise ValueError('incomplete training')
        if len(cases) != cases_per_arm or {c['image_index'] for c in cases} != set(range(cases_per_arm)):
            raise ValueError('complete calibration required')
        for case in cases:
            rate=case['entropy_payload_bpp'];total=case['total_stored_bpp'];psnr=case['psnr_decoded']
            if not all(math.isfinite(v) for v in (rate,total,psnr)) or rate <= 0 or total < rate:
                raise ValueError('invalid measured rate or quality')
        rate=math.fsum(c['entropy_payload_bpp'] for c in cases)/cases_per_arm
        rows.append(dict(arm_id=arm['arm_id'],mean_entropy_payload_bpp=rate,
                         mean_total_stored_bpp=math.fsum(c['total_stored_bpp'] for c in cases)/cases_per_arm,
                         mean_psnr=math.fsum(c['psnr_decoded'] for c in cases)/cases_per_arm,
                         within_rate_window=abs(rate-target) <= target*relative_tolerance))
    eligible=[a for a in rows if a['within_rate_window']]
    chosen=min(eligible,key=lambda a:(abs(a['mean_entropy_payload_bpp']-target),expected_ids.index(a['arm_id']))) if eligible else None
    return dict(rows=rows,selected_arm_id=chosen['arm_id'] if chosen else None,
                selection_scope='development calibration only',paper_baseline_ready=False,
                lower_rate_bound=target*(1-relative_tolerance),upper_rate_bound=target*(1+relative_tolerance))
