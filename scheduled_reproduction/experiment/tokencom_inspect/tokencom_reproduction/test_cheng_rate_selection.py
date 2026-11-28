"""Selection gates reject incomplete grids and invalid measurements."""
import copy
from cheng_rate_selection import select_candidate


def verify():
    checks=[]
    def arm(name,rate):
        return dict(arm_id=name,complete=True,updates_completed=100,expected_updates=100,
                    calibration_cases=[dict(image_index=i,entropy_payload_bpp=rate,total_stored_bpp=rate+.00415,psnr_decoded=25.) for i in range(10)])
    base=[arm('a',.038),arm('b',.041),arm('c',.05)]
    def rejected(values):
        try:select_candidate(values,['a','b','c'])
        except ValueError:return True
        return False
    assert select_candidate(base,['a','b','c'])['selected_arm_id']=='b';checks.append('closest accepted entropy rate')
    assert rejected(base[:2]);checks.append('partial grid refused')
    altered=copy.deepcopy(base);altered[0]['updates_completed']=99
    assert rejected(altered);checks.append('partial optimizer budget refused')
    altered=copy.deepcopy(base);altered[0]['calibration_cases'][1]['image_index']=0
    assert rejected(altered);checks.append('duplicate calibration identity refused')
    altered=copy.deepcopy(base);altered[0]['calibration_cases'][0]['entropy_payload_bpp']=float('nan')
    assert rejected(altered);checks.append('nonfinite rate refused')
    altered=copy.deepcopy(base);altered[0]['calibration_cases'][0]['total_stored_bpp']=.001
    assert rejected(altered);checks.append('invalid container ledger refused')
    outside=[arm('a',.031),arm('b',.05),arm('c',.06)]
    assert select_candidate(outside,['a','b','c'])['selected_arm_id'] is None;checks.append('no passing candidate means no selection')
    tied=[arm('a',.04),arm('b',.04),arm('c',.04)]
    result=select_candidate(list(reversed(tied)),['a','b','c'])
    assert result['selected_arm_id']=='a' and not result['paper_baseline_ready'];checks.append('frozen tie order and development-only scope')
    # A rate outside the entropy window cannot pass merely because headers bring it inside.
    header=[arm('a',.035),arm('b',.03),arm('c',.03)]
    assert select_candidate(header,['a','b','c'])['selected_arm_id'] is None;checks.append('container overhead cannot qualify entropy rate')
    return dict(passed=len(checks),failed=0,checks=checks)


if __name__=='__main__':print(verify())
