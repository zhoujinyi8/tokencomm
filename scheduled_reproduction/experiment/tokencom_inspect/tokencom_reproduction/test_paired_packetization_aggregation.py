"""Exercise completeness gates and paired image means with independent fixtures."""
import copy
import hashlib
import json
from pathlib import Path

from paired_packetization_aggregation import PATHS,paired_snr_summary


def fixtures(offset):
    return [dict(image_index=i,trial=t,snr_db=8.,rows=[dict(path=p,correct_tokens=200.,whole_image_ter=56/256,
             psnr_raw=20+i+t/10+offset,lpips_normalized=.2,clip_clipped=.8) for p in PATHS])
            for i in range(10) for t in range(5)]


def verify():
    fixed=fixtures(0);randomized=fixtures(2)
    result=paired_snr_summary(fixed[::-1],randomized[::2]+randomized[1::2])
    assert abs(result['means_over_images']['fixed']['direct']['psnr_raw']-24.7)<1e-12
    assert abs(result['means_over_images']['randomized']['direct']['psnr_raw']-26.7)<1e-12
    assert all(abs(row['direct']['psnr_raw']-2)<1e-12 for row in result['paired_image_differences'])
    assert abs(result['paired_mean_difference']['direct']['psnr_raw']-2)<1e-12
    failures=[(fixed,randomized[:-1]),(fixed,randomized[:-1]+[randomized[0]])]
    missing=copy.deepcopy(randomized);missing[0]['rows'].pop();failures.append((fixed,missing))
    changed=copy.deepcopy(randomized);changed[0]['snr_db']=7.;failures.append((fixed,changed))
    for left,right in failures:
        try:paired_snr_summary(left,right)
        except AssertionError:pass
        else:raise AssertionError('Incomplete or mismatched cohort accepted')
    root=Path(__file__).parent
    return dict(passed=5,failed=0,model_inference_performed=False,
                helper_sha256=hashlib.sha256((root/'paired_packetization_aggregation.py').read_bytes()).hexdigest(),
                test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                checks=['complete unordered paired means','missing reception refused','duplicate refused','missing path refused','SNR mismatch refused'])


if __name__=='__main__':
    result=verify()
    (Path(__file__).parent/'runs/stage60_paired_aggregation_contract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
