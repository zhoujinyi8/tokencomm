"""Reject incorrect ID metrics, false clean reuse, nonfinite values and missing paths."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from arq_metric_checks import PATHS,validate_rows


def verify():
    sent=np.zeros(256,dtype=int);bad=sent.copy();bad[0]=1
    ids={PATHS[0]:bad,PATHS[1]:sent,PATHS[2]:sent}
    clean=dict(psnr_raw=20.,lpips_normalized=.2,clip_clipped=.8)
    rows=[dict(path=path,metric_source='decoded_endpoint' if i==0 else 'exact_clean_id_reference',
               correct_tokens=255 if i==0 else 256,whole_image_ter=1/256 if i==0 else 0.,**clean)
          for i,path in enumerate(PATHS)]
    validate_rows(rows,ids,sent,clean)
    failures=[]
    broken=copy.deepcopy(rows);broken[0]['correct_tokens']=254;failures.append(broken)
    broken=copy.deepcopy(rows);broken[0]['metric_source']='exact_clean_id_reference';failures.append(broken)
    broken=copy.deepcopy(rows);broken[1]['psnr_raw']=10.;failures.append(broken)
    broken=copy.deepcopy(rows);broken[0]['lpips_normalized']=float('nan');failures.append(broken)
    failures.append(rows[:-1])
    for broken in failures:
        try:validate_rows(broken,ids,sent,clean)
        except AssertionError:pass
        else:raise AssertionError('Invalid endpoint metrics accepted')
    root=Path(__file__).parent
    return dict(passed=6,failed=0,model_inference_performed=False,
        helper_sha256=hashlib.sha256((root/'arq_metric_checks.py').read_bytes()).hexdigest(),
        test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    result=verify()
    (Path(__file__).parent/'runs/stage61_arq_metric_contract.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
