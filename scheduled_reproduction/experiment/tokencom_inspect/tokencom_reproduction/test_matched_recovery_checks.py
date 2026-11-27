import copy
import numpy as np
from matched_recovery_checks import PATHS,validate

def verify():
    tx=np.zeros(256,dtype=np.int16);rx=tx.copy();rx[0]=1
    oracle=np.zeros(256,bool);oracle[:4]=True
    crc=np.zeros(256,bool)  # explicitly preserve a CRC false acceptance
    codes={p:(tx.copy() if p=='ideal_correct_tokens' else rx.copy()) for p in PATHS}
    rows=[dict(path=p,correct_tokens=int((codes[p]==tx).sum()),whole_image_ter=float((codes[p]!=tx).mean()),
               steps=0,psnr_raw=20.,lpips_normalized=.2,clip_clipped=.8) for p in PATHS]
    validate(rows,codes,rx,tx,oracle,crc)
    bad=[]
    c=copy.deepcopy(codes);c['receiver_crc_with_cmi'][0]=0;bad.append((rows,c))
    c=copy.deepcopy(codes);c['oracle_with_cmi'][4]=2;bad.append((rows,c))
    q=copy.deepcopy(rows);q[0]['correct_tokens']=0;bad.append((q,codes))
    q=copy.deepcopy(rows);q[0]['psnr_raw']=float('nan');bad.append((q,codes))
    bad.append((rows[:-1],codes))
    for q,c in bad:
        try:validate(q,c,rx,tx,oracle,crc)
        except AssertionError:continue
        raise AssertionError('invalid result accepted')
    return dict(passed=6,failed=0)
if __name__=='__main__':print(verify())
