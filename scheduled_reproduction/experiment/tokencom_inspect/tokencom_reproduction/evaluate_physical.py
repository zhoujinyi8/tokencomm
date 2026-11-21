"""Measure BER/PER from coded 16-QAM, and run the toy MaskGIT on receiver IDs."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from digital_link import transmit_tokens
from models import VQTokenizer,MaskGIT
from run import synthetic,batch


def wilson(k,n):
    z=1.95996398454; p=k/n; denominator=1+z*z/n
    mid=(p+z*z/(2*n))/denominator
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0,mid-half),min(1,mid+half)]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='runs/tokcom_refined/checkpoint.pt')
    p.add_argument('--out',default='runs/physical_sweep')
    p.add_argument('--packets',type=int,default=256,help='Independent 40-bit payload packets per channel/SNR setting')
    a=p.parse_args()
    if a.packets<1: p.error('--packets must be positive')
    torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck['model']!='tokcom' or ck['classes']!=3: raise ValueError('Toy 3-class checkpoint required')
    vq=VQTokenizer().eval(); vq.load_state_dict(ck['vq'])
    m=MaskGIT(classes=3).eval(); m.load_state_dict(ck['state'])
    im,_,_,labels=batch(synthetic(18,20260926),'cpu'); ids=vq.encode(im)
    # Separate PHY experiment uses the paper's 10-bit token width, four per packet.
    source=np.random.default_rng(4).integers(0,1024,(a.packets,4))
    rows=[]
    for channel in ['awgn','rayleigh']:
        for snr in [0.,4.,8.,12.,16.]:
            _,_,phy=transmit_tokens(source,1024,snr,channel,seed=42)
            received,mask,small=transmit_tokens(ids.numpy(),64,snr,channel,seed=42)
            restored=m.recover(torch.tensor(received),torch.tensor(mask),labels)
            rec=vq.decode(restored)
            low,high=wilson(phy['crc_failed_packets'],phy['packets'])
            row={'channel':channel,'snr_esn0_db':snr,'paper_width_packets':phy['packets'],
                 'paper_width_crc_per':phy['measured_per'],'per_wilson95_low':low,'per_wilson95_high':high,
                 'paper_width_payload_ber':phy['payload_ber'],'paper_width_undetected_packets':phy['undetected_error_packets'],
                 'paper_width_symbols_per_packet':phy['symbols']/phy['packets'],
                 'toy_six_bit_per':small['measured_per'],'toy_psnr_db':-10*math.log10(max(float(F.mse_loss(rec,im)),1e-12))}
            rows.append(row); print(row,flush=True)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps({'scope':'Explicit reference PHY settings; not author-exact PHY or ImageNet benchmark',
        'configuration':{'crc':'CRC-16/CCITT-FALSE','constraint_length':7,'generators_octal':['171','133'],
                         'termination':'six zero bits','qam':'unit-average-energy Gray 16-QAM',
                         'decoder':'max-log soft-input terminated Viterbi','rayleigh':'iid per symbol, perfect CSI',
                         'snr':'Es/N0'},'results':rows},indent=2),encoding='utf-8')
    with (out/'sweep.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__=='__main__': main()
