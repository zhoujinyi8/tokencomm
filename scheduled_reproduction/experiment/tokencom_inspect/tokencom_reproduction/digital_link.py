"""Bit-level reference link. Explicit engineering choices, not undisclosed author settings.

CRC-16/CCITT-FALSE; K=7 (171,133) octal rate-1/2 convolutional code;
zero termination; unit-average-energy Gray 16-QAM; max-log bit LLRs;
terminated soft-input Viterbi. SNR is Es/N0, NOT Eb/N0.
"""
import math
import numpy as np


def crc16(bits):
    crc=0xffff
    for bit in bits:
        feedback=((crc>>15)&1)^int(bit)
        crc=(crc<<1)&0xffff
        if feedback: crc^=0x1021
    return crc


def int_bits(value,width):
    return ((np.asarray(value,dtype=np.int64)[...,None]>>np.arange(width-1,-1,-1))&1).astype(np.uint8)


def bits_int(bits):
    return np.asarray(bits,dtype=np.int64)@(1<<np.arange(bits.shape[-1]-1,-1,-1))


def append_crc(bits):
    return np.concatenate([bits,int_bits(crc16(bits),16)])


def check_crc(bits):
    return len(bits)>=16 and crc16(bits[:-16])==int(bits_int(bits[-16:]))


# Register bit 0 is the newest input; state contains the previous six bits.
STATES=np.arange(64)
NEXT=np.empty((64,2),dtype=np.int64)
OUTPUT=np.empty((64,2,2),dtype=np.uint8)
for s in range(64):
    for u in range(2):
        reg=(s<<1)|u
        NEXT[s,u]=reg&63
        OUTPUT[s,u]=[(reg & g).bit_count()%2 for g in (0o171,0o133)]
PREV=np.stack([STATES>>1,(STATES>>1)|32],1)
INPUT=STATES&1
EXPECTED=OUTPUT[PREV,INPUT[:,None]]


def conv_encode(bits):
    state=0; out=[]
    for bit in np.concatenate([bits,np.zeros(6,dtype=np.uint8)]):
        out.extend(OUTPUT[state,int(bit)])
        state=NEXT[state,int(bit)]
    return np.asarray(out,dtype=np.uint8)


def viterbi(llr):
    """Positive LLR favors zero; output excludes six terminating zero bits."""
    llr=np.asarray(llr,dtype=float)
    if len(llr)%2 or len(llr)<12: raise ValueError('Invalid terminated codeword length')
    metric=np.full(64,np.inf); metric[0]=0
    decisions=np.empty((len(llr)//2,64),dtype=np.uint8)
    for t,soft in enumerate(llr.reshape(-1,2)):
        # Negative log likelihood up to a branch-independent additive constant.
        cost=(EXPECTED*soft).sum(-1)
        candidates=metric[PREV]+cost
        chosen=candidates.argmin(1)
        metric=candidates[STATES,chosen]
        metric-=metric.min()
        decisions[t]=chosen
    state=0; decoded=np.empty(len(decisions),dtype=np.uint8)
    for t in range(len(decisions)-1,-1,-1):
        decoded[t]=state&1; state=PREV[state,decisions[t,state]]
    return decoded[:-6]


BIT_LABELS=int_bits(np.arange(16),4)
GRAY_LEVELS=np.array([-3,-1,3,1],dtype=float)
CONSTELLATION=(GRAY_LEVELS[bits_int(BIT_LABELS[:,:2])]
               +1j*GRAY_LEVELS[bits_int(BIT_LABELS[:,2:])])/math.sqrt(10)


def qam16(bits):
    bits=np.asarray(bits,dtype=np.uint8)
    padding=(-len(bits))%4
    return CONSTELLATION[bits_int(np.pad(bits,(0,padding)).reshape(-1,4))],padding


def qam_llr(received,h,n0):
    distance=np.abs(received[:,None]-h[:,None]*CONSTELLATION[None])**2
    return np.stack([(distance[:,BIT_LABELS[:,k]==1].min(1)
                      -distance[:,BIT_LABELS[:,k]==0].min(1))/n0 for k in range(4)],1).reshape(-1)


def transmit_packet(payload,snr_db,rng,channel='awgn'):
    if channel not in ('awgn','rayleigh','noiseless'): raise ValueError(channel)
    if not math.isfinite(float(snr_db)): raise ValueError('SNR must be finite')
    payload=np.asarray(payload,dtype=np.uint8)
    if payload.ndim!=1 or not np.isin(payload,[0,1]).all(): raise ValueError('Expected binary payload')
    code=conv_encode(append_crc(payload)); symbols,padding=qam16(code)
    n0=10.**(-float(snr_db)/10.) if channel!='noiseless' else 1e-12
    if channel=='rayleigh':
        h=(rng.standard_normal(len(symbols))+1j*rng.standard_normal(len(symbols)))/math.sqrt(2)
    else: h=np.ones(len(symbols),dtype=complex)
    noise=np.zeros(len(symbols),complex) if channel=='noiseless' else (
        rng.standard_normal(len(symbols))+1j*rng.standard_normal(len(symbols)))*math.sqrt(n0/2)
    llr=qam_llr(h*symbols+noise,h,n0)[:len(code)]
    decoded=viterbi(llr); ok=check_crc(decoded)
    return decoded[:len(payload)],ok,{
        'symbols':len(symbols),'coded_bits':len(code),'qam_padding_bits':padding,
        'payload_bit_errors':int(np.count_nonzero(decoded[:len(payload)]!=payload)),
        'raw_coded_bit_errors':int(np.count_nonzero((llr<0)!=code))}


def transmit_tokens(ids,codebook_size=1024,snr_db=8.,channel='awgn',seed=42,packet_tokens=4):
    """Receiver only uses decoded bits and CRC. Ground truth is used for metrics only.

    CRC-undetected errors remain in receiver IDs. Never use TX IDs to repair them.
    Shared interleaver and packet framing are assumed known; overhead excludes them.
    """
    ids=np.asarray(ids)
    if ids.ndim!=2 or ids.size==0 or not np.issubdtype(ids.dtype,np.integer): raise ValueError('Expected nonempty integer [B,N]')
    if codebook_size<2 or codebook_size&(codebook_size-1): raise ValueError('Use power-of-two codebook size')
    if packet_tokens<1 or np.any(ids<0) or np.any(ids>=codebook_size): raise ValueError('Invalid token data')
    width=int(math.log2(codebook_size)); rng=np.random.default_rng(seed)
    b,n=ids.shape; received=np.zeros_like(ids); missing=np.zeros_like(ids,dtype=bool)
    stat={k:0 for k in ['packets','crc_failed_packets','undetected_error_packets','payload_bit_errors',
                        'payload_bits','coded_bits','symbols','qam_padding_bits','raw_coded_bit_errors']}
    order=rng.permutation(n)
    for i in range(b):
        for start in range(0,n,packet_tokens):
            positions=order[start:start+packet_tokens]
            payload=int_bits(ids[i,positions],width).reshape(-1)
            decoded,ok,detail=transmit_packet(payload,snr_db,rng,channel)
            stat['packets']+=1; stat['crc_failed_packets']+=int(not ok)
            stat['undetected_error_packets']+=int(ok and detail['payload_bit_errors']>0)
            stat['payload_bits']+=len(payload)
            for k,v in detail.items(): stat[k]+=v
            if ok: received[i,positions]=bits_int(decoded.reshape(-1,width))
            else: missing[i,positions]=True
    stat.update(snr_db=float(snr_db),snr_definition='Es/N0',channel=channel,
                measured_per=stat['crc_failed_packets']/stat['packets'],
                payload_ber=stat['payload_bit_errors']/stat['payload_bits'],
                observed_token_erasure=float(missing.mean()),token_bits=width,
                packet_tokens=packet_tokens,codebook_size=codebook_size,
                coded_bits_per_source_bit=stat['coded_bits']/stat['payload_bits'])
    return received,missing,stat
