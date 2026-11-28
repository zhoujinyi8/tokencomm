"""Lossless byte/40-bit packet adapters and stateless variable-packet noise identities."""
import numpy as np


def byte_packets(data):
    if not isinstance(data,bytes) or not data:raise ValueError('nonempty bytes required')
    padded=data+bytes((-len(data))%5)
    bits=np.unpackbits(np.frombuffer(padded,dtype=np.uint8),bitorder='big').reshape(-1,4,10)
    return (bits*(1<<np.arange(9,-1,-1))).sum(axis=-1).astype(np.int16)


def packet_bytes(chunks,original_size):
    chunks=np.asarray(chunks)
    if chunks.ndim!=2 or chunks.shape[1]!=4 or not np.issubdtype(chunks.dtype,np.integer) or np.any((chunks<0)|(chunks>=1024)):
        raise ValueError('four ten-bit integers per packet required')
    if not isinstance(original_size,int) or original_size<=0 or not 0<=len(chunks)*5-original_size<5:
        raise ValueError('invalid original byte count')
    bits=((chunks[...,None]>>np.arange(9,-1,-1))&1).astype(np.uint8).reshape(-1)
    return np.packbits(bits,bitorder='big').tobytes()[:original_size]


def noise_identity(base,image,trial,snr_index,packet,attempt):
    values=(base,image,trial,snr_index,packet,attempt)
    if any(not isinstance(v,int) for v in values) or base<0 or not (0<=image<10 and 0<=trial<5 and 0<=snr_index<6 and 0<=packet<65536 and 1<=attempt<=32):
        raise ValueError('noise coordinate outside frozen protocol')
    return values


def byte_packet_noise(base,image,trial,snr_index,packets,attempt,variance):
    if not np.isfinite(variance) or variance<=0:raise ValueError('positive finite variance required')
    rows=[]
    for packet in packets:
        identity=noise_identity(base,image,trial,snr_index,int(packet),attempt)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(identity)))
        values=rng.standard_normal((2,26)).astype(np.float32)
        rows.append((values[0]+1j*values[1])*np.float32(np.sqrt(variance/2)))
    return np.asarray(rows,dtype=np.complex64)
