"""Packet byte identities, variable lengths, and policy-independent noise fixtures."""
import numpy as np
from cheng_byte_transport import byte_packets,packet_bytes,byte_packet_noise,noise_identity
from packet_arq_accounting import replay_trace


def verify():
    checks=[]
    for size in (1,4,5,6,34,256,321,700):
        data=bytes(i%256 for i in range(size));assert packet_bytes(byte_packets(data),size)==data
    checks.append('byte roundtrip including variable packet counts above 64')
    data=bytes(range(256));assert np.array_equal(byte_packets(data)[0],[0,16,128,772]);checks.append('explicit big-endian bit ordering')
    try:packet_bytes(np.zeros((2,4),np.int16),5)
    except ValueError:pass
    else:raise AssertionError('wrong byte count accepted')
    checks.append('wrong trailing length refused')
    for bad in (np.full((1,4),1024),np.full((1,4),-1),np.zeros((1,4),float),np.zeros((1,3),int)):
        try:packet_bytes(bad,5)
        except ValueError:continue
        raise AssertionError('invalid payload accepted')
    checks.append('out-of-range or malformed chunks refused')
    a=byte_packet_noise(2026112804,0,0,0,[0,65,139],1,1.)
    b=byte_packet_noise(2026112804,0,0,0,[139,0,65],1,1.)
    assert np.array_equal(a,b[[1,2,0]]);checks.append('policy and packet batching order invariance')
    assert not np.array_equal(a,byte_packet_noise(2026112804,0,0,0,[0,65,139],2,1.));checks.append('attempt noise independence')
    identities={noise_identity(2026112804,i,t,s,p,a) for i in (0,9) for t in (0,4) for s in range(6) for p in (0,63,64,139,65535) for a in (1,32)}
    assert len(identities)==240;checks.append('unambiguous variable-packet seed coordinates')
    sent=byte_packets(bytes(range(12)));rx=sent[None].copy();crc=np.ones((1,len(sent)),bool);active=crc.copy()
    result=replay_trace(sent,rx,crc,active,1)
    assert all(v['total_packet_attempts']==3 and v['forward_qam_symbols']==78 and v['final_payload_error_packets']==0 for v in result.values())
    checks.append('actual replay supports non-64 packet count')
    return dict(passed=len(checks),failed=0,checks=checks)


if __name__=='__main__':print(verify())
