"""Independent stopping-rule replay and integer costs for padded packet ARQ."""
import numpy as np

POLICIES=('single_pass_padded','receiver_crc_arq_padded','sender_oracle_arq_padded')


def packet_seed(base,image,trial,snr_index,packet,attempt):
    assert 0<=image<10 and 0<=trial<5 and 0<=snr_index<6 and 0<=packet<64 and 1<=attempt<=32
    return base+1000000*image+100000*trial+10000*snr_index+100*packet+attempt


def packet_noise(base,image,trial,snr_index,packets,attempt,variance):
    assert np.isfinite(variance) and variance>0
    rows=[]
    for packet in packets:
        rng=np.random.Generator(np.random.PCG64(packet_seed(base,image,trial,snr_index,int(packet),attempt)))
        values=rng.standard_normal((2,26)).astype(np.float32)
        rows.append((values[0]+1j*values[1])*np.float32(np.sqrt(variance/2)))
    return np.asarray(rows,dtype=np.complex64)


def replay_trace(transmitted,received,crc_valid,active,max_attempts):
    transmitted=np.asarray(transmitted);received=np.asarray(received)
    crc_valid=np.asarray(crc_valid,dtype=bool);active=np.asarray(active,dtype=bool)
    assert transmitted.ndim==2 and transmitted.shape[1]==4
    count=transmitted.shape[0]
    assert received.shape==(max_attempts,count,4) and crc_valid.shape==active.shape==(max_attempts,count)
    assert active[0].all() and ((received[active]>=0)&(received[active]<1024)).all()
    outputs={}
    for policy in POLICIES:
        attempts=[];final=[];accepted=[];final_crc=[]
        for packet in range(count):
            observed=np.flatnonzero(active[:,packet]);n=len(observed)
            assert np.array_equal(observed,np.arange(n))
            payload_correct=np.all(received[:n,packet]==transmitted[packet],axis=1)
            if policy=='single_pass_padded':
                chosen=0;success=True
            else:
                valid=crc_valid[:n,packet] if policy=='receiver_crc_arq_padded' else payload_correct
                passing=np.flatnonzero(valid);success=bool(len(passing))
                if success:chosen=int(passing[0])
                else:
                    assert n==max_attempts,'Unresolved policy trace truncated before cap'
                    chosen=n-1
            attempts.append(chosen+1);final.append(received[chosen,packet].copy())
            accepted.append(success);final_crc.append(bool(crc_valid[chosen,packet]))
        final=np.asarray(final);accepted=np.asarray(accepted,dtype=bool)
        wrong=np.any(final!=transmitted,axis=1);total=sum(attempts)
        outputs[policy]=dict(packet_attempts=attempts,total_packet_attempts=total,
            accepted_packets=int(accepted.sum()),censored_packets=int((~accepted).sum()),
            final_payload_error_packets=int(wrong.sum()),final_token_errors=int((final!=transmitted).sum()),
            final_crc_valid_packets=sum(final_crc),
            accepted_crc_payload_errors=int((wrong&accepted).sum()) if policy=='receiver_crc_arq_padded' else 0,
            attempted_payload_bits=40*total,crc_bits=11*total,convolutional_bits=102*total,padding_bits=2*total,
            forward_wire_bits=104*total,forward_qam_symbols=26*total,
            logical_feedback_events=total if policy!='single_pass_padded' else 0,
            feedback_bits=None,headers_bits=None,cmi_bits=None,
            payload_pixels_per_attempted_bit=65536/(40*total),forward_pixels_per_symbol=65536/(26*total))
        assert outputs[policy]['forward_wire_bits']==4*outputs[policy]['forward_qam_symbols']
    for packet in range(count):
        assert int(active[:,packet].sum())==max(outputs[p]['packet_attempts'][packet]
            for p in ('receiver_crc_arq_padded','sender_oracle_arq_padded')),'Unexpected observation after both policies stopped'
    return outputs
