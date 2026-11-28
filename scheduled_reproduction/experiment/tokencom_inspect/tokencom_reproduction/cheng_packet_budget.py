"""Integer first-pass framing ledger for a diagnostic byte container (no channel simulation)."""
def packet_budget(container_bytes,header_bytes=34):
    if not isinstance(container_bytes,int) or not isinstance(header_bytes,int) or container_bytes<=header_bytes or header_bytes<0:
        raise ValueError('invalid byte lengths')
    packets=(container_bytes+4)//5;tail_bytes=5*packets-container_bytes
    return dict(container_bytes=container_bytes,container_header_bytes=header_bytes,entropy_payload_bytes=container_bytes-header_bytes,
        packet_payload_bits=40,packets=packets,tail_padding_bits=tail_bytes*8,
        crc_bits=11*packets,convolutional_bits=102*packets,qam_padding_bits=2*packets,
        forward_wire_bits=104*packets,forward_qam_symbols=26*packets,
        feedback_bits=None,channel_simulated=False,author_paper_header_identified=False)
def packetize(data):
    b=packet_budget(len(data));padded=data+bytes(b['tail_padding_bits']//8)
    return [padded[i:i+5] for i in range(0,len(padded),5)],b
def reassemble(packets,original_size):
    if not packets or any(not isinstance(p,bytes) or len(p)!=5 for p in packets):raise ValueError('invalid packet sizes')
    b=b''.join(packets)
    if original_size<=0 or not 0<=len(b)-original_size<5 or any(b[original_size:]):raise ValueError('invalid tail')
    return b[:original_size]
