"""Explicit single-image Cheng entropy-stream container; no channel or integrity claim."""
import struct
MODELS=('cheng2020-anchor','cheng2020-attn')
HEADER=struct.Struct('>4sBBBBIIIIH');MAX_BYTES=64*1024*1024
def require(condition,message):
    if not condition:raise ValueError(message)
def pack(model,quality,height,width,shape,streams):
    require(model in MODELS and 1<=quality<=6,'unknown model/quality')
    require(len(shape)==2 and all(isinstance(v,int) and 0<v<=16384 for v in (height,width,*shape)),'invalid geometry')
    require(len(streams)==2 and all(isinstance(s,bytes) and 0<len(s)<=MAX_BYTES for s in streams),'invalid streams')
    header=HEADER.pack(b'CH65',1,MODELS.index(model),quality,0,height,width,*shape,2)
    return header+struct.pack('>II',*(len(s) for s in streams))+b''.join(streams)
def unpack(data):
    require(isinstance(data,bytes) and len(data)>=HEADER.size+8,'truncated header')
    magic,version,model,quality,flags,h,w,zh,zw,n=HEADER.unpack_from(data)
    require(magic==b'CH65' and version==1 and flags==0,'unknown format')
    require(model<len(MODELS) and 1<=quality<=6 and n==2,'unknown codec fields')
    require(all(0<v<=16384 for v in (h,w,zh,zw)),'invalid geometry')
    sizes=struct.unpack_from('>II',data,HEADER.size);offset=HEADER.size+8
    require(all(0<v<=MAX_BYTES for v in sizes),'invalid lengths')
    require(len(data)==offset+sum(sizes),'truncated payload or trailing bytes')
    streams=[data[offset:offset+sizes[0]],data[offset+sizes[0]:]]
    return dict(model=MODELS[model],quality=quality,height=h,width=w,shape=(zh,zw),streams=streams,header_bytes=offset)
