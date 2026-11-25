"""Restore only the frozen holdout identities from the pinned source parquet."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from pretrained_vq import sha256
from prepare_stage5_imagenette import pixel_digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--parquet',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
folder=root/'data/imagenette_stage44_holdout'
protocol_path=root/'runs/stage44_frozen_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
assert sha256(protocol_path)==manifest['protocol_sha256']
assert sha256(folder/'test.jsonl')==manifest['test_jsonl_sha256']
assert sha256(args.parquet)==protocol['source_sha256']
records=[json.loads(s) for s in (folder/'test.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records)==100
needed={r['sha256']:r for r in records}
assert len(needed)==100
restored=set()
for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=64):
    for row in batch.to_pylist():
        raw=row['image']['bytes'];digest=hashlib.sha256(raw).hexdigest()
        if digest not in needed:continue
        record=needed[digest]
        assert pixel_digest(raw)==record['pixel_sha256']
        out=(folder/record['image']).resolve()
        assert folder.resolve() in out.parents
        if out.exists():assert sha256(out)==digest
        else:
            out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
        restored.add(digest)
assert len(restored)==100
print('Restored exactly 100 frozen image identities; no sampling or model evaluation')
