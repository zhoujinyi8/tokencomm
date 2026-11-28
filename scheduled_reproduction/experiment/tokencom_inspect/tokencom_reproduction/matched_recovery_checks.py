import math
import numpy as np

PATHS=('direct','oracle_without_cmi','oracle_with_cmi','receiver_crc_without_cmi','receiver_crc_with_cmi','ideal_correct_tokens')
def validate(rows,codes,rx,tx,oracle,crc):
    assert [r['path'] for r in rows]==list(PATHS)
    assert set(codes)==set(PATHS)
    assert np.array_equal(codes['direct'],rx) and np.array_equal(codes['ideal_correct_tokens'],tx)
    assert not np.any((rx!=tx)&~oracle)
    for row in rows:
        ids=np.asarray(codes[row['path']]);assert ids.shape==(256,) and np.issubdtype(ids.dtype,np.integer)
        assert ((ids>=0)&(ids<1024)).all()
        assert row['correct_tokens']==int((ids==tx).sum())
        assert row['whole_image_ter']==1-row['correct_tokens']/256
        assert 0<=row['steps']<=32
        assert all(math.isfinite(row[k]) for k in ('psnr_raw','lpips_normalized','clip_clipped'))
        if row['path'] not in ('direct','ideal_correct_tokens'):
            mask=crc if row['path'].startswith('receiver_crc') else oracle
            assert np.array_equal(ids[~mask],rx[~mask])
