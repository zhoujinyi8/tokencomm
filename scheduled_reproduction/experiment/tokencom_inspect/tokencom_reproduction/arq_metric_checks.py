"""Pure checks for ARQ endpoint metrics and identity-based clean-reference reuse."""
import math

import numpy as np

KEYS=('correct_tokens','whole_image_ter','psnr_raw','lpips_normalized','clip_clipped')
PATHS=('single_pass_padded','receiver_crc_arq_padded','sender_oracle_arq_padded')


def validate_rows(rows,ids,sent,clean_reference):
    assert [row['path'] for row in rows]==list(PATHS)
    sent=np.asarray(sent)
    for row in rows:
        final=np.asarray(ids[row['path']]);assert final.shape==sent.shape==(256,)
        correct=int((final==sent).sum());assert row['correct_tokens']==correct
        assert row['whole_image_ter']==1-correct/256
        assert all(math.isfinite(row[key]) for key in KEYS)
        exact=np.array_equal(final,sent)
        assert row['metric_source']==('exact_clean_id_reference' if exact else 'decoded_endpoint')
        if exact:
            for key,tolerance in [('psnr_raw',1e-4),('lpips_normalized',1e-5),('clip_clipped',1e-5)]:
                assert math.isclose(row[key],clean_reference[key],abs_tol=tolerance,rel_tol=0)

