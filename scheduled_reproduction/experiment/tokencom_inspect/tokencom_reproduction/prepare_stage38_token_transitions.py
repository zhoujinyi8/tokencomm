"""Capture the final VQGAN input codes without changing author generation."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'batch_stage33_author_psnr.py').read_text(encoding='utf-8')
source = source.replace('for trial in range(1, 5):', '''captured_codes = []
original_decode_code = vq.decode_code
def capture_decode_code(codes):
    captured_codes.append(codes.detach().clone())
    return original_decode_code(codes)
vq.decode_code = capture_decode_code
for trial in range(5):''')
source = source.replace('for image in range(1, 10):', 'for image in range(10):')
source = source.replace('stage33_author_psnr_trial', 'stage38_token_transitions_trial')
source = source.replace('                output, clean,', '                captured_codes.clear()\n                output, clean,')
source = source.replace('                result_rows.append(row)', '''                assert len(captured_codes) == 3
                final_codes = captured_codes[0].reshape_as(tx)
                assert torch.equal(captured_codes[1].reshape_as(tx), tx)
                before_correct = rx == tx
                after_correct = final_codes == tx
                assert int(after_correct.sum()) == row['correct_total_tokens']
                assert torch.equal(final_codes[~mask], rx[~mask])
                row['masked_received_correct'] = int((mask & before_correct).sum())
                row['masked_received_wrong'] = int((mask & ~before_correct).sum())
                row['wrong_to_correct'] = int((~before_correct & after_correct).sum())
                row['correct_to_wrong'] = int((before_correct & ~after_correct).sum())
                row['correct_to_correct_masked'] = int((mask & before_correct & after_correct).sum())
                row['wrong_to_wrong'] = int((~before_correct & ~after_correct).sum())
                row['direct_correct_tokens'] = int(before_correct.sum())
                row['final_token_ids'] = final_codes.reshape(-1).tolist()
                result_rows.append(row)''')
(root / 'batch_stage38_token_transitions.py').write_text(source, encoding='utf-8')
