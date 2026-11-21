"""Pin and check the public TokCom demo source inspected in stage20."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
official = root.parents[1] / 'tokencom_official_reference'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=official, text=True).strip()
assert commit == '0422ed32dcde3347e0bb6697f343f7a9ff7b3214'
notebook_path = official / 'TokCom_Demo.ipynb'
notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
code = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells']
                 if cell['cell_type'] == 'code')
checks = {
    'four_tokens_per_packet': 'WirelessSimulation(toknum=256, toks_per_pkg=4' in code,
    'fixed_interleaver': 'pkg_idx = token_idx % num_packages' in code,
    'crc11': 'CRCEncoder("CRC11")' in code,
    'short_convolutional_code': 'ConvEncoder(gen_poly=["101", "111"])' in code,
    'awgn': 'self.awgn_channel(tx_symbols, snr)' in code,
    'real_class_labels': 'labels=labels' in code,
    'generation_steps_32': 'self.step = 32' in code,
    'tx_rx_error_comparison': 'error_mask_token = (tx_ids != rx_ids).bool()' in code,
}
assert all(checks.values()), checks
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = {
    'source': 'https://github.com/liqiao19/TokenCom_Code',
    'commit': commit,
    'files_sha256': {name: sha(official / name) for name in
                     ['README.md', 'TokCom_Demo.ipynb', 'Performance_metrics.py']},
    'notebook_checks': checks,
    'scope': 'source audit only; official demo not executed with pretrained weights'}
out = root / 'runs/stage20_official_source_audit.json'
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
