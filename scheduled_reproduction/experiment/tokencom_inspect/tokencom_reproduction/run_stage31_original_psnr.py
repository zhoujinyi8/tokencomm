"""Run author-style original-image PSNR diagnostics on the first PHY trial."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent
data = root / 'data/imagenette_stage5'
cache = torch.load(data / 'token_cache/val.pt', map_location='cpu', weights_only=True)
selected = json.loads(np.load(root / 'runs/stage26_official_demo_tokens.npz')['metadata'].item())
for image in range(10):
    record = cache['records'][selected['selected_indices'][image]]
    assert record['sha256'] == selected['rows'][image]['image_sha256']
    out = root / f'runs/stage31_author_metric_smoke_trial0_image{image}.json'
    if out.exists():
        continue
    command = [sys.executable, str(root / 'smoke_stage27_physical_reconstruction.py'),
               '--official-source', str(root.parent.parent / 'tokencom_official_reference'),
               '--trial', '0', '--image-index', str(image),
               '--original-image', str(data / record['image']), '--out', str(out)]
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
    print(f'DONE image={image}', flush=True)
