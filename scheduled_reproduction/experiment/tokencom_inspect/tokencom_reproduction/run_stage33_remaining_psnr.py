"""Complete original-image PSNR for the remaining 36 saved PHY receptions."""
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
for trial in range(1, 5):
    for image in range(1, 10):
        record = cache['records'][selected['selected_indices'][image]]
        assert record['sha256'] == selected['rows'][image]['image_sha256']
        out = root / f'runs/stage33_author_psnr_trial{trial}_image{image}.json'
        if out.exists():
            continue
        command = [sys.executable, str(root / 'smoke_stage27_physical_reconstruction.py'),
                   '--official-source', str(root.parent.parent / 'tokencom_official_reference'),
                   '--trial', str(trial), '--image-index', str(image),
                   '--original-image', str(data / record['image']), '--out', str(out)]
        subprocess.run(command, cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'DONE trial={trial} image={image}', flush=True)
