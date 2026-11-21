"""Exercise the CI regression without downloading or loading model weights."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import torch

root=Path(__file__).resolve().parent
path=root.parents[1]/'tokencomm_github_sync/third_party/perceptual_similarity/lpips/trainer.py'
spec=importlib.util.spec_from_file_location('vendor_lpips_trainer',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
net=torch.nn.Identity()
with patch.object(module.lpips,'LPIPS',return_value=net),patch.object(module.lpips,'print_network') as printer:
    module.Trainer().initialize(use_gpu=False,printNet=True)
    printer.assert_called_once_with(net)
print('vendored LPIPS debug printer smoke passed; no model weights loaded')
