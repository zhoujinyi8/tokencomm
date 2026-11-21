"""Derive the LPIPS range audit from the fixed author PSNR runner."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'batch_stage33_author_psnr.py').read_text(encoding='utf-8')
source = source.replace('import torch\n', 'import torch\nimport lpips\nimport importlib.metadata\n')
source = source.replace('for trial in range(1, 5):', "lpips_model = lpips.LPIPS(net='alex').eval()\nfor trial in range(5):")
source = source.replace('for image in range(1, 10):', 'for image in range(10):')
source = source.replace('stage33_author_psnr_trial', 'stage34_lpips_trial')
source = source.replace('                result_rows.append(row)', '''                row['lpips_author_clipped_no_normalize'] = float(lpips_model(images[image].clamp(0, 1), output.clamp(0, 1)).mean())
                row['lpips_clipped_normalize_true'] = float(lpips_model(images[image].clamp(0, 1), output.clamp(0, 1), normalize=True).mean())
                row['lpips_clean_vqgan_normalize_true'] = float(lpips_model(images[image].clamp(0, 1), clean.clamp(0, 1), normalize=True).mean())
                result_rows.append(row)''')
source = source.replace("'snr_db': 8.,", "'lpips_version': importlib.metadata.version('lpips'), 'lpips_net': 'alex', 'snr_db': 8.,")
(root / 'batch_stage34_lpips.py').write_text(source, encoding='utf-8')
