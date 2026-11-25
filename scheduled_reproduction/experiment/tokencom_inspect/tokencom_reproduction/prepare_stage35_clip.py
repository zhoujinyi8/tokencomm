"""Derive the CLIP input-conversion audit from the fixed author runner."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'batch_stage33_author_psnr.py').read_text(encoding='utf-8')
source = source.replace('import torch\n', 'import torch\nimport clip\n')
setup = '''clip_model, clip_preprocess = clip.load('ViT-B/32', device='cpu')
clip_model.eval()
clip_weight_hash = sha256(Path.home() / '.cache/clip/ViT-B-32.pt')
def clip_similarity(original, reconstructed, clamp):
    if clamp:
        reconstructed = reconstructed.clamp(0, 1)
    pil = transforms.ToPILImage()
    pair = torch.stack([clip_preprocess(pil(original[0])),
                        clip_preprocess(pil(reconstructed[0]))])
    features = clip_model.encode_image(pair)
    return float(torch.nn.functional.cosine_similarity(features[:1], features[1:]).item())
for trial in range(5):'''
source = source.replace('for trial in range(1, 5):', setup)
source = source.replace('for image in range(1, 10):', 'for image in range(10):')
source = source.replace('stage33_author_psnr_trial', 'stage35_clip_trial')
source = source.replace('                result_rows.append(row)', '''                row['clip_author_raw_to_pil'] = clip_similarity(images[image], output, False)
                row['clip_clipped_to_pil'] = clip_similarity(images[image], output, True)
                row['clip_clean_vqgan_clipped_to_pil'] = clip_similarity(images[image], clean, True)
                result_rows.append(row)''')
source = source.replace("'snr_db': 8.,", "'clip_model': 'ViT-B/32', 'clip_weight_sha256': clip_weight_hash, 'snr_db': 8.,")
(root / 'batch_stage35_clip.py').write_text(source, encoding='utf-8')
