"""Write and package the CLIP conversion audit in the established output folder."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
outputs = root.parents[2] / 'outputs'
summary = json.loads((root / 'runs/stage35_clip_summary.json').read_text(encoding='utf-8'))
assert summary['saved_pairs'] == 50 and summary['complete_images'] == 10
first = json.loads((root / 'runs/stage35_clip_trial0_image0.json').read_text(encoding='utf-8'))
report = outputs / 'TokCom_第三十五阶段十图五轮CLIP转换口径诊断_2026-11-19.md'
text = '''# TokCom 第三十五阶段：十图五轮 CLIP 转换口径诊断

完成十张开发图、五轮 AWGN 8 dB 接收共 50 组作者 GenDecoding 重建的 CLIP 图像特征余弦相似度。模型、真实类别、采样种子 20261112 和接收 token 沿用前阶段。所有重跑的 token 正确数与前阶段相同，干净 VQGAN 参考 MSE 差异均小于 1e-8。每张图先对五轮求均值，再对十张图平均。

| CLIP ViT-B/32，越高越好 | 发送端真值比较遮蔽 | 接收端 CRC 失败遮蔽 |
|---|---:|---:|
| 作者直接将原始解码输出转 PIL | 0.711552 | 0.704012 |
| 先裁剪到 [0,1] 再转 PIL | 0.822604 | 0.815336 |
| 原图与干净 VQGAN 重建，先裁剪再转 PIL | 0.820243 | 0.820243 |

两种口径均使用 clip.load('ViT-B/32') 返回的官方预处理、同一图像编码器和余弦相似度。原图采用作者演示的 Resize((256,256))+ToTensor，源图 SHA-256 与缓存一致。作者 Performance_metrics.py 直接把浮点张量传给 ToPILImage；本机 torchvision 对浮点 RGB 先乘 255，再转 uint8，该处没有先裁剪。因此原始解码器超范围输出会影响转换后的图像。本阶段显式对照了直接转换与裁剪转换，数值不得混用。

这是十张开发图的五次重复信道接收，统计图像数仍为 10；生成种子固定，没有评估生成种子波动。真值比较使用发送端原 token，CRC 分支只依据接收校验，两者的信息条件不同。上述数值是转换流程诊断和开发集结果，尚不能视为论文总体性能。真值比较的裁剪 CLIP 均值略高于干净 VQGAN 参考，也不代表恢复了更多真实细节；感知特征相似度不能替代像素或 token 正确性。

机器汇总：runs/stage35_clip_summary.json，含十张图各自的五轮均值；全部单次数据在 runs/stage35_clip_trial*_image*.json。下一步将 token、PSNR、LPIPS、CLIP 按图像配对汇总，冻结统一输入协议后准备新的留出验证。

CLIP 权重 SHA-256：WEIGHT_HASH
'''
report.write_text(text.replace('WEIGHT_HASH', first['clip_weight_sha256']), encoding='utf-8')
names = ['prepare_stage35_clip.py', 'batch_stage35_clip.py', 'summarize_stage35_clip.py',
         'runs/stage35_clip_summary.json', 'runs/stage26_official_demo_tokens.npz',
         'runs/stage26_official_token_link_snr8_5trials.npz']
names += [f'runs/stage35_clip_trial{t}_image{i}.json' for t in range(5) for i in range(10)]
items = [(name, root) for name in names] + [(report.name, outputs)]
manifest = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name, base in items}
target = outputs / 'TokCom_stage35_clip_10images_5trials.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, base in items: archive.write(base / name, name)
    archive.writestr('SHA256_manifest.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert all(hashlib.sha256(archive.read(name)).hexdigest() == h for name, h in manifest.items())
print(report)
print(target, target.stat().st_size)
