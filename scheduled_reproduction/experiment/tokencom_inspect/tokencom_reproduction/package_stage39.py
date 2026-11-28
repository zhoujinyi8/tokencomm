"""Save the completed five-trial transition attribution and evidence package."""
import hashlib
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'runs'
outputs=root.parents[2]/'outputs'
summary=json.loads((run/'stage39_transition_attribution.json').read_text(encoding='utf-8'))
assert summary['pairs']==50 and len(summary['complete_five_trial_images'])==10
report=outputs/'TokCom_第三十九阶段十图五轮修复误改归因_2026-11-20.md'
report.write_text('''# TokCom 第三十九阶段：十图五轮修复与误改归因

完成十张开发图、五轮 AWGN 8 dB 接收共 50 组作者 GenDecoding 最终 token 的逐位置记录。使用原有作者权重、真实类别、固定生成种子 20261112。未遮蔽位置逐组等于接收 token，重跑的正确 token 数和干净 VQGAN 参考 MSE 与第 35 阶段一致。

| 全部 50 组，共 12800 个位置 | 真值比较包级遮蔽 | CRC 失败包级遮蔽 |
|---|---:|---:|
| 接收错误 token | 265 | 265 |
| 被遮蔽的原本正确 token | 595 | 1543 |
| 错误被修复（错→对） | 30 | 34 |
| 正确被改错（对→错） | 509 | 1358 |
| 其中：在数据错误包内的误改 | 509 | 523 |
| 其中：在数据本身正确包内的误改 | 0 | 835 |
| 正确 token 数净变化 | −479 | −1324 |
| 直接接收正确数 | 12535 | 12535 |
| 生成后正确数 | 12056 | 11211 |

净变化满足“最终正确数 = 接收正确数 + 修复数 − 误改数”，分别还原前阶段每组 241.12/256 与 224.22/256 的正确均值。CRC 的 1358 次误改中，835 次发生在数据本身正确但校验失败的包内，占约 61.5%。CRC 遮蔽除包含真正数据错误包，还包含因校验位等错误判失败但 token 数据正确的包；这些额外遮蔽会使模型重写更多原本正确的位置。

两种恢复都存在数据错误包内原本正确 token 的误改。CRC 即使额外完全避免了 835 次误改，其余误改仍为 523，超过本次修复数 34。因此只移除额外 CRC 失败包并不足以据此保证精确 token 净收益；这只是对当前保存输出的计数解释，不能代替重新运行改变掩码后的生成实验。改变掩码会改变上下文和采样轨迹。

数据为十张开发图的五次重复接收，合计次数不是独立图像样本；生成种子固定，未评估生成随机性。真值比较需要发送端原 token，接收端不可直接获得；两者不是同等信息条件的算法对比。精确 token 下降也不能直接推断全部图像指标下降，前阶段已保存 PSNR、LPIPS、CLIP 的不同排序。

下一步以 CRC 接收端信息为约束，建立保留接收 token、按模型置信度选择替换的开发对照；保持直接解码、原作者恢复为基线，并记录替换数量和图像指标。在新留出验证前固定策略及输入协议。

机器归因：runs/stage39_transition_attribution.json；原始最终 token ID：runs/stage38_token_transitions_trial*_image*.json；完整转移计数：runs/stage38_token_transition_summary.json。
''',encoding='utf-8')
names=['prepare_stage38_token_transitions.py','batch_stage38_token_transitions.py',
       'summarize_stage38_token_transitions.py','summarize_stage39_transitions.py',
       'runs/stage39_transition_attribution.json','runs/stage38_token_transition_summary.json',
       'runs/stage26_official_token_link_snr8_5trials.npz']
names += [f'runs/stage38_token_transitions_trial{t}_image{i}.json' for t in range(5) for i in range(10)]
items=[(name,root) for name in names]+[(report.name,outputs)]
manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name,base in items}
target=outputs/'TokCom_stage39_10images_5trials_token_attribution.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,base in items:z.write(base/name,name)
    z.writestr('SHA256_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in manifest.items())
print(report)
print(target,target.stat().st_size)
