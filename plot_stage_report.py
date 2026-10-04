"""Render report figures from saved results only; no training or new inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / '阶段报告图片'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':11})
m = json.loads((ROOT/'runs/real_evaluation/metrics.json').read_text(encoding='utf-8'))
def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(14,5.2),layout='constrained')
methods=['spatial','frequency','maskgit_cmi','maskgit_no_cmi']
names=['空间邻近填充','最高频 token','MaskGIT＋CMI','MaskGIT 无 CMI']
for method,name in zip(methods,names):
    rows=[r for r in m['token_results'] if r['method']==method]
    axs[0].plot([r['per']*100 for r in rows],[r['masked_token_accuracy']*100 for r in rows],marker='o',label=name)
axs[0].set(xlabel='设定整包丢失概率（%）',ylabel='丢失位置精确恢复准确率（%）',title='A. 40 张测试图像：相同擦除掩码',ylim=(0,2.7),xticks=[10,30,50])
axs[0].legend(fontsize=9); axs[0].grid(alpha=.2)
ps=m['image_evaluation']['mean_psnr_db']
v=list(ps.values()); bars=axs[1].bar(range(4),v,color=['#607d8b','#e6a23c','#2878b5','#45a778'])
axs[1].set(xticks=range(4),xticklabels=['无信道损伤\nVQGAN','空间填充','MaskGIT\n＋CMI','MaskGIT\n无 CMI'],ylabel='平均 PSNR（dB）',title='B. 每类首张测试图，共 10 张；PER=30%',ylim=(0,20))
axs[1].bar_label(bars,fmt='%.3f',padding=4); axs[1].grid(axis='y',alpha=.2)
fig.suptitle('图3  恢复准确率与图像质量：样本范围不同，不作指标替代',fontsize=15)
save(fig,'03_恢复指标比较')

classes=[x.lower() for x in m['class_names']]
examples=m['taichi']['results'][0]['examples']
extra=sorted(set(x['predicted'] for x in examples)-set(classes)); pred_names=classes+extra
cm=np.zeros((10,len(pred_names)),dtype=int)
for e in examples: cm[classes.index(e['target']),pred_names.index(e['predicted'])]+=1
fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=4)
ax.set(xticks=range(len(pred_names)),xticklabels=pred_names,yticks=range(10),yticklabels=classes,
       xlabel='模型自由生成的类别',ylabel='真实类别',title='图4  小型 TaiChi 混淆矩阵：AWGN 10 dB，5/40 正确')
plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
for i in range(10):
    for j in range(len(pred_names)):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>=3 else '#243746')
fig.colorbar(im,ax=ax,label='样本数',shrink=.7,ticks=range(5))
save(fig,'04_TaiChi混淆矩阵')

for group in range(2):
    fig,axs=plt.subplots(5,4,figsize=(12,16),layout='constrained')
    for row,c in enumerate(range(group*5,group*5+5)):
        strip=np.asarray(Image.open(ROOT/f'runs/real_evaluation/comparison_class_{c}.png'))
        width=strip.shape[1]//4
        metrics=m['image_evaluation']['per_image'][c]['psnr_db']
        scores=[None,metrics['clean_vqgan'],metrics['spatial'],metrics['maskgit_cmi']]
        for col in range(4):
            ax=axs[row,col]; ax.imshow(strip[:,col*width:(col+1)*width]); ax.set_xticks([]); ax.set_yticks([])
            if row==0: ax.set_title(['原图','无信道损伤 VQGAN','空间填充','MaskGIT＋CMI'][col],fontsize=12)
            ax.set_xlabel('原始参考图' if col==0 else f'PSNR = {scores[col]:.2f} dB',fontsize=10)
            if col==0: ax.set_ylabel(m['class_names'][c],fontsize=10)
    fig.suptitle(f'图{5+group}  各类首张测试图（{group*5+1}–{group*5+5}）：PER=30%，同一擦除条件',fontsize=15)
    save(fig,f'0{5+group}_重建对比')

d=ROOT/'runs/real_digital_inference'
z=np.load(d/'tokens.npz'); missing=z['missing'][0].reshape(16,16)
equal=(z['filled'][0]==z['transmitted'][0]).reshape(16,16)
state=np.zeros((16,16)); state[missing & equal]=1; state[missing & ~equal]=2
from matplotlib.colors import ListedColormap, BoundaryNorm
fig,axs=plt.subplots(2,3,figsize=(13,8.5),layout='constrained')
for ax,file,title in zip(axs[0],['original.png','clean_reconstruction.png','channel_reconstruction.png'],
                         ['原图','无信道损伤：15.624 dB','数字链路＋MaskGIT：11.192 dB']):
    ax.imshow(Image.open(d/file)); ax.set_title(title); ax.axis('off')
axs[1,0].imshow(missing,cmap=ListedColormap(['#dae4ed','#e59336']),vmin=0,vmax=1,interpolation='nearest')
axs[1,0].set_title(f'接收擦除图：{int(missing.sum())}/256 个 token')
cmap=ListedColormap(['#dae4ed','#35a779','#d85656'])
axs[1,1].imshow(state,cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest')
axs[1,1].set_title(f'恢复诊断：擦除位置正确 {int((missing & equal).sum())}/{int(missing.sum())}')
for ax in axs[1,:2]: ax.set(xlabel='token 列',ylabel='token 行',xticks=[0,5,10,15],yticks=[0,5,10,15])
axs[1,2].axis('off')
axs[1,2].text(0,.95,'信道：Rayleigh，Es/N0 = 8 dB\nCRC 失败：27 / 64 包\n实测 PER：42.1875%\n\n擦除图：橙色 = 丢失\n诊断图：绿色 = 恢复正确\n              红色 = 恢复错误\n灰色 = 未擦除位置\n\n图中为反交织后的空间位置。\n真值仅用于事后诊断，\n不作为接收恢复的输入。\n单样本结果，不代表平均性能。',va='top',fontsize=12,linespacing=1.6)
fig.suptitle('图7  真实数字链路的图像结果与 token 位置诊断',fontsize=16)
save(fig,'07_数字链路诊断')
print(json.dumps({'output':str(OUT),'awgn_prediction_counts':{n:sum(e['predicted']==n for e in examples) for n in pred_names},
                  'digital_missing':int(missing.sum()),'digital_correct_missing':int((missing & equal).sum())},ensure_ascii=False))
