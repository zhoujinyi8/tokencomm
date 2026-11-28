"""Standalone, exportable training curves from recorded run logs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,name,title,val_key in zip(axes,['maskgit_imagenette','taichi_imagenette'],
                                   ['MaskGIT | real image tokens','Tiny TaiChi | class-name responses'],['ce','response_ce']):
        root=Path('runs')/name
        rows=[json.loads(x) for x in (root/'history.jsonl').read_text().splitlines()]
        x=np.array([r['step'] for r in rows]); y=np.array([r['train_loss'] for r in rows])
        ax.plot(x,y,color='#9bb5c8',alpha=.35,lw=.6,label='Training batch CE')
        window=min(20,len(y)); avg=np.convolve(y,np.ones(window)/window,mode='valid')
        ax.plot(x[window-1:],avg,color='#216184',lw=1.8,label=f'Training CE ({window}-step mean)')
        points=[r for r in rows if 'validation' in r]
        ax.plot([r['step'] for r in points],[r['validation'][val_key] for r in points],
                color='#db7038',marker='o',ms=4,lw=1.7,label='Fixed validation CE')
        best=json.loads((root/'status.json').read_text())['best_step']
        ax.axvline(best,color='#555',ls='--',lw=1,label=f'Selected step: {best}')
        ax.set(title=title,xlabel='Optimizer step',ylabel='Cross-entropy (nats)')
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False); ax.legend(fontsize=8)
    fig.suptitle('Imagenette pilot: 240 train / 40 validation / 40 held-out test',fontsize=12)
    out=Path('runs/real_evaluation'); out.mkdir(exist_ok=True)
    fig.savefig(out/'training_curves.png',dpi=170)
    fig.savefig(out/'training_curves.svg')
    plt.close(fig)


if __name__=='__main__': main()
