def plot(root,out,prefix,audit):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert audit['codec_cases_complete']
    rows=[json.loads((root/'runs/stage65_cheng_cases'/n).read_text(encoding='utf-8')) for n in audit['case_sha256']]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,model in zip(axes,('cheng2020-anchor','cheng2020-attn')):
        values=sorted([v for v in rows if v['model']==model],key=lambda v:v['image_index']);assert len(values)==10
        for key,label,marker in [('entropy_payload_bpp','Actual complete entropy streams','o'),('total_stored_bpp','Actual streams + CH65 header','s'),('estimated_likelihood_bpp','Likelihood estimate (proxy)','^')]:
            ax.plot([v['image_index']+1 for v in values],[v[key] for v in values],marker=marker,label=label)
        ax.axhline(.04,color='black',linestyle='--',label='Paper stated target ~0.04')
        ax.set_title(model+' / stock MSE quality 1');ax.set_xlabel('Frozen development image index');ax.set_ylabel('bits per pixel');ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.035),fontsize=9)
    fig.suptitle('Clean stock-Cheng coding diagnostic: actual byte rate versus likelihood estimate',fontsize=13)
    fig.text(.5,.012,'10 old development images; no channel. CH65 is a custom 34-byte diagnostic header. Stock checkpoints are not paper fine-tuned baselines.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.85,bottom=.25,wspace=.25)
    paths=[out/(prefix+'_实际码率.'+ext) for ext in ('svg','png')]
    for p in paths:fig.savefig(p,dpi=160)
    plt.close(fig);return paths
