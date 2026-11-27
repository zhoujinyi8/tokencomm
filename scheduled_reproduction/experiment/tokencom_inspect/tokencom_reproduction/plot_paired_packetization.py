"""Plot all four fixed/random paths only when all six paired SNR groups are complete."""
import math


def plot_paired_packetization(audit,outputs,prefix):
    assert audit['complete_randomized_study'] and audit['paired_quality_snr_groups']==6
    groups=audit['groups'];snrs=sorted(map(float,groups))
    assert snrs==[6.,6.5,7.,7.5,8.,9.]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=['direct','without_cmi','with_cmi','ideal_correct_tokens']
    labels=['Direct decode','Without CMI','With CMI','Ideal correct tokens']
    fields=[('whole_image_ter','Residual TER difference (percentage points)',100),
            ('psnr_raw','PSNR difference (dB)',1),('lpips_normalized','LPIPS difference',1),
            ('clip_clipped','CLIP cosine difference',1)]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,(key,ylabel,scale) in zip(axes.flat,fields):
        for path,label,marker in zip(paths,labels,['o','s','^','D']):
            values=[]
            for snr in snrs:
                group=groups[str(snr)]
                assert group['receptions_per_stream']==50 and group['images']==10
                delta=group['paired_mean_difference'][path][key]
                assert math.isclose(delta,sum(r[path][key] for r in group['paired_image_differences'])/10,abs_tol=1e-12)
                assert math.isfinite(delta);values.append(scale*delta)
            axis.plot(snrs,values,marker=marker,label=label)
        axis.axhline(0,color='gray',linestyle='--',linewidth=1)
        axis.set(xlabel='SNR (dB)',ylabel=ylabel,xticks=snrs);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Complete old-development paired comparison: randomized minus fixed\n'
                 '10 images x 5 receptions per SNR; all four paths; no pipeline selected',fontsize=12)
    fig.supxlabel('Average five trials within image, then ten images. Favorable directions: lower TER/LPIPS, higher PSNR/CLIP.',fontsize=9)
    outputs.mkdir(parents=True,exist_ok=True);figures=[]
    for suffix in ('svg','png'):
        path=outputs/f'{prefix}_模型配对差值.{suffix}'
        fig.savefig(path,dpi=160);figures.append(path)
    plt.close(fig)
    return figures
