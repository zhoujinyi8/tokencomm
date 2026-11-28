"""All nine endpoint paths in complete matched-framing development groups."""
def plot(comparison,out,prefix):
    assert comparison['complete'] and len(comparison['complete_snr_groups'])==6
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=comparison['complete_snr_groups'];snrs=sorted(groups,key=float)
    x=list(map(float,snrs));first=groups[snrs[0]]
    paths=[(category,path) for category in ('neural_means','arq_means') for path in first[category]]
    labels={'direct':'Direct decode','oracle_without_cmi':'Oracle mask, no CMI','oracle_with_cmi':'Oracle mask, CMI',
        'receiver_crc_without_cmi':'CRC mask, no CMI','receiver_crc_with_cmi':'CRC mask, CMI',
        'ideal_correct_tokens':'Ideal correct-token reference','single_pass_padded':'ARQ framing, single pass',
        'receiver_crc_arq_padded':'Receiver CRC ARQ','sender_oracle_arq_padded':'Sender-oracle ARQ'}
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    metrics=[('whole_image_ter','Final token error rate (%)',100),('psnr_raw','Raw-output PSNR (dB)',1),
             ('lpips_normalized','Normalized LPIPS',1),('clip_clipped','CLIP cosine',1)]
    for ax,(key,title,scale) in zip(axes.flat,metrics):
        for index,(category,path) in enumerate(paths):
            ax.plot(x,[scale*groups[s][category][path][key] for s in snrs],
                marker=('o','s','^','v','D','x','+','P','*')[index],linestyle='-' if index<5 else '--',
                color=plt.get_cmap('tab10')(index),linewidth=1.5,markersize=5,label=labels[path])
        ax.set_title(title);ax.set_xlabel('SNR (dB)');ax.grid(alpha=.25)
    fig.suptitle('Complete matched-padded development comparison: 10 images x 5 receptions per SNR\nAll six neural paths and three ARQ endpoints; same first-pass framing',fontsize=14)
    handles,legends=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,fontsize=10)
    fig.text(.5,.012,'Direct and single-pass endpoints overlap. Forward data only; feedback/header/CMI airtime unmodeled. Ideal tokens are a reference.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.19,hspace=.3,wspace=.25)
    files=[out/(prefix+'_完整对照.'+ext) for ext in ('svg','png')]
    for p in files:fig.savefig(p,dpi=160)
    plt.close(fig)
    return files
