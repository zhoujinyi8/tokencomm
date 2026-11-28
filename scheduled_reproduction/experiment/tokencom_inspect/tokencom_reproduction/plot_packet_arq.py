"""Plot matched padded-packet forward costs and VQGAN endpoint quality."""


def plot_packet_arq(physical,metrics,outputs,prefix):
    assert physical['complete'] and metrics['complete']
    snrs=[6.,6.5,7.,7.5,8.,9.]
    assert len(physical['complete_snr_groups'])==len(metrics['complete_snr_groups'])==6
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=['single_pass_padded','receiver_crc_arq_padded','sender_oracle_arq_padded']
    labels=['Single pass','Receiver CRC ARQ','Sender-oracle ARQ']
    fields=[('physical','payload_tce_ratio_of_sums','Pixels / attempted payload bit',1),
            ('physical','forward_pixels_per_symbol_ratio_of_sums','Pixels / forward QAM symbol',1),
            ('metric','whole_image_ter','Final token error rate (%)',100),
            ('metric','psnr_raw','Raw-output PSNR (dB)',1),
            ('metric','lpips_normalized','Normalized LPIPS',1),
            ('metric','clip_clipped','CLIP cosine',1)]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for axis,(kind,key,ylabel,scale) in zip(axes.flat,fields):
        for policy,label,marker in zip(policies,labels,['o','s','^']):
            values=[scale*(physical['complete_snr_groups'][str(s)][policy][key] if kind=='physical' else
                           metrics['complete_snr_groups'][str(s)]['means_over_images'][policy][key]) for s in snrs]
            axis.plot(snrs,values,marker=marker,label=label)
        axis.set(xlabel='SNR (dB)',ylabel=ylabel,xticks=snrs);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched padded-packet protocol: 10 old images x 5 receptions per SNR\n'
                 'All three VQGAN endpoint policies; actual retries; cap/censoring retained; no MaskGIT path',fontsize=11)
    fig.supxlabel('Forward data only: 104 bits / 26 QAM symbols per attempt. Feedback, headers and CMI airtime remain unmodeled.',fontsize=9)
    paths=[]
    for suffix in ('svg','png'):
        path=outputs/f'{prefix}_填充包ARQ.{suffix}';fig.savefig(path,dpi=160);paths.append(path)
    plt.close(fig)
    return paths
