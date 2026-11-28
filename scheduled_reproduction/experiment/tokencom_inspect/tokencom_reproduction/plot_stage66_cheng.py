"""Standalone scientific figures for all fixed-grid arms and complete held-out rates."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def plot(root,out,prefix):
    read=lambda n:json.loads((root/'runs'/n).read_text(encoding='utf-8'))
    session=read('stage66_session_audit.json');stock=read('stage65_cheng_audit.json')['complete_model_groups']['cheng2020-anchor']['means']
    resource=read('stage66_resource_budget.json');plan=read('stage66_holdout_plan.json');selected=session['selected_arm_id']
    rows=session['grid_selection']['rows'];fig,axes=plt.subplots(1,3,figsize=(14,4.7));colors=['#2274a5','#f49d37','#7b4ab0']
    for ax,key,label in zip(axes,('psnr_decoded','lpips_normalized','clip_clipped'),('PSNR (dB), higher is better','LPIPS, lower is better','CLIP cosine, higher is better')):
        for index,row in enumerate(rows):
            name=row['arm_id'];value=session['grid_metric_means'][name][key];is_chosen=name==selected
            ax.scatter(row['mean_entropy_payload_bpp'],value,s=105 if is_chosen else 60,c=colors[index],marker='*' if is_chosen else 'o',zorder=3,label=f"lambda={int(name.split('_')[1])/100000:.5f}")
        ax.scatter(stock['entropy_payload_bpp'],stock[key],s=65,c='#333333',marker='x',label='stock Anchor q1')
        ax.axvspan(.036,.044,color='#b6e2c4',alpha=.4,label='frozen rate window')
        ax.axvline(.04,color='#4d4d4d',linestyle=':',linewidth=1)
        ax.set_xlabel('Actual entropy payload (bits/pixel)');ax.set_ylabel(label);ax.grid(alpha=.22)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9,bbox_to_anchor=(.5,.01))
    fig.suptitle('Clean development calibration: all 3 fixed-budget arms (10 images each)',fontsize=14)
    fig.tight_layout(rect=(0,.13,1,.93));paths=[]
    for ext in ('svg','png'):
        path=out/(prefix+'_开发码率与质量.'+ext);fig.savefig(path,dpi=155,bbox_inches='tight');paths.append(path)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));x=np.arange(2);width=.34
    cohorts=[resource['groups'][selected],resource['groups']['frozen_holdout']]
    for shift,mode,label,color in [(-width/2,'entropy_only_metadata_out_of_band','Entropy only; metadata out of band','#2274a5'),(width/2,'with_custom_header','Complete custom CH65 container','#f49d37')]:
        values=[v['means'][mode]['forward_qam_symbols'] for v in cohorts]
        bars=axes[0].bar(x+shift,values,width,color=color,label=label)
        axes[0].bar_label(bars,fmt='%.2f',padding=10,fontsize=9)
    axes[0].axhline(1664,color='#333333',linestyle='--',label='TokCom padded framing reference: 1664')
    axes[0].set_xticks(x,['Development (10)','Frozen holdout (100)']);axes[0].set_ylabel('Mean first-pass forward QAM symbols');axes[0].set_ylim(0,2600);axes[0].legend(fontsize=8,loc='upper left');axes[0].grid(axis='y',alpha=.2)
    rates=[v['entropy_payload_bpp'] for v in session['holdout_per_image']]
    axes[1].hist(rates,bins=15,color='#2274a5',edgecolor='white');axes[1].axvspan(.036,.044,color='#b6e2c4',alpha=.45)
    axes[1].axvline(np.mean(rates),color='#d24b40',label=f'Mean {np.mean(rates):.5f}')
    axes[1].set_xlabel('Actual held-out entropy bits/pixel');axes[1].set_ylabel('Image count');axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.suptitle('Frozen checkpoint: measured transport costs and held-out rate variation',fontsize=14)
    fig.text(.5,.01,'Forward data only. Custom header is an engineering choice; feedback, packet control and checkpoint metadata are excluded.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    for ext in ('svg','png'):
        path=out/(prefix+'_留出码率与首发开销.'+ext);fig.savefig(path,dpi=155,bbox_inches='tight');paths.append(path)
    plt.close(fig)
    transport=read('stage66_transport_audit.json')
    if transport['complete']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.6));groups=transport['complete_snr_groups'];snrs=[float(v) for v in groups]
        for policy,label,color in [('single_pass_padded','Single pass','#2274a5'),('receiver_crc_arq_padded','Receiver CRC ARQ','#d97822'),('sender_oracle_arq_padded','Sender-truth oracle ARQ','#6d4b99')]:
            axes[0].plot(snrs,[groups[str(v)][policy]['mean_forward_qam_symbols'] for v in snrs],marker='o',label=label,color=color)
            axes[1].plot(snrs,[100*groups[str(v)][policy]['complete_and_exact_frames']/50 for v in snrs],marker='o',label=label,color=color)
        axes[0].set_ylabel('Mean total forward QAM symbols');axes[1].set_ylabel('Byte-exact uncensored frames (%)');axes[1].set_ylim(-4,106)
        for ax in axes:ax.set_xlabel('SNR (dB)');ax.grid(alpha=.22);ax.legend(fontsize=9)
        fig.suptitle('Measured Cheng-container transport: 50 frames per SNR, 10 development images',fontsize=13)
        fig.text(.5,.01,'CRC wrong acceptances remain in outcomes. Feedback/control airtime excluded; this is not a full paper image-quality benchmark.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.93))
        for ext in ('svg','png'):
            path=out/(prefix+'_实际字节传输.'+ext);fig.savefig(path,dpi=155,bbox_inches='tight');paths.append(path)
        plt.close(fig)
    indices=plan['fresh_decoder_image_indices'];fig,axes=plt.subplots(2,5,figsize=(13,5.8))
    for col,index in enumerate(indices):
        image=plan['images'][index]
        with Image.open(root/image['path']) as source:original=source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        with np.load(root/'runs/stage66_holdout_cases'/f'image{index:03d}.npz') as a:decoded=a['pixels'][0].transpose(1,2,0)
        row=next(v for v in session['holdout_per_image'] if v['image_index']==index)
        axes[0,col].imshow(original);axes[1,col].imshow(decoded)
        axes[0,col].set_title(f'Fixed index {index}',fontsize=10)
        axes[1,col].set_title(f'{row["entropy_payload_bpp"]:.4f} bpp; {row["psnr_decoded"]:.2f} dB',fontsize=9)
        for ax in axes[:,col]:ax.set_xticks([]);ax.set_yticks([])
    axes[0,0].set_ylabel('Original');axes[1,0].set_ylabel('Frozen Cheng')
    fig.suptitle('Preregistered display indices; existing ten-class holdout, clean channel',fontsize=13);fig.tight_layout(rect=(0,0,1,.94))
    path=out/(prefix+'_定点重建.png');fig.savefig(path,dpi=145,bbox_inches='tight');paths.append(path);plt.close(fig)
    return paths
