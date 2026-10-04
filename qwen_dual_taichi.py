"""Large TaiChi variant: dual frozen vision features + BAN/KAN + semantic channel + Qwen2.5 LoRA."""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from models import BAN, SplineKAN, Channel


class QwenDualTaiChi(nn.Module):
    def __init__(self, llm_path, visual_dim=64, channel_dim=64, lora_rank=8, fusion_variant='ban_kan'):
        super().__init__()
        self.llm_path = str(llm_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(self.llm_path, local_files_only=True, dtype=torch.float32)
        base.requires_grad_(False)
        config = LoraConfig(r=lora_rank, lora_alpha=2*lora_rank, lora_dropout=0.0, bias='none',
                            task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])
        self.llm = get_peft_model(base, config)
        hidden = int(base.config.hidden_size)
        self.vit_adapter = nn.Sequential(nn.Conv2d(768, visual_dim, 1), nn.GroupNorm(1, visual_dim))
        self.conv_adapter = nn.Sequential(nn.Conv2d(768, visual_dim, 1), nn.GroupNorm(1, visual_dim))
        if fusion_variant not in ('ban_kan','ban_mlp','concat_mlp'):
            raise ValueError(f'Unknown fusion variant: {fusion_variant}')
        self.fusion_variant=fusion_variant
        if fusion_variant in ('ban_kan','ban_mlp'):
            self.ban = BAN(visual_dim)
        if fusion_variant=='ban_kan':
            self.kan = nn.Sequential(nn.LayerNorm(visual_dim), SplineKAN(visual_dim, visual_dim), SplineKAN(visual_dim, visual_dim))
        elif fusion_variant=='ban_mlp':
            self.kan = nn.Sequential(nn.LayerNorm(visual_dim),nn.Linear(visual_dim,2*visual_dim),nn.GELU(),nn.Linear(2*visual_dim,visual_dim))
        else:
            self.concat_fusion=nn.Sequential(nn.Conv2d(2*visual_dim,visual_dim,1),nn.GroupNorm(1,visual_dim),nn.GELU())
            self.kan=nn.Sequential(nn.LayerNorm(visual_dim),nn.Linear(visual_dim,2*visual_dim),nn.GELU(),nn.Linear(2*visual_dim,visual_dim))
        self.visual_to_llm = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden))
        self.channel_encoder = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, channel_dim))
        self.channel = Channel()
        self.channel_decoder = nn.Sequential(nn.Linear(channel_dim, hidden//2), nn.GELU(), nn.Linear(hidden//2, hidden))
        self.visual_tokens = 16
        self.model_config = {'visual_dim':visual_dim,'channel_dim':channel_dim,'lora_rank':lora_rank,'fusion_variant':fusion_variant,
                             'llm_hidden':hidden,'visual_tokens':self.visual_tokens}

    def visual_prefix(self, image, snr_db=10., kind='awgn'):
        vit, conv = image
        if vit.shape[1:] != (768,4,4) or conv.shape[1:] != (768,7,7):
            raise ValueError('Expected ViT [B,768,4,4] and ConvNeXt [B,768,7,7] cached features')
        v=self.vit_adapter(vit); c=self.conv_adapter(conv)
        if self.fusion_variant in ('ban_kan','ban_mlp'):
            x=self.ban(v,c).flatten(2).transpose(1,2)
        else:
            c=nn.functional.adaptive_avg_pool2d(c,(4,4)); x=self.concat_fusion(torch.cat([v,c],1)).flatten(2).transpose(1,2)
        x=self.kan(x)
        x = self.visual_to_llm(x)
        if kind != 'none':
            x = self.channel_decoder(self.channel(self.channel_encoder(x), snr_db, kind))
        return x

    def tokenize_supervised(self, questions, answers, max_text_tokens=96, device='cpu'):
        sequences=[]; answer_starts=[]
        for q,a in zip(questions,answers):
            prompt=f'Question: {q}\nAnswer:'
            p=self.tokenizer(prompt,add_special_tokens=True).input_ids
            full=self.tokenizer(prompt+' '+a+getattr(self.tokenizer,'eos_token',''),add_special_tokens=True).input_ids[:max_text_tokens]
            sequences.append(full); answer_starts.append(min(len(p),len(full)))
        width=max(map(len,sequences)); pad_id=self.tokenizer.pad_token_id
        ids=torch.full((len(sequences),width),pad_id,dtype=torch.long,device=device)
        attention=torch.zeros_like(ids)
        labels=torch.full((len(sequences),self.visual_tokens+width),-100,dtype=torch.long,device=device)
        for i,(seq,start) in enumerate(zip(sequences,answer_starts)):
            ids[i,:len(seq)]=torch.tensor(seq,device=device); attention[i,:len(seq)]=1
            labels[i,self.visual_tokens+start:self.visual_tokens+len(seq)]=torch.tensor(seq[start:],device=device)
        return ids,attention,labels

    def forward(self, image, questions, answers, snr_db=10., kind='awgn', max_text_tokens=96):
        prefix=self.visual_prefix(image,snr_db,kind)
        ids,text_mask,labels=self.tokenize_supervised(questions,answers,max_text_tokens,prefix.device)
        text_emb=self.llm.get_input_embeddings()(ids)
        inputs=torch.cat([prefix,text_emb],1)
        attention=torch.cat([torch.ones(prefix.shape[:2],dtype=text_mask.dtype,device=prefix.device),text_mask],1)
        return self.llm(inputs_embeds=inputs,attention_mask=attention,labels=labels,use_cache=False)

    @torch.inference_mode()
    def generate_answers(self, image, questions, snr_db=10., kind='awgn', max_new_tokens=24):
        self.eval(); prefix=self.visual_prefix(image,snr_db,kind); b=prefix.shape[0]
        prompts=[f'Question: {q}\nAnswer:' for q in questions]
        encoded=self.tokenizer(prompts,padding=True,return_tensors='pt',add_special_tokens=True)
        ids=encoded.input_ids.to(prefix.device); mask=encoded.attention_mask.to(prefix.device)
        emb=torch.cat([prefix,self.llm.get_input_embeddings()(ids)],1)
        attention=torch.cat([torch.ones(prefix.shape[:2],dtype=mask.dtype,device=prefix.device),mask],1)
        generated=[]; done=torch.zeros(b,dtype=torch.bool,device=prefix.device)
        # Prefill the visual/text prompt once, then decode one token at a time
        # from Qwen's KV cache. Recomputing the full sequence at every step is
        # needlessly expensive during validation, especially on CPU.
        out=self.llm(inputs_embeds=emb,attention_mask=attention,use_cache=True)
        last=(attention.sum(1)-1).long()
        logits=out.logits[torch.arange(b,device=prefix.device),last]
        past=out.past_key_values
        for _ in range(max_new_tokens):
            nxt=logits.argmax(-1)
            nxt=torch.where(done,torch.full_like(nxt,self.tokenizer.eos_token_id),nxt)
            generated.append(nxt)
            done |= nxt.eq(self.tokenizer.eos_token_id)
            if done.all(): break
            attention=torch.cat([attention,torch.ones(b,1,dtype=attention.dtype,device=attention.device)],1)
            out=self.llm(input_ids=nxt[:,None],attention_mask=attention,past_key_values=past,use_cache=True)
            logits=out.logits[:,-1]; past=out.past_key_values
        tokens=torch.stack(generated,1) if generated else torch.empty(b,0,dtype=torch.long)
        return [self.tokenizer.decode(row,skip_special_tokens=True).strip() for row in tokens]

    def trainable_parameter_summary(self):
        total=sum(p.numel() for p in self.parameters()); trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_parameters':total,'trainable_parameters':trainable,'trainable_fraction':trainable/total}
