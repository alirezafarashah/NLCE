import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from template import template_dict
from utils import *
from einops import rearrange

def trans_target(max_idx, original_target):
    max_idx_ = list(set(max_idx))
    transform_target = torch.zeros_like(original_target, dtype=original_target.dtype, device=original_target.device)
    remain_idx = [idx for idx in range(len(original_target)) if idx not in max_idx_]
    all_idx = max_idx_ + remain_idx
    for i, idx in enumerate(all_idx):
        transform_target[i, :, :] = original_target[idx, :, :]
    return transform_target

class VisualAttentionProcess(nn.Module):

    def __init__(self, module_name=None, atten_type='original', target_records=None, record=False, 
    record_type=None, sigmoid_setting=None, token_sim=False, decomp_timestep=0,  **kwargs):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records  = target_records
        self.record = record
        self.record_type = record_type
        self.sigmoid_setting = sigmoid_setting
        self.token_sim = token_sim
        self.decomp_timestep = decomp_timestep

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(self.module_name, self.atten_type, self.target_records, self.record, self.record_type, self.sigmoid_setting, self.token_sim, self.decomp_timestep)
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)
    

class AttnProcessor():

    def __init__(self, module_name=None, atten_type='original', target_records=None, record=False, record_type=None, sigmoid_setting=None, token_sim=False, decomp_timestep=0) -> None:
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = copy.copy(target_records)
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.token_sim=token_sim
        self.decomp_timestep=decomp_timestep

    def sigmoid(self, x, setting): 
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_ortho_decomp(self, target_value, pro_record, ortho_basis=None, project_matrix=None): 

        if ortho_basis is None and project_matrix is None:
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1) # [77, 640]
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1) # [77, 640]
            dot1 = (tar_record_ * pro_record_).sum(-1)
            dot2 = (tar_record_ * tar_record_).sum(-1)
            if self.token_sim:
                cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
                if self.sigmoid_setting is not None:
                    cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            else:
                cos_sim = torch.ones_like(dot1)
            weight = torch.nan_to_num(cos_sim * (dot1 / dot2), nan=0.0)
            weight[0].fill_(0)
            era_record = weight.unsqueeze(0).unsqueeze(-1) * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
        else:
            device = pro_record.device
            tar_record_ = rearrange(target_value, 'b h l d -> l b (h d)') # [77, num_concepts, 640]
            pro_record_ = rearrange(pro_record, 'h l d -> l (h d)').unsqueeze(1) # [77, 1, 640]
            dot1 = (ortho_basis * pro_record_).sum(-1)
            dot2 = (ortho_basis * ortho_basis).sum(-1)
            weight = torch.nan_to_num((dot1 / dot2).unsqueeze(1), nan=0.0)
            weight[0].fill_(0)
            if self.token_sim:
                cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1) # [77, 2]
                if self.sigmoid_setting is not None:
                    cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            else:
                cos_sim = torch.ones_like(tar_record_[:, :, 0])
            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            era_record = torch.bmm(weight, projected_basis).view((77, 16, -1)).permute(1, 0, 2)

        return era_record.to(device)

    def record_ortho_decomp(self, target_record, current_record):
        current_name = next(k for k in target_record if k.endswith(self.module_name))
        current_timestep, current_block = current_name.split('.', 1)
        (target_value, project_matrix, ortho_basis) = target_record.pop(current_name)

        if int(current_timestep) <=  self.decomp_timestep:
            return current_record, current_record

        target_value = target_value.view((2, int(len(target_value)//16), -1)+ target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        current_record = current_record.view((2, int(len(current_record)//16), -1)+ target_value.size()[-2:])
        current_record = current_record.permute(1, 0, 2, 3, 4).contiguous().view((current_record.size()[1], -1) + target_value.size()[-2:])
        erase_record, retain_record = [], []

        for pro_record in current_record:
            era_record = self.cal_ortho_decomp(target_value, pro_record, ortho_basis, project_matrix)
            ret_record = pro_record - era_record
            erase_record.append(era_record.view((2, -1) + era_record.size()[-2:]))
            retain_record.append(ret_record.view((2, -1) + ret_record.size()[-2:]))
        retain_record = rearrange(torch.stack(retain_record, dim=0), 'b n c l d -> (n b c) l d')
        erase_record =  rearrange(torch.stack(erase_record, dim=0), 'b n c l d -> (n b c) l d')
        return (erase_record, retain_record)

    def cal_gram_schmidt(self, target_value):
        target_value = target_value.view((2, int(len(target_value)//16), -1)+ target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1)+target_value.size()[-2:])
        target_value_ = rearrange(target_value, 'b h l d -> b l (h d)')
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0) # [77, 2, 2]
        basis_ortho = torch.stack([result[1] for result in results], dim=0) # [77, 2, 640]
        return project_matrix, basis_ortho
    
    def gram_schmidt(self, V): # [n, 1, d] 
        n = len(V)
        project_matrix = torch.zeros((n,n), dtype=V.dtype).to(V.device)+ torch.diag(torch.ones(n, dtype=V.dtype)).to(V.device)
        for i in range(1, n):
            vi = V[i:i+1, :]
            for j in range(i):
                qj = V[j:j+1, :]
                project_matrix[i][j] = -torch.dot(qj.view(-1), vi.view(-1)) / torch.dot(qj.view(-1), qj.view(-1))
        ortho_basis = torch.matmul(project_matrix.to(V.device), V)# n d
        
        return project_matrix, ortho_basis

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value) # [batch, head, len, dim//head]

        if not self.record and encoder_hidden_states.shape[1] == 77 and self.atten_type!='original':
            if 'queries' in self.target_records:
                erase_query, retain_query = self.record_ortho_decomp(
                    target_record=self.target_records['queries'],
                    current_record=query,
                )
                query = retain_query if self.atten_type == 'retain' else erase_query if self.atten_type == 'erase' else query
            if 'keys' in self.target_records:
                erase_key, retain_key = self.record_ortho_decomp(
                    target_record=self.target_records['keys'],
                    current_record=key,
                )
                key = retain_key if self.atten_type == 'retain' else erase_key if self.atten_type == 'erase' else key

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not self.record and encoder_hidden_states.shape[1] == 77 and self.atten_type!='original':
            if 'attn_maps' in self.target_records:
                erase_attention_probs, retain_attention_probs = self.record_ortho_decomp(
                    target_record=self.target_records['attn_maps'],
                    current_record=attention_probs,
                )
                attention_probs = retain_attention_probs if self.atten_type == 'retain' else erase_attention_probs if self.atten_type == 'erase' else attention_probs

        if encoder_hidden_states.shape[1] != 77:
            # self-attention
            hidden_states = torch.bmm(attention_probs, value)
        else:  
            # cross-attention
            if self.record:
                for kk, vv in {'queries': query, 'keys': key, 'values': value, 'attn_maps': attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1: # single-concept
                            self.records[kk][self.module_name] = [vv] + [None, None]
                        else: # multi-concept
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif self.atten_type=='original':
                value = value
            elif 'values' in self.target_records:
                erase_value, retain_value = self.record_ortho_decomp(
                    target_record=self.target_records['values'],
                    current_record=value,
                )
                value = retain_value if self.atten_type == 'retain' else erase_value if self.atten_type == 'erase' else value
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # # dropout

        if input_ndim == 4: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection: hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def set_attenprocessor(unet, atten_type='original', target_records=None, record=False, record_type=None, sigmoid_setting=None, token_sim=False, decomp_timestep=0):
    for name, m in unet.named_modules():
        if name.endswith('attn2') or name.endswith('attn1'):
            cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            m.set_processor(VisualAttentionProcess(
                module_name=name, 
                atten_type=atten_type,
                target_records=target_records, 
                record=record,
                record_type=record_type,
                cross_attention_dim=cross_attention_dim, 
                sigmoid_setting=sigmoid_setting,
                token_sim=token_sim,
                decomp_timestep=decomp_timestep
            ))
    return unet


def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, record=False, record_type=None, desc=None, **kwargs,):

    visualize_map_withstep = {key: {} for key in record_type.strip().split(',')} if record_type is not None else {}

    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample
        if record:
            for type in record_type.strip().split(','):
                for value in unet.attn_processors.values():  # This 'value' is different from the 'value' in CA/SA.
                    for k, v in value.records[type].items():
                        visualize_map_withstep[type][f'{timestep.item()}.{k}'] = v
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        ) # epsilon(t, x_t, c)
        
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map_withstep) if record else latents


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="models/stable-diffusion-v1-4")
    parser.add_argument('--seed', type=int, default=0)
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, retain')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=30, help='The total timesteps of the sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate' )
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size of the sampling process')
    parser.add_argument('--prompts', type=str, default=None)
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity, nsfw')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='erase, retention', help='Benchmark you want to use, I2P or your benchmark')
    args = parser.parse_args()

    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')

    # region [Prepare Models]
    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    if 'original' in mode_list: unet_original = copy.deepcopy(unet)
    if 'retain' in mode_list: unet_retain = copy.deepcopy(unet)
    # endregion

    # Sampling process
    if 'erase' in args.contents or 'retention' in args.contents:
        dataset = AdaDataset(data_path =f'data/{args.erase_type}.csv', guidance_scale=args.guidance_scale)
    dataloader = DataLoader(dataset, batch_size=bs, drop_last=False)
    target_concepts = [item.strip() for item in args.target_concept.split(', ')]
    
    # region [Prepare embeddings]
    target_concept_encoding = []
    for concept in target_concepts:
        concept_token = get_token(concept, tokenizer)
        target_eot_idxs = get_eot_idx(concept_token)
        target_concept_encodings_ = get_textencoding(concept_token, text_encoder)
        target_concept_encoding.append(get_spread_embedding(target_concept_encodings_, target_eot_idxs))
    target_concept_encoding = torch.concat(target_concept_encoding)
    uncond_encoding = get_textencoding(get_token('', tokenizer), text_encoder)
    # endregion

    for content in args.contents.split(', '):
        for count, data in enumerate(dataloader):
            if content in ['erase', 'retention'] and content != data['type'][0]:
                continue
            embedding = get_textencoding(get_token(data['prompt'], tokenizer), text_encoder)
            embedding_sim = torch.cosine_similarity(embedding.unsqueeze(1), target_concept_encoding, dim=-1).sum(-1).max(dim=-1)[-1].tolist()
            trans_target_encodings = trans_target(embedding_sim, copy.deepcopy(target_concept_encoding))
            unet = set_attenprocessor(unet, atten_type='original', record=True, record_type='values')
            _, target_records = diffusion(
                unet=unet, scheduler=pipe.scheduler, 
                latents=torch.zeros(len(target_concept_encoding), 4, 64, 64).to(pipe.device, dtype=target_concept_encoding.dtype),
                text_embeddings=torch.cat([uncond_encoding] * len(target_concept_encoding) + [trans_target_encodings], dim=0),
                total_timesteps=1, start_timesteps=0, guidance_scale=args.guidance_scale, 
                record=True, record_type='values', desc="Calculating target records",
            )
            pipe.scheduler.set_timesteps(args.total_timesteps)
            original_keys = target_records['values'].keys()
            target_records['values'].update({
                f"{timestep}.{'.'.join(key.split('.')[1:])}": target_records['values'][key]
                for timestep in pipe.scheduler.timesteps
                for key in original_keys
            })

            save_images = {}
            latent = torch.randn((bs, 4, 64, 64), generator=torch.Generator('cpu').manual_seed(data['seed'][0].item())).to(pipe.device, dtype=pipe.dtype)
            embedding = get_textencoding(get_token(data['prompt'], tokenizer), text_encoder)

            if 'original' in mode_list:
                unet_original = copy.deepcopy(unet)
                unet_retain = set_attenprocessor(unet_original, atten_type='original', record=False)
                save_images['original'] = diffusion(unet=unet_original, scheduler=pipe.scheduler, 
                                                latents=latent, start_timesteps=0, 
                                                text_embeddings=torch.cat([uncond_encoding] * bs + [embedding], dim=0), 
                                                total_timesteps=args.total_timesteps, 
                                                guidance_scale=args.guidance_scale, 
                                                desc=f"{count * len(data['prompt'])} x prompts | original")
                del unet_original
            if 'retain' in mode_list:
                unet_retain = copy.deepcopy(unet)
                unet_retain = set_attenprocessor(unet_retain, atten_type='retain', target_records=copy.deepcopy(target_records), 
                                                     sigmoid_setting=(100, 0.9, 2), 
                                                     token_sim=True)
                save_images['retain'] = diffusion(unet=unet_retain, scheduler=pipe.scheduler,
                                            latents=latent, start_timesteps=0, 
                                            text_embeddings=torch.cat([uncond_encoding] * bs + [embedding], dim=0), 
                                            total_timesteps=args.total_timesteps, 
                                            guidance_scale=args.guidance_scale, 
                                            desc=f"{count * len(data['prompt'])} x prompts | retain")
                del unet_retain
            
            save_path = os.path.join(args.save_root, args.erase_type, content)
            for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)
            if len(mode_list) > 1: os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

            # Decode and process images
            decoded_imgs = {
                name: [process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]) for img in img_list]
                for name, img_list in save_images.items()
            }

            # Save images
            def combine_images_horizontally(Images):
                widths, heights = zip(*(img.size for img in Images))
                new_img = Image.new('RGB', (sum(widths), max(heights)))
                for i, img in enumerate(Images): new_img.paste(img, (sum(widths[:i]), 0))
                return new_img
            for idx in range(len(decoded_imgs[mode_list[0]])):
                if content == 'nudity':
                    save_filename = f'{data["idx"][idx]}_' + re.sub(r'[^\w\s]', '', data["prompt"][idx]).replace(' ', '_')[:100]+ f"_{int(idx + bs * idx)}.png"
                elif content in ['erase', 'retain']:
                    save_filename = re.sub(r'[^\w\s]', '', data["prompt"][idx]).replace(' ', '_') + f'_{data["idx"][idx]}.png'
                images_to_combine = []
                for mode in mode_list: 
                    decoded_imgs[mode][idx].save(os.path.join(save_path, mode, save_filename))
                    images_to_combine.append(decoded_imgs[mode][idx])
                if len(mode_list) > 1:
                    img_combined = combine_images_horizontally(images_to_combine)
                    img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))

class AdaDataset(Dataset):
    def __init__(self, data_path, seed=None, guidance_scale=None, max_num=100000):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.prompt_list = list(self.data['text'])[:max_num]
        self.idx = list(self.data['id'])[:max_num]
        self.seed = list(self.data['seed'])
        self.concept = list(self.data['concept'])[:max_num]
        self.guidance_scale = [guidance_scale] * len(self.prompt_list)
        self.type = list(self.data['type'])[:max_num]

    def __getitem__(self, idx):
        item = {
            'prompt': self.prompt_list[idx],
            'idx': self.idx[idx],
            'seed': self.seed[idx],
            'guidance': self.guidance_scale[idx],
            'type': self.type[idx]
        }
        return item
    
    def __len__(self):
        return len(self.prompt_list)


if __name__ == '__main__':
    main()

