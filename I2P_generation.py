import torch
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import os
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
from sklearn.decomposition import PCA
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(description="Run SD projection experiment on I2P prompts.")
parser.add_argument("--start", type=int, default=0, help="Start index in CSV")
parser.add_argument("--end", type=int, default=None, help="End index in CSV")
parser.add_argument("--erasure_beta", type=float, default=1.0, help="Erasure beta value")
parser.add_argument("--preserve_gamma", type=float, default=1.0, help="Preserve gamma value")
parser.add_argument("--thr", type=float, default=14.0, help="Threshold")
parser.add_argument("--erase_rank", type=int, default=16, help="Erasure rank")
parser.add_argument("--start_time", type=float, default=0.9, help="Start time")
parser.add_argument("--output_dir", type=str, default="results_i2p_new_version", help="Directory to save results")
args = parser.parse_args()


home_dir = os.path.expanduser("~")
cache_dir = os.path.join(home_dir, "scratch/SD/")

model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    safety_checker=None
)


model = model.to("cuda")


import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np

tokenizer = model.tokenizer
text_encoder = model.text_encoder

device = "cuda"
target_word = [
    "Sexual Acts", "Content Meant To Arouse Sexual Excitement", "Nudity", "Pornography", "Erotic Art", 
    "Lustful", "Seductive", "Orgasmic", "Libido", "Nude Art", "Female Nude Painting", "Kinky", 
    "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse", "Sexually Attractive", "Porn Fantasy", 
    "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification", "Sexual Activity", 
    "Porn Category", "Porn", "Feet", "Belly", "Armpit", "Breast", "Buttocks", "Tits", "Nude", "Naked", "Boobs", 
    "Pussy", "Erotic Depiction", "Nudity Cartoon Style", "Skin Exposure", "Uncovering Skin", "Female Breast", 
    "Male Breast", "Undressed Body", "Naked Body Skin", "Human Body Exposed", "Uncovered Belly", "Female Nude Body", 
    "Barely", "Bare", "Revealed", "Nudity Artistic", "Woman Naked Body", "Skin Exposure", "Nudity Woman", 
    "Woman", "Female", "Nudity Cartoon", "Bare Skin", "Undressed", "Unclothed", "Female Skin", "Woman Breast", 
    "Exposed Breast", "Woman Body", "Woman Figure", "Human Body", "Female Body", "Erotic Content", 
    "Sensuality", "Lingerie", "Striptease", "Sensual Art", "Erotic Photography", "Seductive Pose", 
    "Sexual Symbolism", "BDSM", "Bondage", "Fetish", "Roleplay", "Tease", "Erotic Fantasies", "Intimate", 
    "Sexual Expression", "Sensual Touch", "Exposed Skin", "Erotic Imagery", "Desire", "Passion", "Tease", 
    "Stimulating", "Sexual Fantasy Artwork", "Sexual Objectification", "Exposed Intimacy", "Erotic Appeal", 
    "Detailed Body Parts", "Explicit Body Details", "Body Exposure", "Explicit Anatomy", "Exposed Genitalia", 
    "Detailed Breast", "Detailed Buttocks", "Exposed Navel", "Visible Genitals", "Naked Figures", "Exposed Thighs", 
    "Detailed Skin", "Sexualized Body", "Explicit Body Art", "Body Exposure Art", "Anatomical Exposure", "Intimate Body Parts", 
    "Nude Anatomy", "Exposed Private Parts", "Flesh Exposure", "Sensual Body Art", "Explicit Body Depictions", 
    "Sexualized Imagery", "Exposed Curves", "Erotic Body Positioning", "Sexual Detail", "Body and Flesh", "Body Exposed in Detail"
]


manual_neighbors = [
 'Clothed',
 'Dressed',
 'Bodysuit',
 'Covered',
 "Masked"]

def encode_text(text_list):
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(
            inputs.input_ids,
            return_dict=True
        )

    last_hidden = outputs.last_hidden_state  # (B, L, D)
    attention_mask = inputs.attention_mask   # (B, 77)
    attention_mask[:, 0] = 0  # ensure CLS token is not included
    B, L, D = last_hidden.shape
    last_hidden = last_hidden.view(B * L, D)
    attention_mask = attention_mask.view(B * L)

    # Only return embeddings where attention_mask == 1 (i.e., non-padding)
    non_padded_hidden = last_hidden[attention_mask.bool()]  # (total_non_pad_tokens, D)
    return non_padded_hidden


def spectral_expansion_weights(S, alpha):
    spectral_energy = S ** 2
    r = spectral_energy / spectral_energy.sum()
    weights = (alpha * r) / ((alpha - 1) * r + 1)
    return weights


e_targets = encode_text(target_word).float().cpu()   # [n, d]
d = e_targets.shape[1]
target_aggregation = "svd"

if target_aggregation == "mean":
    e_target = e_targets.mean(dim=0)
    u_f = e_target / e_target.norm(dim = 0)  # [d]
    U_F = u_f.unsqueeze(1)            # [d, 1]
    P_F = U_F @ U_F.T                 # [d, d]
    u_f = torch.tensor(u_f)
else:
    E_t_np = e_targets.numpy()
    U_t, S_t, Vh_t = np.linalg.svd(E_t_np, full_matrices=False)
    spectral_w = spectral_expansion_weights(S_t, 200)
    rank_erase = min(args.erase_rank, Vh_t.shape[0])
    Vh_t = Vh_t[:rank_erase]
    #  np.diag(spectral_w[:rank_erase]) 
    P_F = Vh_t.T @ np.diag(spectral_w[:rank_erase]) @ Vh_t
    P_F = torch.tensor(P_F)
    u_f = torch.tensor(Vh_t.T)
    pc = Vh_t.T
    e_target = torch.from_numpy(pc).to(e_targets.dtype)



if manual_neighbors is not None:
    neighbor_tokens = manual_neighbors
    print(f"Using manual neighbors ({len(neighbor_tokens)}):", neighbor_tokens)
else:
    vocab_dict = tokenizer.get_vocab()
    vocab_tokens = [t for t in vocab_dict.keys() if t.isalpha() and len(t) >= 3]
    vocab_tokens = list(set(vocab_tokens))
    print("Vocabulary size:", len(vocab_tokens))

    batch_size = 512
    embeddings = []
    for i in range(0, len(vocab_tokens), batch_size):
        batch = vocab_tokens[i:i + batch_size]
        emb = encode_text(batch).cpu()
        embeddings.append(emb)
    vocab_embeddings = torch.cat(embeddings, dim=0)
    target_word = target_word[0]
    similarities = F.cosine_similarity(e_target.unsqueeze(0), vocab_embeddings)
    top_k = 64
    top_indices = torch.topk(similarities, top_k + 1).indices
    neighbor_indices = [i for i in top_indices.tolist() if vocab_tokens[i].lower() != target_word.lower()][:top_k]

    neighbor_tokens = [vocab_tokens[i] for i in neighbor_indices]
    E_N = vocab_embeddings[neighbor_indices]
    print(f"Automatic neighbors ({len(neighbor_tokens)}):", neighbor_tokens)


if manual_neighbors is not None:
    E_N = encode_text(neighbor_tokens).float().cpu()



E_N_np = E_N.cpu().numpy()                            # [k, d]
U, S, Vh = np.linalg.svd(E_N_np, full_matrices=False)  # U: [k, k], S: [k], Vh: [k, d]

U_c = torch.from_numpy(Vh.T).to(e_target.dtype).to(device)  # [d, k]

# spectral_components = min(4, len(neighbor_tokens))
spectral_components = min(E_N_np.shape[0], 1)
S_tensor = S[:spectral_components]                         # [c]
Vh_top = Vh[:spectral_components, :]                    # [c, d]
U_c = torch.from_numpy(Vh_top.T).to(e_target.dtype).to(device)  # [d, c]

alpha = 100
weights = spectral_expansion_weights(S_tensor, alpha)  # [k]
Λ = torch.diag(torch.from_numpy(weights).to(e_target.dtype).to(device))  # [k, k]
P_R = U_c @ Λ @ U_c.T  # [d, d]
print("Spectral expansion projection matrix shape:", P_R.shape)

d = e_targets.shape[1]
I = torch.eye(d, device=device)
P_c = (I - args.erasure_beta * P_F.to(device)) + args.preserve_gamma * torch.matmul(P_F.to(device),P_R.to(device))

P_c = P_c.to(device).to(torch.float16)
print("Projection matrix P_c shape:", P_c.shape)


import copy
from diffusers.models.attention_processor import Attention

model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    safety_checker=None
)

model = model.to("cuda")
model_copy = copy.deepcopy(model)


class ProcessorState:
    def __init__(
        self,
        using_prompt_tokens: bool = False,
        decay_threshold: bool = False,
        apply_projection_on_weights: bool = False,
        stage_3: bool = False,
        attention_supression: bool = False,
        hard_soft_scrub: bool = False,
    ):
        self.saved_masks = {}
        self.is_second_pass = False

        # flags
        self.using_prompt_tokens = using_prompt_tokens
        self.decay_threshold = decay_threshold
        self.apply_projection_on_weights = apply_projection_on_weights
        self.stage_3 = stage_3
        self.attention_supression = attention_supression
        self.hard_soft_scrub = hard_soft_scrub


orig_forward = model.unet.forward
def make_double_forward(orig_forward, state):
    def double_forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        for m in self.modules():
            if isinstance(m, Attention):
                m.processor_state = state
                m.current_timestep = timestep

        # state.saved_masks.clear()
        state.is_second_pass = False

        # Pass 1 → compute masks
        _ = orig_forward(sample, timestep, encoder_hidden_states, **kwargs)

        # Pass 2 → apply masks
        state.is_second_pass = True
        noise_pred_2 = orig_forward(sample, timestep, encoder_hidden_states, **kwargs)
        state.is_second_pass = False

        return noise_pred_2
    return double_forward




state = ProcessorState(using_prompt_tokens = True,
stage_3 = True,
attention_supression = True,
apply_projection_on_weights = True,
decay_threshold = True,
hard_soft_scrub = True
)

model.unet.forward = make_double_forward(orig_forward, state).__get__(model.unet, type(model.unet))


import math

MASK_THR = 0.01
START_TIME = args.start_time
T = model.scheduler.num_train_timesteps

def make_wrapped_processor(orig_proc, layer_name, state, P_c, u_f):
    orig_call = orig_proc.__call__

    class WrappedProcessor:
        __class__ = orig_proc.__class__
        def __init__(self, inner):
            self._inner = inner
            self.original_weights = None
            self.state = state
            self.P_c = P_c
            self.u_f = u_f
            self.__dict__.update(inner.__dict__)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, attn, hidden_states,
                     encoder_hidden_states=None, attention_mask=None, temb=None,
                     *args, **kwargs):

            t = attn.current_timestep
            # only intervene on cross-attn
            if encoder_hidden_states is None or (t > START_TIME * T):
                return orig_call(attn, hidden_states, encoder_hidden_states,
                                 attention_mask, temb, *args, **kwargs)

            
            # pass 1: compute & save mask at Down-block 2 
            if not self.state.is_second_pass and "down_blocks.2.attentions.1" in layer_name:
                original_hidden_states = hidden_states
                original_encoder_hidden_states = encoder_hidden_states

                hidden_states  = hidden_states.clone()                     
                encoder_hidden_states = encoder_hidden_states.clone()
                if not self.state.using_prompt_tokens:
                    encoder_hidden_states = torch.cat([e_target, e_target], dim=0)
                # hidden_states = hidden_states[1,...].unsqueeze(dim=0)
                # print(hidden_states.shape)
                
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    # scaled_dot_product_attention expects attention_mask shape to be
                    # (batch, heads, source_length, target_length)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)


                B, H, Q, D = query.shape
                _, _, K, _ = key.shape
                query = query.reshape(B * H, Q, D)   
                key = key.reshape(B * H, K, D)
                scores = attn.get_attention_scores(query, key, attention_mask=None).detach().cpu()
                scores = scores.view(B, H, Q, K)
                scores = scores.mean(dim=1)
                if not self.state.using_prompt_tokens:
                    scores = scores[...,1].float()
                else:
                    # print(indices)
                    scores = scores[:, :, indices]
                    scores = scores.mean(dim=2).float()
                    
    
                mask = scores.float()
                self.state.saved_masks[t] = mask
                
                return orig_call(attn,
                        original_hidden_states,
                        encoder_hidden_states=original_encoder_hidden_states,
                        attention_mask=attention_mask,
                        temb=temb,
                        *args, **kwargs) 
                 
            # pass 2: apply that mask on every cross-attn output 
            if self.state.is_second_pass and t in self.state.saved_masks:
                dev, dtype = hidden_states.device, hidden_states.dtype
       
                
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

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    # scaled_dot_product_attention expects attention_mask shape to be
                    # (batch, heads, source_length, target_length)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                
                
                # apply projection    
                if self.state.apply_projection_on_weights:
                    encoder_hidden_states = encoder_hidden_states @ self.P_c

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                
        

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)

                
                # resizing mask
                
                mask_orig = self.state.saved_masks[t]
                # checking the size for upsampling or downsampling
                B, old_seq = mask_orig.shape
                new_seq = hidden_states.shape[1]
                if new_seq != old_seq:
                    old_size = int(math.sqrt(old_seq))
                    new_size = int(math.sqrt(new_seq))
                    assert old_size * old_size == old_seq, f"{old_seq} not a perfect square"
                    assert new_size * new_size == new_seq, f"{new_seq} not a perfect square"
                    mask_grid = mask_orig.view(B, 1, old_size, old_size)     
                    mask_resized = F.interpolate(
                        mask_grid,
                        size=(new_size, new_size),
                        mode="bilinear",     
                        align_corners=None
                    )
                    mask = mask_resized.view(B, new_seq)                       
                else:
                    mask = mask_orig  
                        
                        
                        
                # attention suppresion
                B, H, Q, D = query.shape
                K = key.shape[2]
                # d = D
                attn_mask = torch.zeros((B, H, Q, K), dtype=query.dtype, device=query.device)
                if self.state.attention_supression:
                    sliced_mask = torch.where(mask > MASK_THR , torch.full_like(mask, -torch.inf), torch.zeros_like(mask))
                    sliced_mask = sliced_mask.unsqueeze(1).unsqueeze(-1)   # (B,1,Q,1)
                    attn_mask[..., indices] = sliced_mask.to(attn_mask.device, dtype=attn_mask.dtype)
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                )
            
            
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)

                # stage 3  
                if self.state.stage_3 and len(indices)!=0:
                    h = hidden_states
                    dev, dtype = h.device, h.dtype
                    binary_mask = (mask > MASK_THR).to(dtype=dtype, device=dev).unsqueeze(-1)
                    out = (1. - binary_mask) * h 
                    hidden_states = out

                

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor
                
                return hidden_states
 

            return orig_call(attn, hidden_states, encoder_hidden_states,
                             attention_mask, temb, *args, **kwargs)

    return WrappedProcessor(orig_proc)

# patch all attention layers in the UNet
for name, module in model.unet.named_modules():
    if isinstance(module, Attention):
        proc = module.get_processor()
        module.set_processor(make_wrapped_processor(proc, name, state, P_c, None))


i2p = pd.read_csv("I2P_prompts_4703.csv")

from tqdm import tqdm

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model.set_progress_bar_config(leave=False)
model.set_progress_bar_config(disable=True)

start_index = args.start
end_index = args.end

for idx, row in tqdm(i2p.iloc[start_index:end_index].iterrows(), 
                     total=len(i2p), 
                     initial=start_index, 
                     desc="Generating images"):
    prompt = row["prompt"]
    if pd.isna(prompt):
        continue
    seed = row["evaluation_seed"]
    evaluation_guidance = row["evaluation_guidance"]
    attention_mask_for_pad = model.tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True).attention_mask.to(device)  
    num_tokens = attention_mask_for_pad.sum(dim=1).item()
    s_j = (model.encode_prompt(prompt, device=device, do_classifier_free_guidance=False, num_images_per_prompt=1)[0].float() @ P_F.to(model.device)).norm(dim=-1)
    s_j_valid = s_j[:,1:num_tokens-1]   
    delta = max(0.6 * s_j_valid.max(), args.thr)
    indices = (s_j_valid.squeeze(0) > delta).nonzero(as_tuple=True)[0].cpu() + 1
    state.is_second_pass = False
    state.saved_masks = {}
    generator = torch.Generator(device).manual_seed(seed) 
    new_img = model(prompt=prompt, generator=generator, guidance_scale=evaluation_guidance)[0][0]
    filename = row.get("filename")
    if pd.isna(filename) or not str(filename).strip():
        safe_prompt = prompt.strip().replace(" ", "_")
        index_number = row["Unnamed: 0"]
        filename = f"{index_number}.png"
    filepath = os.path.join(output_dir, filename)
    if isinstance(new_img, torch.Tensor):
        from torchvision.utils import save_image
        save_image(new_img, filepath)
    elif hasattr(new_img, "save"):  # PIL Image
        new_img.save(filepath)
    else:
        raise TypeError(f"Unsupported image type: {type(new_img)}")

    