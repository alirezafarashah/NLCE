import os, pdb
import random
import torch
import numpy as np
import safetensors
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file


def seed_everything(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_token(prompt, tokenizer=None):
    tokens = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    return tokens


def get_textencoding(input_tokens, text_encoder):
    text_encoding = text_encoder(input_tokens.to(text_encoder.device))[0]
    return text_encoding


def get_eot_idx(tokens): 
    return (tokens == 49407).nonzero(as_tuple=True)[1][0].item()


def get_spread_embedding(original_token, idx):
    spread_token = original_token.clone()
    spread_token[:, 1:, :] = original_token[:, idx-1, :].unsqueeze(1)
    return spread_token


def process_img(decoded_image):
    decoded_image = decoded_image.squeeze(0)
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = (decoded_image * 255).byte()
    decoded_image = decoded_image.permute(1, 2, 0)

    decoded_image = decoded_image.cpu().numpy()
    return Image.fromarray(decoded_image)

        