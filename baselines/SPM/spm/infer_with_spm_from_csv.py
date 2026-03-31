import argparse
import gc
import os
import re
from pathlib import Path
from typing import Literal

import torch
import pandas as pd

from src.configs.generation_config import load_config_from_yaml, GenerationConfig
from src.configs.config import parse_precision
from src.engine import train_util
from src.models import model_util
from src.models.spm import SPMLayer, SPMNetwork
from src.models.merge_spm import load_state_dict

DEVICE_CUDA = torch.device("cuda:0")
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def calculate_matching_score(
    prompt_tokens,
    prompt_embeds, 
    erased_prompt_tokens, 
    erased_prompt_embeds, 
    matching_metric: MATCHING_METRICS,
    special_token_ids: set[int],
    weight_dtype: torch.dtype = torch.float32,
):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
            prompt_embeds.flatten(1, 2), 
            erased_prompt_embeds.flatten(1, 2), 
            dim=-1
        ).cpu()
        scores.append(clipcos)

    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu", dtype=weight_dtype))

    return torch.max(torch.stack(scores), dim=0)[0]


def infer_with_spm_from_csv(
    csv_path: str,
    spm_paths: list[str],
    config: GenerationConfig,
    matching_metric: MATCHING_METRICS,
    assigned_multipliers: list[float] = None,
    base_model: str = "CompVis/stable-diffusion-v1-4",
    v2: bool = False,
    precision: str = "fp32",
):
    # Load SPMs
    spm_model_paths = [Path(lp) / f"{Path(lp).name}_last.safetensors" if Path(lp).is_dir() else Path(lp) for lp in spm_paths]
    weight_dtype = parse_precision(precision)

    # Load the pretrained SD
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype).eval()
    unet.to(DEVICE_CUDA, dtype=weight_dtype).eval()
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)

    # Load SPM modules
    spms, metadatas = zip(*[load_state_dict(spm_model_path, weight_dtype) for spm_model_path in spm_model_paths])
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    erased_prompts_flatten = [p for sublist in erased_prompts for p in sublist]
    print(f"Erased prompts: {erased_prompts}")

    erased_prompt_embeds, erased_prompt_tokens = train_util.encode_prompts(
        tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True
    )

    network = SPMNetwork(
        unet,
        rank=int(float(metadatas[0]["rank"])),
        alpha=float(metadatas[0]["alpha"]),
        module=SPMLayer,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} prompts from {csv_path}")

    for idx, row in df.iterrows():
        species = str(row["species"])
        prompt = str(row["prompt"])
        seed = int(row["evaluation_seed"])
        concept_type = str(row["type"])

        # Filename-safe version of species
        safe_species = re.sub(r'[\\/*?:"<>|]', "_", species.replace(" ", "_").replace("/", "_").replace("-", "_"))

        print(f"\n[Case {idx}] Generating for prompt: {prompt} | Seed: {seed} | Type: {concept_type}")

        # Encode prompt
        prompt_embeds, prompt_tokens = train_util.encode_prompts(
            tokenizer, text_encoder, [prompt], return_tokens=True
        )

        # Determine multipliers
        if assigned_multipliers is not None:
            multipliers = torch.tensor(assigned_multipliers).to("cpu", dtype=weight_dtype)
        else:
            multipliers = calculate_matching_score(
                prompt_tokens,
                prompt_embeds, 
                erased_prompt_tokens, 
                erased_prompt_embeds, 
                matching_metric=matching_metric,
                special_token_ids=special_token_ids,
                weight_dtype=weight_dtype
            )
            multipliers = torch.split(multipliers, erased_prompts_count)

        weighted_spm = dict.fromkeys(spms[0].keys())
        used_multipliers = []

        for spm, multiplier in zip(spms, multipliers):
            max_multiplier = torch.max(multiplier)
            for key, value in spm.items():
                if weighted_spm[key] is None:
                    weighted_spm[key] = value * max_multiplier
                else:
                    weighted_spm[key] += value * max_multiplier
            used_multipliers.append(max_multiplier.item())

        network.load_state_dict(weighted_spm)

        # Output directory
        folder_path = Path(f"{config.save_path}")
        folder_path.mkdir(parents=True, exist_ok=True)

        filename = f"spm_{safe_species}_{idx:04d}_seed{seed}_{concept_type}_img0.png"
        file_path = folder_path / filename

        if file_path.exists():
            print(f"✅ File already exists: {filename}")
            continue

        with torch.no_grad(), network:
            images = pipe(
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.manual_seed(seed),
                num_images_per_prompt=config.generate_num,
                prompt_embeds=prompt_embeds,
            ).images

        for i, image in enumerate(images):
            image.save(file_path)
            print(f"💾 Saved: {file_path}")

        flush()


def main(args):
    spm_path = [Path(lp) for lp in args.spm_path]
    generation_config = load_config_from_yaml(args.config)

    infer_with_spm_from_csv(
        args.csv_path,
        spm_path,
        generation_config,
        args.matching_metric,
        assigned_multipliers=args.spm_multiplier,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generation.yaml", help="Base configs for image generation.")
    parser.add_argument("--csv_path", required=True, help="Path to CSV file containing prompts.")
    parser.add_argument("--spm_path", required=True, nargs="*", help="SPM(s) to use.")
    parser.add_argument("--spm_multiplier", nargs="*", type=float, default=None, help="Manual SPM multipliers.")
    parser.add_argument("--matching_metric", type=str, default="clipcos_tokenuni", help="Matching metric to use.")
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base model name.")
    parser.add_argument("--v2", action="store_true", help="Use Stable Diffusion 2.x model.")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision (fp32/fp16/bf16).")
    args = parser.parse_args()

    main(args)
