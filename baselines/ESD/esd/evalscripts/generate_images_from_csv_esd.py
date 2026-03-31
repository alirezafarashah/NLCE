from diffusers import DiffusionPipeline
import torch
import pandas as pd
import argparse
import os
import re
from safetensors.torch import load_file
from pathlib import Path

torch.enable_grad(False)


def generate_images_from_csv_esd(
    base_model: str,
    esd_path: str,
    csv_path: str,
    save_path: str,
    device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.float16,
):
    """
    Generate images using a pretrained diffusion model (optionally ESD-modified)
    from a CSV containing species, prompt, evaluation_seed, and type columns.
    """
    # Derive model name
    if esd_path is not None:
        model_name = os.path.basename(esd_path).split(".")[0]
    else:
        if "xl" in base_model:
            model_name = "sdxl"
        elif "Comp" in base_model:
            model_name = "sdv14"
        else:
            model_name = "custom"

    print(f"🚀 Loading model: {base_model}")
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch_dtype, safety_checker=None).to(device)

    # Load erased/styled UNet weights if provided
    if esd_path is not None:
        try:
            esd_weights = load_file(esd_path)
            pipe.unet.load_state_dict(esd_weights, strict=False)
            print(f"✅ Loaded ESD weights from: {esd_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load ESD weights: {e}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} prompts from {csv_path}")

    # Loop over each row
    for idx, row in df.iterrows():
        species = str(row["species"])
        prompt = str(row["prompt"])
        seed = int(row["evaluation_seed"])
        concept_type = str(row["type"])

        # Sanitize filename
        safe_species = re.sub(r'[\\/*?:"<>|]', "_", species.replace(" ", "_").replace("/", "_").replace("-", "_"))

        folder_path = Path(f"{save_path}")
        folder_path.mkdir(parents=True, exist_ok=True)

        method_name = "esdx" if "esdx" in esd_path else "esdu" if "esdu" in esd_path else "esd"
        filename = f"{method_name}_{safe_species}_{idx:04d}_seed{seed}_{concept_type}_img0.png"
        file_path = folder_path / filename

        if file_path.exists():
            print(f"✅ File already exists: {filename}")
            continue

        print(f"\n🎨 Generating for prompt: {prompt} | Seed: {seed} | Type: {concept_type}")

        generator = torch.Generator(device=device).manual_seed(seed)

        # Run diffusion
        images = pipe(
            prompt,
            width=512,
            height=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            num_images_per_prompt=1,
        ).images

        for i, im in enumerate(images):
            im.save(file_path)
            print(f"💾 Saved: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from CSV using (E)SD model.")
    parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base model to load.")
    parser.add_argument("--esd_path", type=str, default=None, help="Path to ESD weights (.safetensors).")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with species, prompt, evaluation_seed, type.")
    parser.add_argument("--save_path", type=str, default="esd-images/", help="Directory to save generated images.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to run on.")
    parser.add_argument("--precision", type=str, default="fp16", help="Precision mode (fp32, fp16, bf16).")
    args = parser.parse_args()

    # Load config

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(args.precision, torch.float16)

    generate_images_from_csv_esd(
        base_model=args.base_model,
        esd_path=args.esd_path,
        csv_path=args.csv_path,
        save_path=args.save_path,
        device=args.device,
        torch_dtype=torch_dtype,
    )
