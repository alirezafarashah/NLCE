import argparse
import torch
import pandas as pd
import os
from sld import SLDPipeline
from PIL import Image
import re

@torch.no_grad()
def generate_images(base_model,
                    prompts_path,
                    save_path,
                    device='cuda:0',
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    num_samples=1,
                    from_case=0,
                    safety_concept=None):
    """
    Generate images using SLDPipeline.

    The CSV should have headers:
        - 'case_number': unique ID for saving images
        - 'prompt': text prompt
        - 'seed': random seed for reproducibility
    """

    # Load model
    pipe = SLDPipeline.from_pretrained(base_model, safety_checker=None).to(device)

    if safety_concept is not None:
        print(f"Original safety concept: {pipe.safety_concept}")
        pipe.safety_concept = safety_concept
        print(f"Updated safety concept: {pipe.safety_concept}")

    # Load prompts
    df = pd.read_csv(prompts_path)

    # Create save directory
    model_name = os.path.basename(base_model).replace('/', '_')
    folder_path = os.path.join(save_path, model_name)
    os.makedirs(folder_path, exist_ok=True)

    for idx, row in df.iterrows():
        species = str(row["species"])
        prompt = str(row["prompt"])
        seed = int(row["evaluation_seed"])
        concept_type = str(row["type"])

        # Filename-safe version of species
        safe_species = re.sub(r'[\\/*?:"<>|]', "_", species.replace(" ", "_").replace("/", "_").replace("-", "_"))

        generator = torch.Generator(device=device).manual_seed(seed)

        for sample_idx in range(num_samples):
            output = pipe(
                prompt=prompt,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            img: Image.Image = output.images[0]

            filename = f"sld_{safe_species}_{idx:04d}_seed{seed}_{concept_type}_img0.png"
            img.save(os.path.join(folder_path, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images using SLDPipeline")
    parser.add_argument('--base_model', type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Base model path or HuggingFace repo ID")
    parser.add_argument('--prompts_path', type=str, required=True,
                        help="CSV file with 'case_number', 'prompt', 'seed'")
    parser.add_argument('--save_path', type=str, default="sld-images/",
                        help="Folder where images will be saved")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale")
    parser.add_argument('--num_inference_steps', type=int, default=50, help="Inference steps")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples per prompt")
    parser.add_argument('--safety_concept', type=str, default=None,
                        help="Safety concept to use (e.g. 'chesapeake bay retriever')")

    args = parser.parse_args()

    generate_images(
        base_model=args.base_model,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
        safety_concept=args.safety_concept
    )
