import os
from PIL import Image
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from argparse import ArgumentParser
import torch

@torch.no_grad()
def mean_clip_score(image_dir, prompts_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="../scratch/").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="../scratch/")

    # Load prompts
    df = pd.read_csv(prompts_path)
    prompts = list(df["prompt"])

    similarities = []

    print("Computing CLIP scores for image-prompt pairs...")

    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        try:
            filename = f"coco_sample_{idx:03}.png"
            image_path = os.path.join(image_dir, filename)

            if not os.path.exists(image_path):
                print(f"Image not found: {filename}")
                continue

            image = Image.open(image_path).convert("RGB")

            inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            clip_score = outputs.logits_per_image[0][0].item()  # scalar

            similarities.append(clip_score)

        except Exception as e:
            print(f"Error with {filename}: {e}")
            continue

    similarities = np.array(similarities)

    if len(similarities) == 0:
        print("No CLIP scores could be computed.")
        return None

    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    print('\n-------------------------------------------------')
    print(f"Mean CLIP score ± Standard Deviation: {mean_similarity:.4f} ± {std_similarity:.4f}")

    return mean_similarity

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='./output_coco_sample')
    parser.add_argument("--prompts_path", type=str, default='./sampled_coco_150.csv')
    args = parser.parse_args()

    mean_clip_score(args.image_dir, args.prompts_path)
