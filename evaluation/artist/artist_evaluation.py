# =======================
# GPT image classification
# =======================
import os
import re
import csv
import base64
import random
from openai import OpenAI
from tqdm import tqdm

# --- Global configuration knobs ---
TEMPERATURE = 0.0              # 1) Deterministic decoding (temperature = 0)
RANDOMIZE_OPTIONS = True       # 2) Randomize the order of artist options
RANDOM_SEED = 42               #    Set to None for non-deterministic shuffling


def encode_image(image_path):
    """Read image and return base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def make_prompt(options):
    """Constructs a classification prompt with a list of options."""
    options_text = "\n  - " + "\n  - ".join(options)
    prompt = f"""Which artist's style does the following image most resemble?
The options are as follows:{options_text}

Please respond in the following format: ANS: X
Where X is the selected artist option."""
    return prompt

# Artist label sets
allowed_artists_van_gogh = ["Van Gogh", "Pablo Picasso", "Rembrandt", "Andy Warhol", "Caravaggio"]
allowed_artists_kelly_mckernan = ["Tyler Edlin", "Thomas Kinkade", "Kilian Eng", "Kelly McKernan", "Demi Human"]
allowed_artists_table = {"van-gogh": allowed_artists_van_gogh, "kelly-mckernan": allowed_artists_kelly_mckernan}

def gpt_classify(prompt_text, base64_image, client, allowed_artists):
    """
    Run GPT-5 image classification with:
      - deterministic temperature = 0
      - explicit base64 input (bug fixed)
      - retries up to 5 times if output malformed
    """
    for _ in range(5):
        response = client.responses.create(
            model="gpt-5",
            temperature=TEMPERATURE,  # Deterministic output
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    ],
                }
            ],
        )

        parts = response.output_text.split(":")
        if len(parts) != 2:
            continue

        prediction = parts[1].strip()
        if prediction in allowed_artists:
            return prediction

    print("Failed to obtain a valid prediction.")
    return "unknown"


# Location of all image folders
variant_base_templates = {
}

artists = ["van-gogh", "kelly-mckernan"]

# Optional reproducibility for randomized option order
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# ================================================
# Run GPT Classification
# ================================================
for config in artists:
    print(f"\n Processing CONFIG: {config}")

    for variant in variant_base_templates.keys():
        print(f"   Variant: {variant.upper()}")

        folder_path = variant_base_templates[variant].format(config=config)
        if not os.path.exists(folder_path):
            print(f"   Missing folder: {folder_path}")
            continue

        _, _, files = next(os.walk(folder_path))
        print("Files:", len(files))

        # Determine where to write results
        results_folder = (
            os.path.dirname(os.path.dirname(folder_path))
            if variant in ["ours", "cure", "orig"]
            else os.path.dirname(folder_path)
        )
        results_file = os.path.join(results_folder, f"results_{variant}_{config}.csv")

        if os.path.exists(results_file):
            print(f"   Already exists: {results_file}")
            continue

        results = []
        for filename in tqdm(os.listdir(folder_path)):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(folder_path, filename)
            b64_img = encode_image(image_path)

            # Randomize options per image
            options = list(allowed_artists_table[config])
            if RANDOMIZE_OPTIONS:
                random.shuffle(options)

            # Apply GPT classification
            prediction = gpt_classify(make_prompt(options), b64_img, client, set(options))

            # Extract ground-truth artist from filename
            isMatch = re.search(r'_([a-zA-Z_]+)_\d+', filename)
            true_artist = isMatch.group(1) if isMatch else "unknown"

            is_correct = (
                true_artist.replace("_", "").replace(" ", "").lower() ==
                prediction.replace(" ", "").replace("_", "").lower()
            )

            results.append({
                "filename": filename,
                "true_artist": true_artist,
                "predicted_artist": prediction,
                "is_correct": is_correct,
                "options_order": "|".join(options),  # Helpful for debugging
            })

        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "true_artist", "predicted_artist", "is_correct", "options_order"],
            )
            writer.writeheader()
            writer.writerows(results)

        print(f" Saved: {results_file}")


# =======================
# LPIPS with configurable backbone
# =======================
import torch
import lpips
from PIL import Image
from torchvision import transforms
import pandas as pd

LPIPS_BACKBONE = "vgg"  # 3) Change LPIPS backbone ('alex', 'vgg', 'squeeze')

# Initialize LPIPS with a different backbone
lpips_model = lpips.LPIPS(net=LPIPS_BACKBONE).cuda()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def load_image_as_tensor(path):
    """Load and preprocess image for LPIPS."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).cuda()

# Remaining LPIPS evaluation code stays unchanged
