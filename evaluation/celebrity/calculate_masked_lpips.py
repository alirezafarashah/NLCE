import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms
import torch.nn.functional as F

# --- Helper functions for LPIPS ---

def load_image_tensor(image_path, target_size, device):
    """
    Loads an image, resizes it, and converts to a
    normalized tensor for LPIPS.
    """
    img = Image.open(image_path).convert('RGB')
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    
    # Transform to tensor and normalize to [-1, 1]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    tensor = normalize(to_tensor(img)).unsqueeze(0).to(device)
    return tensor, img.size

def load_mask_tensor(mask_path, target_size, device):
    """
    Loads the mask, resizes it to the LPIPS map size,
    and inverts it.
    """
    mask = Image.open(mask_path).convert('L')
    
    # Transform to tensor
    to_tensor = transforms.ToTensor()
    mask_tensor = to_tensor(mask).unsqueeze(0).to(device)
    
    # Resize mask to the target map size
    # We use 'nearest' to avoid creating new in-between values
    resized_mask = F.interpolate(mask_tensor, size=target_size, mode='nearest')
    
    # Invert the mask: 1.0 for background, 0.0 for face
    mask_inv = 1.0 - resized_mask
    
    return mask_inv

def find_generated_image(generated_folder, original_stem):
    """
    Finds the matching generated image, trying common extensions.
    (e.g., .jpg, .png)
    """
    extensions = ['.png', '.jpg', '.jpeg']
    for ext in extensions:
        gen_path = generated_folder / f"{original_stem}{ext}"
        if gen_path.exists():
            return gen_path
    return None

def compute_masked_lpips(
    lpips_model, 
    original_img_path, 
    generated_img_path, 
    mask_path, 
    device
):
    """
    Calculates LPIPS score only on the background (inverted mask area).
    """
    
    # 1. Load original image to get the definitive size
    try:
        orig_img_pil = Image.open(original_img_path)
        target_size = orig_img_pil.size
        
        orig_tensor, _ = load_image_tensor(original_img_path, target_size, device)
        
        # 2. Load generated image, resizing to match original
        gen_tensor, _ = load_image_tensor(generated_img_path, target_size, device)
    
    except FileNotFoundError as e:
        print(f"  [Error] Could not load image: {e}")
        return None
    except Exception as e:
        print(f"  [Error] Processing images {original_img_path.name}: {e}")
        return None
        
    with torch.no_grad():
        # 3. Get the spatial LPIPS map (not the final avg score)
        lpips_map = lpips_model(orig_tensor, gen_tensor)
        
        # Squeeze batch and channel dims (e.g., from 1,1,H,W to H,W)
        lpips_map = lpips_map.squeeze() 
        map_size = (lpips_map.shape[0], lpips_map.shape[1])

        # 4. Load and resize the mask to match the LPIPS map
        try:
            mask_inv = load_mask_tensor(mask_path, map_size, device)
            mask_inv = mask_inv.squeeze() # Match lpips_map dims
        except FileNotFoundError:
            print(f"  [Error] Mask not found: {mask_path}")
            return None
        
        # 5. Calculate the mean LPIPS *only* on the background
        
        # Get the sum of all LPIPS differences in the background
        masked_lpips_sum = (lpips_map * mask_inv).sum()
        
        # Get the total number of "background" pixels in the map
        mask_inv_sum = mask_inv.sum()
        
        if mask_inv_sum == 0:
            # This happens if the mask covers the entire image
            return 0.0 
        
        # The final score is the mean of *only* the background pixels
        mean_lpips_score = masked_lpips_sum / mask_inv_sum
        
        return mean_lpips_score.item()


def main_lpips():
    """
    Main execution function to calculate LPIPS.
    """
    
    # --- Configuration ---
    #Update this path to your generated images !!
    GENERATED_IMAGES_BASE_FOLDER = "PATH/TO/YOUR/GENERATED/IMAGES" 

    
    # These paths point to the outputs of your *first* script
    ORIGINAL_IMAGES_BASE_FOLDER = "./original_celebrity_images"
    OUTPUT_BASE_FOLDER = "./output_sam"
    celebrities = {
            "elon_musk": {
                "original": f"{ORIGINAL_IMAGES_BASE_FOLDER}/Elon_Musk",
                "generated": f"{GENERATED_IMAGES_BASE_FOLDER}/Elon_Musk",
                "output": f"{OUTPUT_BASE_FOLDER}/elon_musk",
                "name": "Elon Musk"
            },
            "anna_kendrick": {
                "original": f"{ORIGINAL_IMAGES_BASE_FOLDER}/Anna_Kendrick",
                "generated": f"{GENERATED_IMAGES_BASE_FOLDER}/Anna_Kendrick",
                "output": f"{OUTPUT_BASE_FOLDER}/anna_kendrick",
                "name": "Anna Kendrick"
            },
            "bill_clinton": {  
                "original": f"{ORIGINAL_IMAGES_BASE_FOLDER}/Bill_Clinton",
                "generated": f"{GENERATED_IMAGES_BASE_FOLDER}/Bill_Clinton",
                "output": f"{OUTPUT_BASE_FOLDER}/bill_clinton",
                "name": "Bill Clinton"
            }
        }
    
    
    # --- Initialize LPIPS Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # We initialize with spatial=True to get the 2D difference map,
    # not the final averaged score.
    lpips_model = lpips.LPIPS(net='alex', spatial=True).to(device)
    lpips_model.eval() # Set to evaluation mode
    
    all_celebrity_results = {}

    # --- Process each celebrity ---
    for celeb_key, paths in celebrities.items():
        print(f"\n{'='*50}")
        print(f"Calculating LPIPS for: {paths['name']}")
        print(f"{'='*50}")
        
        output_folder = Path(paths["output"])
        original_folder = Path(paths["original"])
        generated_folder = Path(paths["generated"])
        
        summary_path = output_folder / "metadata" / "summary.json"
        
        if not summary_path.exists():
            print(f"Error: summary.json not found at {summary_path}")
            continue
            
        if not generated_folder.exists():
            print(f"Error: Generated images folder not found at {generated_folder}")
            continue

        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            
        celebrity_scores = []
        
        for item in tqdm(summary_data, desc=f"Processing {paths['name']}"):
            if item['status'] == 'success':
                
                original_img_path = original_folder / item['image']
                original_stem = original_img_path.stem
                
                mask_path = output_folder / item['mask_path']
                
                # Find the matching generated image
                gloce=False
                start_idx_name = 0
                if gloce:
                    start_idx_name = 2
                generated_img_path = find_generated_image(generated_folder, original_stem[start_idx_name:])
                
                if not generated_img_path:
                    print(f"  Warning: Skipping. Could not find generated image for {original_stem}")
                    continue
                
                # --- This is the core calculation ---
                score = compute_masked_lpips(
                    lpips_model,
                    original_img_path,
                    generated_img_path,
                    mask_path,
                    device
                )
                
                if score is not None:
                    celebrity_scores.append(score)
        
        # Calculate and print mean for this celebrity
        if celebrity_scores:
            mean_score = np.mean(celebrity_scores)
            all_celebrity_results[paths['name']] = mean_score
            print(f"\n--- Results for {paths['name']} ---")
            print(f"  Images processed: {len(celebrity_scores)}")
            print(f"  Mean LPIPS (Background Only): {mean_score:.4f}")
        else:
            print(f"\n--- No successful images found for {paths['name']} ---")

    # --- Final Summary ---
    print(f"\n\n{'='*50}")
    print("Final Mean LPIPS (Background Only)")
    print(f"{'='*50}")
    for name, score in all_celebrity_results.items():
        print(f"  {name}: {score:.4f}")

if __name__ == "__main__":
    main_lpips()