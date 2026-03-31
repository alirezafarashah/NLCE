"""
Celebrity Face Segmentation using SAM (Segment Anything Model)
GPU-enabled version for Quadro RTX 8000
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from dotenv import load_dotenv
from skimage import io
import sys

# --- Giphy Model Imports ---
sys.path.append('./giphy') 
from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.preprocessors.face_detection.face_detector import FaceDetector

# --- SAM Imports ---
from segment_anything import sam_model_registry, SamPredictor


class CelebrityFaceSegmenter:
    def __init__(self, 
                 sam_checkpoint_path, 
                 giphy_resources_path, 
                 model_type="vit_h", 
                 device="cuda"):
        """
        Initialize the segmenter with SAM and Giphy models.
        """
        load_dotenv('./giphy/examples/.env')

        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # --- 1. Load SAM model ---
        print("Loading SAM model...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("SAM model loaded.")

        # --- 2. Load Giphy Models ---
        print("Loading Giphy face models...")
        self.image_size = int(os.getenv('APP_FACE_SIZE', 224))
        use_cuda_giphy = os.getenv('APP_USE_CUDA') == "true"
        
        self.model_labels = Labels(resources_path=giphy_resources_path)
        
        self.face_detector = FaceDetector(
            giphy_resources_path,
            margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
            use_cuda=use_cuda_giphy
        )
        self.face_recognizer = FaceRecognizer(
            labels=self.model_labels,
            resources_path=giphy_resources_path,
            use_cuda=use_cuda_giphy,
            top_n=1
        )
        print("Giphy models loaded.")

    
    def detect_and_identify_target_face(self, image_path, target_celebrity_name):
        """
        Detects all faces, identifies the target celebrity, and returns
        their bounding box.
        """
        try:
            original_image = io.imread(image_path)
        except Exception as e:
            print(f"  Error reading image {image_path}: {e}")
            return None, None, []

        all_faces_data = self.face_detector.perform_single(original_image)
        
        if not all_faces_data:
            return None, original_image, []

        processed_faces = [preprocess_image(img, self.image_size) 
                           for img, _ in all_faces_data]
        all_boxes = [box for _, box in all_faces_data]
        
        predictions = self.face_recognizer.perform(processed_faces)

        for i, pred in enumerate(predictions):
            pred_name = str(pred[0][0][0]).split('_[')[0].replace('_', ' ')
            if pred_name.lower() == target_celebrity_name.lower():
                return all_boxes[i], original_image, all_boxes

        return None, original_image, all_boxes
    
    
    def segment_face(self, original_image_rgb, face_box):
        """
        Use SAM to segment the face region from the original image.
        """
        try:
            self.predictor.set_image(original_image_rgb)
            
            # Face detector returns [x1, y1, x2, y2, confidence]
            # SAM only needs [x1, y1, x2, y2]
            input_box = np.array(face_box).flatten()[:4]
            
            # SAM expects shape (1, 4) for box input
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            return masks[0], scores[0]
        except Exception as e:
            print(f"  [SAM Error]: {e}")
            print(f"  Box content: {face_box}, type: {type(face_box)}, shape: {np.array(face_box).shape}")
            import traceback
            traceback.print_exc()
            return None, 0
    

    def apply_mask_and_visualize(self, original_image, mask, target_box, all_boxes):
        """
        Creates a visualization image with the mask and bounding boxes.
        """
        vis_img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
        # Draw all detected boxes (blue)
        # Boxes have format [x1, y1, x2, y2, confidence]
        for box in all_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw target box (red)
        x1, y1, x2, y2 = map(int, target_box[:4])
        cv2.rectangle(vis_img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Create colored mask overlay (green)
        colored_mask = np.zeros_like(vis_img_bgr)
        colored_mask[mask] = [0, 255, 0]
        
        vis_img_bgr = cv2.addWeighted(vis_img_bgr, 0.7, colored_mask, 0.3, 0)
        
        return vis_img_bgr

    
    def process_folder(self, folder_path, target_celebrity_name, output_folder):
        """
        Process all images in a celebrity folder.
        """
        folder_path = Path(folder_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        masks_folder = output_folder / "masks"
        visualizations_folder = output_folder / "visualizations"
        metadata_folder = output_folder / "metadata"
        
        masks_folder.mkdir(exist_ok=True)
        visualizations_folder.mkdir(exist_ok=True)
        metadata_folder.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = []
        
        print(f"\nProcessing {len(image_files)} images for '{target_celebrity_name}'...")
        for img_path in tqdm(image_files):
            try:
                # 1. Detect and Identify Face
                target_face_box, original_image, all_boxes = self.detect_and_identify_target_face(
                    img_path, target_celebrity_name
                )
                
                if target_face_box is None:
                    print(f"  Target '{target_celebrity_name}' not found in {img_path.name}")
                    results.append({
                        'image': img_path.name,
                        'status': 'target_not_identified'
                    })
                    continue
                
                # 2. Segment Face
                mask, score = self.segment_face(original_image, target_face_box)
                if mask is None:
                    raise Exception("SAM segmentation failed.")
                
                # 3. Save mask
                mask_filename = masks_folder / f"{img_path.stem}_mask.png"
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_filename), mask_img)
                
                # 4. Create and save visualization
                vis_img = self.apply_mask_and_visualize(
                    original_image, mask, target_face_box, all_boxes
                )
                vis_filename = visualizations_folder / f"{img_path.stem}_vis.jpg"
                cv2.imwrite(str(vis_filename), vis_img)
                
                # 5. Save metadata
                x1, y1, x2, y2 = target_face_box[:4]
                metadata = {
                    'image': img_path.name,
                    'status': 'success',
                    'target_celebrity': target_celebrity_name,
                    'face_box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence_score_sam': float(score),
                    'mask_path': str(mask_filename.relative_to(output_folder)),
                    'visualization_path': str(vis_filename.relative_to(output_folder))
                }
                results.append(metadata)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                results.append({
                    'image': img_path.name,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save summary
        summary_path = metadata_folder / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {output_folder}")
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"Successfully processed: {success_count}/{len(image_files)} images")


def main():
    """
    Main execution function.
    """
    # --- Configuration ---
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h"
    
    GIPHY_RESOURCES = "./giphy/examples/resources"

    celebrities = {
        "elon_musk": {
            "folder": "./original_celebrity_images/Elon_Musk",
            "name": "Elon Musk" 
        },
        "anna_kendrick": {
            "folder": "./original_celebrity_images/Anna_Kendrick",
            "name": "Anna Kendrick"
        },
        "bill_clinton": {  
            "folder": "./original_celebrity_images/Bill_Clinton",
            "name": "Bill Clinton"
        }
    }
    
    # Initialize segmenter
    segmenter = CelebrityFaceSegmenter(
        sam_checkpoint_path=SAM_CHECKPOINT,
        giphy_resources_path=GIPHY_RESOURCES,
        model_type=MODEL_TYPE,
        device="cuda"  
    )
    
    # Process each celebrity folder
    for celeb_key, paths in celebrities.items():
        print(f"\n{'='*50}")
        print(f"Processing: {celeb_key} (Name: {paths['name']})")
        print(f"{'='*50}")
        
        output_folder = f"output/{celeb_key}"
        segmenter.process_folder(
            folder_path=paths["folder"],
            target_celebrity_name=paths["name"],
            output_folder=output_folder
        )

if __name__ == "__main__":
    main()