import os, sys, re, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class Generate_Dataset(Dataset):
    def __init__(self, path, content, sub_root):
        super().__init__()
        root_path = os.path.join(path, content, sub_root)
        self.content = content
        self.images = [os.path.join(root_path, name) for name in os.listdir(root_path)]
    
    def __len__(self,):
        return len(self.images)
    
    def __getitem__(self, idx):
        text = self.images[idx].split('/')[-1]
        text = ('_').join(text.split('_')[:-1])
        image = self.images[idx]
        return {'text': text, 'image': image, 'content': self.content}
    

class CLIP_Score():
    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda' if torch.cuda.is_available() else 'cpu'): # 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.device = device
        self.model = self.model.to(self.device)
    
    def __call__(self, dataloader):
        out_score = 0
        for item in dataloader:
            image = [Image.open(img) for img in item['image']]
            captions = item['text']
            out_score_matrix  = self.model_output(captions, image)
            out_score += out_score_matrix.mean().item() 
        return out_score/len(dataloader) #(out_score + intact_score).mean().item()# input_score_matrix,  #
    
    def model_output(self, text, img):
        torch.cuda.empty_cache()
        images_feats = self.processor(images=img, return_tensors="pt").to('cuda')
        images_feats = self.model.get_image_features(**images_feats)

        texts_feats = self.tokenizer(text, padding=True, truncation=True, max_length=77, return_tensors="pt",).to('cuda')
        texts_feats = self.model.get_text_features(**texts_feats)

        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
        score = (images_feats * texts_feats).sum(-1)
        return score


class Pytorch_FID():
    def __init__(self, batch_size, device, dims=2048, num_workers=8):
        self.batch_size = batch_size
        self.device = device
        self.dims = dims
        self.num_workers = num_workers
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)

    def __call__(self, root_path, pretrained_path, content, sub_root):
        paths = [os.path.join(root_path, content, sub_root), os.path.join(pretrained_path, content, 'original')]
        m1, s1 = fid_score.compute_statistics_of_path(paths[0], self.model, self.batch_size,
                                            self.dims, self.device, self.num_workers)
        m2, s2 = fid_score.compute_statistics_of_path(paths[1], self.model, self.batch_size,
                                            self.dims, self.device, self.num_workers)
        fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

def find_root_paths(root_dir, sub_root):
    return sorted(
        list({os.path.abspath(os.path.join(dirpath, '..')) 
                for dirpath, dirnames, _ in os.walk(root_dir) if sub_root in dirnames})
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contents', type=str)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--sub_root', type=str, default='retain')
    parser.add_argument('--pretrained_path', type=str)
    args = parser.parse_args()
    contents = [item.strip() for item in args.contents.split(',')]
    root_paths = find_root_paths(args.root_path, args.sub_root)

    CS_calculator = CLIP_Score(version='openai/clip-vit-large-patch14')
    FID_calculator = Pytorch_FID(batch_size=50, device='cuda')

    for root_path in root_paths:
        save_txt = os.path.join(root_path, 'record_metrics.txt')
        if os.path.exists(save_txt):
            pass
        else:
            with open(save_txt, 'a') as f:
                f.writelines('*************************** \n')
                f.writelines(f'Calculating the metrics for {root_path} \n')

        with open(save_txt, 'r') as f:  
            txt_content = f.read()
        for content in tqdm(contents):
            if content + ':' in txt_content:
                pass
            else:
                dataloader = DataLoader(Generate_Dataset(root_path, content, args.sub_root), batch_size=10)
                score = CS_calculator(dataloader)
                fid_value = FID_calculator(root_path, args.pretrained_path, content, args.sub_root)
                with open(save_txt, 'a') as f:
                    f.writelines(f'The {content}: FID is {fid_value}, CS is {score * 100} \n')