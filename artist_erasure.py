import os
from pathlib import Path
import torch
from tqdm import tqdm
import io


OUTPUT_INDEX_FILE = Path("/LACE/dual_embedding_index_with_scores.pt")

class TqdmIO:
    """
    A file-like object wrapper to provide a smooth tqdm progress bar for torch.load.
    It forces large read requests into smaller chunks to ensure the progress bar updates continuously.

    NOTE: This will be slower than a direct file read.
    """
    def __init__(self, file_path):
        print("Starting by parsing object structure before reading...")
        self.file_obj = open(file_path, 'rb')
        self.file_size = os.path.getsize(file_path)
        self.pbar = tqdm(
            total=self.file_size,
            unit='B',
            unit_scale=True,
            desc=f"Loading {OUTPUT_INDEX_FILE.name}",
            mininterval=0.1 # Refresh bar at least every 0.1s
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()
        self.file_obj.close()

    def read(self, n=-1):
        # If n=-1, it's a request to read the rest of the file.
        if n == -1:
            n = self.file_size - self.file_obj.tell()

        # Define a fixed-size buffer for our reads
        buffer_size = 1024 * 1024  # 1MB

        chunks = []
        bytes_read = 0
        while bytes_read < n:
            to_read = min(buffer_size, n - bytes_read)
            chunk = self.file_obj.read(to_read)
            if not chunk:
                break

            chunks.append(chunk)
            bytes_read += len(chunk)
            self.pbar.update(len(chunk))

        return b"".join(chunks)

    def readinto(self, b):
        # Create a memoryview of the buffer 'b' to write into it directly
        view = memoryview(b)
        total_bytes_read = 0

        # Define the size of our own smaller read chunks
        internal_buffer_size = 1024 * 1024 # 1MB

        while total_bytes_read < len(b):
            # Determine how much to read in the next chunk
            to_read = min(internal_buffer_size, len(b) - total_bytes_read)
            chunk = self.file_obj.read(to_read)
            if not chunk:
                break # End of file

            num_bytes_in_chunk = len(chunk)
            # Place the read chunk into the correct slice of the buffer
            view[total_bytes_read : total_bytes_read + num_bytes_in_chunk] = chunk
            total_bytes_read += num_bytes_in_chunk
            self.pbar.update(num_bytes_in_chunk)

        return total_bytes_read

    def seek(self, *args, **kwargs):
        return self.file_obj.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.file_obj.tell(*args, **kwargs)

with TqdmIO(OUTPUT_INDEX_FILE) as f:
    data = torch.load(f, map_location="cpu")

import torch
import clip
from pathlib import Path
import re
from tqdm import tqdm
import Levenshtein
import math
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline

QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

class CLIPIndexSearcher:
    """
    A class to load the CLIP model and index data once and perform multiple queries.
    """
    def __init__(self, data: Path, device: str = "cuda"):
        self.device = device

        print(f"Using device: {self.device}")

        stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        stable_diffusion_pipeline = stable_diffusion_pipeline.to(self.device)
        self.stable_diffusion_pipeline = stable_diffusion_pipeline


        # Load the CLIP model once
        print("Loading CLIP model 'ViT-B/32'...")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        print(f"Loading Sentence Transformer '{QWEN_MODEL_NAME}'...")
        self.qwen_model = SentenceTransformer(QWEN_MODEL_NAME, device=self.device)

        self.data = data
        # Load the large index file once
        print(f"Processing data...")
        self.clip_embeddings = data['clip_embeddings'].to(self.device)
        self.qwen_embeddings = data['qwen_embeddings'].to(self.device)
        self.all_concepts = data['concepts']
        self.all_concreteness = data['conc_scores']
        self.all_popularity = data['pop_scores']
        print("Model and data loaded successfully.")


    def get_mean_image_embedding_for_concept(self, concept: str, n_images: int) -> torch.Tensor:
        """
        Generates N images for a concept, encodes them with CLIP,
        and returns the normalized mean embedding.
        """
        with torch.no_grad():
            # Generate N images
            prompt = f"a photo of a {concept}"
            images = self.stable_diffusion_pipeline(
                prompt=prompt,
                num_inference_steps=50,
                num_images_per_prompt=n_images
            ).images

            # Preprocess all generated images for CLIP
            preprocessed_images = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)

            # Encode images and normalize the embeddings
            image_embeddings = self.clip_model.encode_image(preprocessed_images)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            # Calculate the mean embedding
            mean_embedding = image_embeddings.mean(dim=0)

            # Normalize the final mean embedding
            mean_embedding /= mean_embedding.norm()

            return mean_embedding

    def query(self, target_concept, top_n, min_concreteness=0.0, lexical_threshold=0.8, pop_log_thresh=1.0, min_popularity=10, disallowed_words=None, n_images_per_concept=0):
        """
        Queries the index, retrieving and re-ranking candidates,
        and displays all relevant scores for the final results.
        """
        if disallowed_words is None:
            disallowed_words = []

        num_candidates = top_n * 100

        with torch.no_grad(), torch.autocast(device_type="cuda"):
            # Qwen encoding (for retrieval)
            target_qwen_embedding = self.qwen_model.encode(
                target_concept, convert_to_tensor=True, device=self.device
            )
            target_qwen_embedding = target_qwen_embedding / target_qwen_embedding.norm(dim=-1, keepdim=True)

            # CLIP encoding (for re-ranking)
            tokenized_target_clip = clip.tokenize([target_concept]).to(self.device)
            target_clip_embedding = self.clip_model.encode_text(tokenized_target_clip)
            target_clip_embedding = target_clip_embedding / target_clip_embedding.norm(dim=-1, keepdim=True)

            # --- Step 1: Qwen Retrieval ---
            qwen_sim = torch.mm(target_qwen_embedding.unsqueeze(0), self.qwen_embeddings.T).squeeze(0)
            candidate_results = torch.topk(qwen_sim, k=min(num_candidates, len(self.all_concepts)))
            candidate_indices = candidate_results.indices
            qwen_initial_scores = candidate_results.values

            # --- Step 2: CLIP Re-ranking ---
            candidate_clip_embeddings = self.clip_embeddings[candidate_indices]
            clip_rerank_scores = torch.mm(target_clip_embedding, candidate_clip_embeddings.T).squeeze(0)

        # --- Step 3: Build a Comprehensive List of Candidates ---
        # Create a list of dictionaries, each containing ALL relevant scores from the start
        all_candidate_data = []
        for i in range(len(candidate_indices)):
            original_index = candidate_indices[i].item()
            all_candidate_data.append({
                "clip_score": clip_rerank_scores[i].item(),
                "qwen_score": qwen_initial_scores[i].item(),
                "concept": self.all_concepts[original_index],
                "popularity": self.all_popularity[original_index].item(),
                "concreteness": self.all_concreteness[original_index].item()
            })

        # --- Step 4: Filter the Comprehensive List ---
        filtered_results = []
        for res in all_candidate_data:
            # Filter by popularity
            # if not are_log_similar(target_popularity, res['popularity'], pop_log_thresh):
            #     continue
            if res['popularity'] <= min_popularity:
                continue

            # Filter by disallowed words
            if any(d.lower() in res['concept'].lower() for d in disallowed_words):
                continue

            # Filter by concreteness
            if res['concreteness'] < min_concreteness:
                continue

            filtered_results.append(res)


        # --- Step 5: Optional Visual Filtering ---
        if n_images_per_concept > 0 and len(filtered_results) > 0:
            filtered_results = filtered_results[:top_n * 10]

            target_embedding = self.get_mean_image_embedding_for_concept(target_concept, n_images_per_concept)

            for candidate in tqdm(filtered_results, desc="Visual filtering"):
                print(f"Processing '{candidate['concept']}'...")
                cand_emb = self.get_mean_image_embedding_for_concept(candidate['concept'], n_images_per_concept)
                similarity = torch.nn.functional.cosine_similarity(target_embedding, cand_emb, dim=0)
                candidate['visual_score'] = similarity.item()
                print(f"Visual score: {similarity.item():.4f}")

            sort_key = 'visual_score'
        else:
            sort_key = 'clip_score'

        # --- Step 6: Sort and Return ---
        filtered_results.sort(key=lambda x: x[sort_key], reverse=True)
        final_top_n = filtered_results[:top_n]

        print(f"\nTop {len(final_top_n)} results for '{target_concept}':")
        for i, result in enumerate(final_top_n):
            print(
                f"{i+1:2d}. \"{result['concept']}\"\n"
                f"    {sort_key.capitalize().replace('_', ' ')}: {result[sort_key]:.4f} | "
                f"CLIP Score: {result['clip_score']:.4f} | "
                f"Qwen Score: {result['qwen_score']:.4f} | "
                f"Popularity: {result['popularity']:.0f} | "
                f"Concreteness: {result['concreteness']:.2f}"
            )

        return final_top_n

# These helper functions don't depend on the class state, so they can stay outside
def calculate_lexical_similarity(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 1.0
    return 1.0 - (distance / max_len)

def are_log_similar(views_a, views_b, log_threshold=1.0):
    log_a = math.log10(max(1, views_a))
    log_b = math.log10(max(1, views_b))
    return abs(log_a - log_b) <= log_threshold

# OUTPUT_INDEX_FILE = Path("clip_index_with_scores.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize the searcher ONCE. This will load the 19GB file into VRAM.
searcher = CLIPIndexSearcher(data=data, device=DEVICE)


import json
from tqdm import tqdm
import pandas as pd

# --- 1. SETUP ---

def load_targets_and_neighbors(filename):
    # Load the CSV
    df = pd.read_csv(filename)

    # Define all the targets you want to process
    erased_species = df[df["type"] == "erased"]["species"].unique().tolist()
    print("Erased species:", erased_species)

    # The number of neighbors to find for each target
    N_NEIGHBORS = 10

    # --- 2. MAIN LOGIC TO FIND DISJOINT NEIGHBORHOODS ---

    # This dictionary will store the final results
    all_neighbors = {}

    # This set will keep track of concepts that have been assigned to a neighborhood
    # to ensure there is no overlap between the final sets.
    used_concepts = set()

    # This set will accumulate all disallowed words from the synonyms of targets
    # as we process them.
    cumulative_disallowed_words = set()

    # Loop through each target and find its unique neighbors
    for target in tqdm(erased_species, desc="Finding neighbors for all targets"):
        print(f"\n{'='*20}\nProcessing target: {target}\n{'='*20}")

        # Step A: Update the disallowed words with the current target's name
        current_disallowed = set()
        current_disallowed.update(target.replace('_', ' ').split()) # Add 'golden', 'retriever'
        cumulative_disallowed_words.update(current_disallowed)

        # Step B: Combine all disallowed words and already used concepts for the query
        # This is the master exclusion list for this specific query.
        combined_disallowed_list = list(cumulative_disallowed_words | used_concepts)

        # Step C: Query for concepts, requesting more than needed to account for filtering
        query_results = searcher.query(
            target_concept=target.replace('_', ' '),
            top_n=N_NEIGHBORS,  # Ask for a large pool of candidates
            lexical_threshold=0.2,
            min_concreteness=4.0,
            pop_log_thresh=2.0,
            min_popularity=100,
            disallowed_words=combined_disallowed_list,
            n_images_per_concept=10
        )

        # Step D: Extract the top N valid neighbors and update the state
        target_neighbors = [c["concept"] for c in query_results][:N_NEIGHBORS]
        all_neighbors[target] = target_neighbors

        # Add the newly found neighbors to the set of used concepts for the next iteration
        used_concepts.update(target_neighbors)

        print(f"Found {len(target_neighbors)} neighbors for '{target}'.")
        print(f"Total used concepts so far: {len(used_concepts)}")


    # --- 3. DISPLAY AND SAVE FINAL RESULTS ---

    print("\n\n" + "="*50)
    print("          FINAL NEIGHBORHOOD CONCEPTS")
    print("="*50 + "\n")

    for target, neighbors in all_neighbors.items():
        print(f"--- Target: {target} ---")
        if not neighbors:
            print("  No neighbors found.")
        for i, neighbor in enumerate(neighbors):
            print(f"  {i+1:2d}. {neighbor}")
        print("\n")

    print("\n\n" + "="*50)
    print("          ALL CONCEPTS IN A SINGLE SET")
    print("="*50 + "\n")

    # Use a nested set comprehension to gather all neighbors into one set.
    # This iterates through each list of neighbors and then through each neighbor in that list.
    all_concepts_set = {
        neighbor
        for neighbor_list in all_neighbors.values()
        for neighbor in neighbor_list
    }

    # Print the total count of unique concepts and the final set
    print(f"Found {len(all_concepts_set)} unique neighborhood concepts in total.\n")
    print(all_concepts_set)

    return erased_species, list(all_concepts_set)

"""## Stage 1 (Text Embedding Using Spectral Expansion)"""

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
from diffusers import StableDiffusionPipeline

model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None
)
model = model.to("cuda")

tokenizer = model.tokenizer
text_encoder = model.text_encoder

device = "cuda"


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

    # Flatten attention mask and last_hidden across batch and sequence length
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



def calculate_projection_matrix(target_word, manual_neighbors=None, beta=1.0, gamma=1.0):
    e_targets = encode_text(target_word).float().cpu()   # [n, d]
    print(e_targets.shape)

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
        spectral_w = spectral_expansion_weights(S_t, 2)
        Vh_t = Vh_t[:64]
        P_F = Vh_t.T @ Vh_t
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
    spectral_components = min(77 * len(neighbor_tokens), 16)
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

    # P_c = (I - P_F.to(device)) + P_R.to(device)
    P_c = (I - P_F.to(device)) + torch.matmul(P_F.to(device), P_R.to(device))

    # print("Neighbor list:", neighbor_tokens)
    P_c = P_c.to(device).to(torch.float16)
    print("Projection matrix P_c shape:", P_c.shape)

    return P_c, P_F


import copy
from diffusers.models.attention_processor import Attention

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

import math

MASK_THR = 0.025

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
            if encoder_hidden_states is None or (t > 0.9 * T):
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
                    mask_dev = mask.to(device=dev, dtype=dtype).unsqueeze(-1)
                    out = (1. - mask_dev) * h
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

# session_info.show(excludes=['pybind11_abseil'])
# # session_info.show(write_req_file=True, req_file_name='requirements.txt')



from tqdm import tqdm
import glob
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
import os
import matplotlib.pyplot as plt
from PIL import Image
import json

import torch.nn.functional as F
from sklearn.decomposition import PCA

beta = 1.0
gamma = 1.0

folder_path = '/artists/'
config_files = glob.glob(os.path.join(folder_path, '*.csv'))

for config_file in config_files:
    print(f"Processing config: {config_file}")
    df = pd.read_csv(config_file)

    path_without_ext = os.path.splitext(config_file)[0]
    # Now get just the final component (the base name)
    name_only = os.path.basename(path_without_ext)
    print(f"Base name without extension: {name_only}")

    # Set up output folder
    base_path = os.path.join(folder_path, name_only)
    ours_folder = os.path.join(base_path, f"ours_b{beta}_g{gamma}")
    cure_folder = os.path.join(base_path, "cure")
    orig_folder = os.path.join(base_path, "orig")
    comparison_folder = os.path.join(base_path, "comparison")

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(ours_folder, exist_ok=True)
    os.makedirs(cure_folder, exist_ok=True)
    os.makedirs(orig_folder, exist_ok=True)
    os.makedirs(comparison_folder, exist_ok=True)

    # setup model
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    model = model.to("cuda")
    model_copy = copy.deepcopy(model)

    model.set_progress_bar_config(disable=True)
    model_copy.set_progress_bar_config(disable=True)


    orig_forward = model.unet.forward
    state = ProcessorState(
        using_prompt_tokens = True,
        stage_3 = True,
        attention_supression = True,
        apply_projection_on_weights = True,
        decay_threshold = True,
        hard_soft_scrub = True
    )

    model.unet.forward = make_double_forward(orig_forward, state).__get__(model.unet, type(model.unet))

    target_word_file = os.path.join(base_path, f"target_word.json")
    manual_neighbors_file = os.path.join(base_path, f"manual_neighbors.json")

    if not os.path.exists(target_word_file) or not os.path.exists(manual_neighbors_file):
        print(f"Generating target word and manual neighbors for {name_only}")
        if not os.path.exists(target_word_file):
            print(f"Target word file not found for {target_word_file}")
        if not os.path.exists(manual_neighbors_file):
            print(f"Manual neighbors file not found for {manual_neighbors_file}")

        target_word, manual_neighbors = load_targets_and_neighbors(config_file)
    else:
        print(f"Loading target word and manual neighbors from files for {name_only}")
        with open(target_word_file, "r") as f:
            target_word = json.load(f)
            print(target_word)

        with open(manual_neighbors_file, "r") as f:
            manual_neighbors = json.load(f)
            print(manual_neighbors)

    # Save target word and manual neighbors to file as JSON
    with open(target_word_file, "w") as f:
        json.dump(target_word, f)

    with open(manual_neighbors_file, "w") as f:
        json.dump(manual_neighbors, f)


    P_c, P_F = calculate_projection_matrix(target_word, manual_neighbors, beta, gamma)


    # patch all attention layers in the UNet
    for name, module in model.unet.named_modules():
        if isinstance(module, Attention):
            proc = module.get_processor()
            module.set_processor(make_wrapped_processor(proc, name, state, P_c, None))


    # Start running generation
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt = row["prompt"]
        species = row["species"]
        seed = int(row["evaluation_seed"])
        concept_type = row["type"]

        safe_species = species.replace(" ", "_").replace("/", "_").replace("-", "_")

        i=0
        filename_orig = f"{orig_folder}/orig_{safe_species}_{idx:04d}_seed{seed}_img{i}.png"
        filename_ours = f"{ours_folder}/ours_{safe_species}_{idx:04d}_seed{seed}_{concept_type}_img{i}.png"
        filename_cure = f"{cure_folder}/cure_{safe_species}_{idx:04d}_seed{seed}_{concept_type}_img{i}.png"
        comparison_filename = f"{comparison_folder}/comparison_{safe_species}_{idx:04d}_seed{seed}_{concept_type}_img{i}.png"

        # if all(os.path.exists(p) for p in [filename_orig, filename_ours, filename_cure, comparison_filename]):
        if all(os.path.exists(p) for p in [filename_ours]):
            print(f"Skipping {safe_species} (index {idx}) — images already exist.")
            continue


        attention_mask_for_pad = model.tokenizer(prompt, return_tensors="pt", max_length=77, padding="max_length", truncation=True).attention_mask.to(device)
        num_tokens = attention_mask_for_pad.sum(dim=1).item()
        s_j = (model.encode_prompt(prompt, device=device, do_classifier_free_guidance=False, num_images_per_prompt=1)[0].float() @ P_F.to(model.device)).norm(dim=-1)
        s_j_valid = s_j[:,1:num_tokens-1]

        delta = max(0.6 * s_j_valid.max(), 20)
        indices = (s_j_valid.squeeze(0) > delta).nonzero(as_tuple=True)[0].cpu() + 1

        state.is_second_pass = False
        state.saved_masks = {}
        generator = torch.Generator(device).manual_seed(seed)
        new_img = model(prompt=prompt, generator=generator)


        generator = torch.Generator(device).manual_seed(seed)
        orig_img = model_copy(prompt=prompt, generator=generator)

        orig_img[0][i].save(filename_orig)
        new_img[0][i].save(filename_ours)
