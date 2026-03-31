#!/bin/bash

declare -A target_concept_map
declare -A contents_map

# Define target concepts and contents
target_concept_map["instance"]="instance"
target_concept_map["style"]="style"
target_concept_map["celebrity"]="celebrity"

contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator"
contents_map["style"]="Van Gogh, Picasso, Monet, Andy Warhol, Caravaggio"
contents_map["celebrity"]="Bruce Lee, Marilyn Monroe, Melania Trump, Anne Hathaway, Tom Cruise"

# Define the array of GPU indices to be used
GPU_IDX=('0' '1' '2' '3' '4' '5' '6' '7') 
NUM_GPUS=${#GPU_IDX[@]} # Calculate the number of GPUs

# Initialize the GPU allocation index
gpu_idx=0

# Function: Submit a task to a specified GPU
run_task() {
  local erase_type=$1
  local gpu_id=$2

  echo "Running task for $erase_type on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python src/main.py \
    --erase_type "$erase_type" \
    --target_concept "$erase_type" \
    --contents "${contents_map[$erase_type]}" \
    --mode 'original' \
    --num_samples 10 --batch_size 10 \
    --save_root "data/pretrain" &  
}

# Iterate through all tasks
for erase_type in "instance" "style" "celebrity"; do
  run_task "$erase_type" ${GPU_IDX[$gpu_idx]}

  # Update the GPU index and cycle through GPUs
  gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

  # If all GPUs have been assigned, wait for all current processes to finish before continuing续
  if [ $gpu_idx -eq 0 ]; then
    wait # Wait for all background tasks to complete
  fi
done

# Wait for the last batch of tasks to complete
wait