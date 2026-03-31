#!/bin/bash

declare -A targets_map
declare -A contents_map

# Erase Mode Config
mode="retain"

# Erase Task Config
erase_types=("instance" "style" "celebrity")
targets_map["instance"]="Snoopy;Snoopy, Mickey;Snoopy, Mickey, Spongebob"
targets_map["style"]="Van Gogh;Picasso;Monet"
targets_map["celebrity"]="Bruce Lee;Marilyn Monroe;Melania Trump"

# GPU Config
GPU_IDX=('0' '1' '2' '3' '4' '5' '6' '7')
# ============================

contents_map["instance"]="Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator"
contents_map["style"]="Van Gogh, Picasso, Monet, Andy Warhol, Caravaggio"
contents_map["celebrity"]="Bruce Lee, Marilyn Monroe, Melania Trump, Anne Hathaway, Tom Cruise"

NUM_GPUS=${#GPU_IDX[@]}  # Calculate the number of GPUs
gpu_idx=0 # Initialize the GPU allocation inde

# Hyper Config
params="100 0.93 2.0"

read a b c <<< "$params"
save_root="logs/${a}_${b}_${c}"

# Function: Submit a task to a specific GPU
run_task() {
  local erase_type=$1
  local target=$2
  local content=$3
  local gpu_id=$4
  local a=$5
  local b=$6
  local c=$7
  local save_root=$8

  echo "Running task for $erase_type with $target on GPU $gpu_id with a=$a, b=$b, c=$c"

  CUDA_VISIBLE_DEVICES=$gpu_id python src/main.py \
    --erase_type "$erase_type" \
    --target_concept "$target" \
    --contents "$content" \
    --mode "$mode" \
    --num_samples 10 --batch_size 10 \
    --sigmoid_a "$a" --sigmoid_b "$b" --sigmoid_c "$c" \
    --save_root "$save_root" 
}

# Iterate through all task
for erase_type in "${erase_types[@]}"; do

  IFS=';' read -ra targets <<< "${targets_map[$erase_type]}"
  IFS=',' read -ra contents <<< "${contents_map[$erase_type]}"

  for target in "${targets[@]}"; do

    for content in "${contents[@]}"; do

      # Sample for the specified target and content on a single GPU
      run_task "$erase_type" "$target" "$content" "${GPU_IDX[$gpu_idx]}" "$a" "$b" "$c" "${save_root}/${erase_type}" &

      # Update the GPU index and cycle through GPUs
      gpu_idx=$((gpu_idx + 1))

      # Check if the GPU index exceeds NUM_GPUS, if so, wait for current tasks to finish
      if (( gpu_idx >= NUM_GPUS )); then
          wait
          gpu_idx=0  # Reset GPU index
      fi

    done

    wait # Wait for all contents under the current target to finish before proceeding with evaluation
    gpu_idx=0  # Reset GPU index

    CUDA_VISIBLE_DEVICES=${GPU_IDX[0]} python src/clip_score_cal.py \
        --contents "${contents_map[$erase_type]}" \
        --root_path "${save_root}/${erase_type}/$(echo "$target" | sed 's/, /_/g')" \
        --pretrained_path "data/pretrain/${erase_type}"

  done
done

# Wait for the last batch of tasks to complete
wait