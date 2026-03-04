#!/bin/bash
# Hebrew Fine-tuning script for a single GPU

# conda activate voicecraft_linux
# export CUDA_VISIBLE_DEVICES=0
# export WORLD_SIZE=1

export DATASET_DIR="/kaggle/input/voicecraft-hebrew-fleurs/voicecraft_data"
export EXP_ROOT="/kaggle/working/exp"
export WORLD_SIZE=1 # בקאגל לרוב נשתמש ב-1 (אלא אם הגדרת Multi-GPU T4 x2)

# Paths - Update these to your actual paths!
dataset=hebrew_fleurs
mkdir -p ./logs/${dataset}

exp_root="./experiments"
exp_name="hebrew_v1_330M_kaggle" # Recommended to use 330M architecture for stability
dataset_dir="./voicecraft_data" # The folder containing 'phonemes', 'manifest', and Encodec codes
load_model_from="./pretrained_models/giga330M.pth"

mkdir -p ./logs/${dataset}

# Running with reduced parameters to fit a single GPU
torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:41977 --nproc_per_node=${WORLD_SIZE} \
./main.py \
--dataset_dir ${dataset_dir} \
--exp_dir "${exp_root}/${dataset}/${exp_name}" \
--dataset "gigaspeech" \
--manifest_name "manifest" \
--num_decoder_layers 8 \
--text_vocab_size 100 \
--text_pad_token 100 \
--num_steps 5000 \
--lr 0.00001 \
--batch_size 1 \
--gradient_accumulation_steps 8 \
--n_codebooks 4 \
--audio_vocab_size 2048 \
--n_special 4 \
--eos 2051 \
--num_workers 2 \
--val_every_n_steps 50 \
--print_every_n_steps 10 \
--d_model 2048 \
--nhead 16 \
--reduced_eog 1 \
--warmup_fraction 0.1 \
--early_stop_threshold 0.001 \
--early_stop_step 200 \
--max_mask_portion 0.5 \
--load_model_from ${load_model_from} 
# --num_decoder_layers 16 \
# --dataset $dataset \
# --lr 0.0005 \