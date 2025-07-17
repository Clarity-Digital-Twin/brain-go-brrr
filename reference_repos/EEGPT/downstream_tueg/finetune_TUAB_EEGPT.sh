#!/usr/bin/env bash
set -x  # print the commands

# export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes
export MASTER_PORT=$((12000 + $RANDOM % 20000))

# official train/test splits. valid numbers: 1, 2, 3
SPLIT=${SPLIT:-1}

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

# Please refer to `run_mae_pretraining.py` for the meaning of the following hyperreferences
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr="localhost" \
        run_class_finetuning_EEGPT_change.py \
        --output_dir ./checkpoints/finetune_tuab_eegpt/ \
        --log_dir ./log/finetune_tuab_eegpt \
        --model EEGPT \
        --finetune ../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
        --weight_decay 0.05 \
        --batch_size 100\
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 50 \
        --layer_decay 0.65 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUAB \
        --disable_qkv_bias \
        --seed 0