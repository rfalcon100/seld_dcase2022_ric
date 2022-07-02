#!/usr/bin/env bash

# This file should be called from an array job, so:
# sbatch run_train_array.sh 'table01_2022' 1234
# Where the first param is the suffix used, and the last is the seed

# Read the cuda devices
cuda_device=$1
param=$2
seed=$3
job_id=$4

hostname
echo 'Inside job. Setting CUDA_VISIBLE_DEVICES to = '
echo $cuda_device
echo 'Param = '
echo $param
echo 'Seed'
echo $seed
echo 'Job_id'
echo $job_id

case $param in
    1)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model samplecnn --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000 \
    --wandb --exp_name "profile__samplecnn_base" --seed $seed --job_id $job_id
    ;;
    2)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model samplecnn --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --model_augmentation --input_shape 4 144000 \
    --wandb --exp_name "profile__samplecnn_aug" --seed $seed --job_id $job_id
    ;;
    3)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000 \
    --wandb --exp_name "profile__samplecnn-gru_base" --seed $seed --job_id $job_id
    ;;
    4)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --model_augmentation --input_shape 4 144000 \
    --wandb --exp_name "profile__samplecnn-gru_aug" --seed $seed --job_id $job_id
    ;;
    5)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model crnn10 --dataset_chunk_size_seconds 1.27 --model_features_transform stft_iv --mode train \
    --wandb --exp_name "profile__crnn10-1.27_base" --seed $seed --job_id $job_id
    ;;
    6)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model crnn10 --dataset_chunk_size_seconds 1.27 --model_features_transform stft_iv --mode train --model_augmentation \
    --wandb --exp_name "profile__crnn10-1.27_aug" --seed $seed --job_id $job_id
    ;;
    7)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train \
    --wandb --exp_name "profile__crnn10-2.55_base" --seed $seed --job_id $job_id
    ;;
    8)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/profile_train_default.yaml  --logging_interval 500 --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train --model_augmentation \
    --wandb --exp_name "profile__crnn10-2.55_aug" --seed $seed --job_id $job_id
    ;;
esac




