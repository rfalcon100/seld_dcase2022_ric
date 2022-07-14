#!/usr/bin/env bash

# This file should be called from an array job, so:
# sbatch run_train_array.sh 'table01_2022' 1234
# Where the first param is the exp_group, and the last is the seed

# Read the cuda devices and other params
cuda_device=$1
param=$2
exp_group=$3
seed=$4
job_id=$5
job_sub_id=$6

hostname
echo 'Inside job. Setting CUDA_VISIBLE_DEVICES to = '
echo $cuda_device
echo 'Param = '
echo $param
echo 'exp_group = '
echo $exp_group
echo 'Seed'
echo $seed
echo 'Job_id'
echo $job_id
echo 'Job_sub_id'
echo $job_sub_id



case $param in
    1)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform bandpass --mode train --input_shape 4 144000 \
    --wandb --exp_name "samplecnn-gru_bandpass" --seed $seed --job_id $job_id --task_id $job_sub_id --lr 1e-3
    ;;
    2)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform bandpass --mode train --input_shape 4 144000 --model_spatialmixup --model_augmentation --model_rotations \
    --wandb --exp_name "samplecnn-gru_bandpass+spm+aug+rot" --seed $seed --job_id $job_id --task_id $job_sub_id --lr 1e-3
    ;;
    3)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform bandpass --mode train --input_shape 4 144000 --model_spatialmixup --model_augmentation --model_rotations --model_rotations_noise\
    --wandb --exp_name "samplecnn-gru_bandpass+spm+aug+rot+rotnoise" --seed $seed --job_id $job_id --task_id $job_sub_id --lr 1e-3
    ;;
    4)
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform bandpass --mode train --input_shape 4 144000 --model_spatialmixup --model_augmentation --model_rotations --use_mixup \
    --wandb --exp_name "samplecnn-gru_bandpass+spm+aug+rot+mixup" --seed $seed --job_id $job_id --task_id $job_sub_id --lr 1e-3
    ;;
esac

# These were from testing input features and augmentaiton for CRNN, I ma missing mixup here
#
#case $param in
#    1)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv" --seed $seed --job_id $job_id --task_id $job_sub_id
#    ;;
#    2)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train --model_spatialmixup --model_augmentation --model_rotations --model_rotations_mode "azi" \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv+spm+aug+rot_azi" --seed $seed --job_id $job_id --task_id $job_sub_id
#    ;;
#    3)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train --model_spatialmixup --model_augmentation --model_rotations --model_rotations_mode "ele" \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv+spm+aug+rot_ele" --seed $seed --job_id $job_id --task_id $job_sub_id
#    ;;
#    4)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train --model_spatialmixup --model_augmentation --model_rotations --model_rotations_mode "azi-ele" \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv+spm+aug+rot_aziele" --seed $seed --job_id $job_id --task_id $job_sub_id
#    ;;
#esac


# These were from testing input features and augmentaiton for CRNN, I ma missing mixup here
#
#case $param in
#    1)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train \
#    --wandb --exp_name "crnn10-2.55_base_stft_iv" --seed $seed --job_id $job_id
#    ;;
#    2)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train --model_spatialmixup --model_augmentation --model_rotations \
#    --wandb --exp_name "crnn10-2.55_base_stft_iv+spm+aug+rot" --seed $seed --job_id $job_id
#    ;;
#    3)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train --model_spatialmixup --model_augmentation --model_spec_augmentation --model_rotations \
#    --wandb --exp_name "crnn10-2.55_base_stft_iv+spm+aug+specaug+rot" --seed $seed --job_id $job_id
#    ;;
#    4)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv" --seed $seed --job_id $job_id
#    ;;
#    5)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train --model_spatialmixup --model_augmentation --model_rotations \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv+spm+aug+rot" --seed $seed --job_id $job_id
#    ;;
#    6)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform mel_iv --mode train --model_spatialmixup --model_augmentation --model_spec_augmentation --model_rotations \
#    --wandb --exp_name "crnn10-2.55_base_mel_iv+spm+aug+specaug+rot" --seed $seed --job_id $job_id
#    ;;
#esac



#
#case $param in
#    1)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model samplecnn --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000 \
#    --wandb --exp_name "rot-samplecnn_base" --seed $seed --job_id $job_id
#    ;;
#    2)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model samplecnn --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --model_augmentation --input_shape 4 144000 \
#    --wandb --exp_name "rot-samplecnn_aug" --seed $seed --job_id $job_id
#    ;;
#    3)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000 \
#    --wandb --exp_name "rot-samplecnn-gru_base" --seed $seed --job_id $job_id
#    ;;
#    4)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --model_augmentation --input_shape 4 144000 \
#    --wandb --exp_name "rot-samplecnn-gru_aug" --seed $seed --job_id $job_id
#    ;;
#    5)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model crnn10 --dataset_chunk_size_seconds 1.27 --model_features_transform stft_iv --mode train \
#    --wandb --exp_name "rot-crnn10-1.27_base" --seed $seed --job_id $job_id
#    ;;
#    6)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model crnn10 --dataset_chunk_size_seconds 1.27 --model_features_transform stft_iv --mode train --model_augmentation \
#    --wandb --exp_name "rot-crnn10-1.27_aug" --seed $seed --job_id $job_id
#    ;;
#    7)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train \
#    --wandb --exp_name "rot-crnn10-2.55_base" --seed $seed --job_id $job_id
#    ;;
#    8)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model crnn10 --dataset_chunk_size_seconds 2.55 --model_features_transform stft_iv --mode train --model_augmentation \
#    --wandb --exp_name "rot-crnn10-2.55_aug" --seed $seed --job_id $job_id
#    ;;
#    9)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model crnn10 --dataset_chunk_size_seconds 5.11 --model_features_transform stft_iv --mode train \
#    --wandb --exp_name "rot-crnn10-5.11_base" --seed $seed --job_id $job_id
#    ;;
#    10)
#    CUDA_VISIBLE_DEVICES=$cuda_device python main.py \
#    -c ./configs/run_train_dcase2021.yaml --exp_group $exp_group --model_rotations --model crnn10 --dataset_chunk_size_seconds 5.11 --model_features_transform stft_iv --mode train --model_augmentation \
#    --wandb --exp_name "rot-crnn10-5.11_aug" --seed $seed --job_id $job_id
#    ;;
#esac




