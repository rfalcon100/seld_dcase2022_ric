# Datapaths

### From VRGPU


````bash
/m/triton/scratch/work/falconr1/sony/data_dcase2022
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
/m/triton/scratch/work/falconr1/sony/data_dcase2022_sim
````


### From Triton

````bash
/scratch/work/falconr1/sony/data_dcase2022
/scratch/work/falconr1/sony/data_dcase2021_task3
/scratch/work/falconr1/sony/data_dcase2022_sim
````


# List for files:
Command to find the fnames of all wavs.

```bash
find ./data_dcase2022_sim/foa -type f > data_dcase2022_sim/list_dataset/dcase2022_sim_all.txt
find ./data_dcase2022_sim/foa/dev-train -type f > data_dcase2022_sim/list_dataset/dcase2022_sim_all.txt
find ./data_dcase2022/foa_dev/*train* -type f > data_dcase2022/list_dataset/dcase2022_devtrain_all.txt
find ./data_dcase2022/foa_dev/*test* -type f > data_dcase2022/list_dataset/dcase2022_devtest_all.txt
```


## DCASE 2022
```
dcase2022_devtest_all.txt  
dcase2022_devtrain_all.txt
dcase2022_devtrain_debug.txt
dcase2022_eval_all.txt
dcase2022_sim_debug.txt
dcase2021t3_foa_overfit.txt
```


## Debugger Fix for the custom Infinitedataloader

Set the:
- Build. Execution, Deployment > Python Debugger
- Geven compatible --> checked

Also (maybe):
- Set the variables loading policy --> Synchronously

https://codehunter.cc/a/python-3.x/debugger-times-out-at-collecting-data



What I need for the main loop:

iteration id
model
criterion
dataloader_train
dataloader_valid
config
writer

timer
loss_train
loss_valid
metrics_valid


## DCASE re submission:
```bash
--model
samplecnn_gru
--dataset_chunk_size_seconds
6
--model_features_transform
none
--mode
valid
```


--model
crnn10
--dataset_chunk_size_seconds
2.55
--model_features_transform
stft_iv
--mode
valid


--model samplecnn_gru 
--dataset_chunk_size_seconds 6 
--model_features_transform none 
--mode train
--model_augmentation
--input_shape 4 144000


--model
crnn10
--dataset_chunk_size_seconds
2.55
--model_features_transform
stft_iv
--mode
train
--num_workers
0
--model_augmentation
--wandb



## Params when using the new config files_

-c
./configs/run_debug.yaml
--wandb
--num_iters
500
--model
samplecnn
--dataset_chunk_size_seconds
6
--model_features_transform
none
--mode
train
--input_shape
4
144000


## PRofiling WANDB

-c
./configs/profile_train_default.yaml
--dataset_trim_wavs
15
--exp_group
profile
--model
samplecnn
--dataset_chunk_size_seconds
6
--model_features_transform
none
--mode
train
--input_shape
4
144000


-c
./configs/run_debug.yaml
--num_iters
10000
--logging_interval
1000
--dataset_trim_wavs
-1
--model
samplecnn
--dataset_chunk_size_seconds
1.27
--model_features_transform
stft_iv
--model_rotations
--model_augmentation
--mode
train
--input_shape
4
144000



### For validaiton
```bash
-c
./configs/run_debug.yaml
--print_every
200
--num_iters
10000
--logging_interval
5000
--dataset_trim_wavs
-1
--model
crnn10
--dataset_chunk_size_seconds
2.55
--model_features_transform
stft_iv
--input_shape
7
257
256
--mode
valid
--evaluation_overlap_fraction
1
--model_spatialmixup
--model_augmentation
--model_spec_augmentation
--model_rotations
--dataset_root_valid
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
--dataset_list_valid
dcase2021t3_foa_devtest.txt
```


### Debugging ###
```bash
-c
./configs/run_debug.yaml
--print_every
500
--num_iters
10000
--logging_interval
1000
--dataset_trim_wavs
-1
--dataset_backend
baseline
--model
crnn10
--dataset_chunk_size_seconds
5.11
--model_features_transform
mel_iv
--mode
train
--dataset_root
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
--dataset_list_train
dcase2021t3_foa_devtrain.txt
--dataset_root_valid
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
--dataset_list_valid
dcase2021t3_foa_devtest.txt
--evaluation_overlap_fraction
1
--curriculum_scheduler
linear
--wandb
--oracle_mode
foa_devtest.txt
```

### DEbugging DCASE 2021 ###
```bash
-c
./configs/run_debug.yaml
--batch_size
2
--print_every
500
--num_iters
10000
--logging_interval
1000
--dataset_trim_wavs
20
--dataset_backend
baseline
--model
crnn10
--dataset_chunk_size_seconds
2.55
--model_features_transform
mel_iv
--mode
train
--dataset_root
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
--dataset_list_train
dcase2021t3_foa_overfit.txt
--dataset_root_valid
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
--dataset_list_valid
dcase2021t3_foa_overfit.txt
--evaluation_overlap_fraction
1
--curriculum_scheduler
linear
```


## Debugging NaNs in disc loss, so we run a full training ##
```bash
-c
./configs/run_debug.yaml
--batch_size
32
--print_every
500
--num_iters
100000
--logging_interval
10000
--dataset_trim_wavs
-1
--dataset_backend
sony
--model
crnn10
--dataset_chunk_size_seconds
2.55
--model_features_transform
stft_iv
--mode
train
--wandb
--solver
DAN
--curriculum_scheduler
fixed
--D_lr
0.1
--D_lr_min
1e-10
--D_lr_scheduler
lrstep
--w_rec
100
--w_adv
0.0
--curriculum_w_adv
0.0
--curriculum_scheduler
fixed
--G_crit
ls
--D_crit
ls
--dataset_root
/m/triton/work/falconr1/sony/data_dcase2022
/m/triton/work/falconr1/sony/data_dcase2022_sim
--dataset_list_train
dcase2022_devtrain_all.txt
dcase2022_sim_all.txt
--dataset_root_valid
/m/triton/work/falconr1/sony/data_dcase2022
--dataset_list_valid
dcase2022_devtest_all.txt
```


## DEbugging discrminator, general debugging ##
```bash
-c
./configs/run_debug_DAN.yaml
--batch_size
2
--print_every
500
--num_iters
40000
--logging_interval
2000
--dataset_trim_wavs
-1
--dataset_backend
sony
--model
crnn10
--dataset_chunk_size_seconds
2.55
--model_features_transform
stft_iv
--mode
train
--dataset_root
/m/triton/work/falconr1/sony/data_dcase2021_task3
--dataset_list_train
dcase2021t3_foa_overfit.txt
--dataset_root_valid
/m/triton/work/falconr1/sony/data_dcase2021_task3
--dataset_list_valid
dcase2021t3_foa_overfit.txt
--evaluation_overlap_fraction
1
--curriculum_scheduler
linear
--D_lr_scheduler
lrstep
--w_adv
0.3
--D_lr_scheduler_step
100
--wandb
```


This are commands that I can call from the command line in vrgpu to do some manual profiling.
The idea is to monitor the step time, and look in my excel file for comparison.
```bash
python main.py -c ./configs/profile_train_default.yaml --dataset_trim_wavs 15 --exp_group profile --model samplecnn --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000
python main.py -c ./configs/profile_train_default.yaml --dataset_trim_wavs 15 --exp_group profile --model samplecnn --model_augmentation --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000
python main.py -c ./configs/profile_train_default.yaml --dataset_trim_wavs 15 --exp_group profile --model samplecnn_gru --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000
python main.py -c ./configs/profile_train_default.yaml --dataset_trim_wavs 15 --exp_group profile --model samplecnn_gru --model_augmentation --dataset_chunk_size_seconds 6 --model_features_transform none --mode train --input_shape 4 144000
```
## Some useful git commands


```bash
git remote -v

git add files.py
gig commit -m "my commit"
git push gitohubo ---> commits to github 
git push  
or
git push origin  --> commits to gitlab 


cd torch_audiomentations 
git status
git add files.py
git commit -m "commit_message"
git push origin  ---> commits to torch_audiomentations repo (my fork in github)
OR
git pull   ---> pulls from my fork, I should merge in github from the original repo is needed
```
