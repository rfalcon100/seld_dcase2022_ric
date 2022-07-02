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



## Some useful git commands


```bash
git remote -v

git add files.py
gig commit -m "my commit"
git push gitohubo ---> commits to github 
git push  
or
git push origin  --> commits to gitlab 
```
