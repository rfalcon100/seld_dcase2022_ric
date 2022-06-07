# Datapaths

### From VRGPU


````bash
/m/triton/scratch/work/falconr1/sony/data_dcase2022
/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3
/m/triton/scratch/work/falconr1/sony/data_dcase2022_sim
````


# List for files:
Command to find the fnames of all wavs.

```bash
find ./data_dcase2022_sim/foa -type f > data_dcase2022_sim/list_dataset/dcase2022_sim_all.txt
find ./data_dcase2022/foa_dev/*train* -type f > data_dcase2022/list_dataset/dcase2022_devtrain_all.txt
find ./data_dcase2022/foa_dev/*test* -type f > data_dcase2022/list_dataset/dcase2022_devtest_all.txt
```


## DCASE 2022
```
dcase2022_devtest_all.txt  
dcase2022_devtrain_all.txt
```


## Debugger Fix for the custom Infinitedataloader

Set the:
- Build. Execution, Deployment > Python Debugger
- Geven compatible --> checked

Also (maybe):
- Set the variables loading policy --> Synchronously

https://codehunter.cc/a/python-3.x/debugger-times-out-at-collecting-data
