#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta|ampere
#SBATCH --cpus-per-task=6
#SBATCH --mem 90G
##SBATCH --time 0-3:50:00
#SBATCH --time 0-16:00:00
##SBATCH --time 0-00:30:00
#SBATCH -o "slurm/train_%j.out"

module purge
module load anaconda #has cuda 10

source activate audio2022
pwd

# Example:
# >> sbatch run_train.sh dcase22_plus_dcase22-sim_w-aug 0

# Read the exp_name and task ids
exp_n=$1
num_w=$2
echo 'Setting exp_n to = '
echo exp_n
echo 'Setting $num_w to = '
echo $num_w

#Sanity check, is the anaconda environment loaded ?
hostname
python -c 'import librosa'
which python
echo "init done"

echo "Start of job"
#srun python main.py --exp_name $exp_n --num_workers $num_w --job_id $SLURM_JOBID --mode "train" --dataset_chunk_size_seconds 2.55
srun python main.py --exp_name $exp_n --num_workers $num_w --job_id $SLURM_JOBID --mode "train" --dataset_chunk_size_seconds 6 --model "samplecnn" --model_features_transform "none" --model_augmentation
echo "End of job"

conda deactivate
