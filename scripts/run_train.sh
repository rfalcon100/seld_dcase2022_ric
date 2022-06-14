#!/usr/bin/env bash
#SBATCH --gres=gpu:1
###SBATCH --constraint=volta|pascal|ampere
#SBATCH --cpus-per-task=6
#SBATCH --mem 90G
##SBATCH --time 0-3:50:00
#SBATCH --time 0-10:00:00
##SBATCH --time 0-00:30:00
#SBATCH -o "slurm/train_%j.out"

module purge
module load anaconda #has cuda 10

source activate audio2022
pwd

# Read the task_id
#task_id=$1
#echo 'Setting task_id to = '
#echo $task_id

#Sanity check, is the anaconda environment loaded ?
hostname
python -c 'import librosa'
which python
echo "init done"

echo "Start of job"
srun python main.py
echo "End of job"

conda deactivate
