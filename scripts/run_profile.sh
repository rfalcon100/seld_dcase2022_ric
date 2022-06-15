#!/usr/bin/env bash
#SBATCH --gres=gpu:1
###SBATCH --constraint=volta|pascal|ampere
#SBATCH --cpus-per-task=6
#SBATCH --mem 90G
##SBATCH --time 0-3:50:00
##SBATCH --time 0-10:00:00
#SBATCH --time 0-00:30:00
#SBATCH -o "slurm/profile_%j.out"

module purge
module load anaconda #has cuda 10

source activate audio2022
pwd

# Read the task_id
num_w=$1
echo 'Setting num_w to = '
echo $num_w

#Sanity check, is the anaconda environment loaded ?
hostname
python -c 'import librosa'
which python
echo "init done"

echo "Start of job"
srun python main.py --exp_name profile_"$num_w" --num_workers $num_w --job_id %j
echo "End of job"

conda deactivate
