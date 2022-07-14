#!/usr/bin/env bash
#SBATCH --gres=gpu:1
###SBATCH --constraint=volta|pascal|ampere
#SBATCH --constraint=volta|ampere
#SBATCH --cpus-per-task=6
#SBATCH --mem 80G
##SBATCH --time 0-0:30:00
#SBATCH --time 0-13:00:00
#SBATCH -o "slurm/train_%A_%a.out"
#SBATCH --array=0-10

# April01, table 01
case $SLURM_ARRAY_TASK_ID in
   0)  param=1 ;;
   1)  param=2 ;;
   2)  param=3 ;;
   3)  param=4 ;;
   4)  param=5 ;;
   5)  param=6 ;;
   6)  param=7 ;;
   7)  param=8 ;;
   8)  param=9 ;;
   9)  param=10 ;;
   10)  param=11 ;;
   11)  param=12 ;;
esac

# example
# >> sbatch scripts/run_train_array.sh training-monday 1111
# or
# >> sbatch run_train_array.sh 'table01_2022' 1234
# Where the params are:
# exp_group
# seed

# Useful variables when working with array jobs:
# https://slurm.schedmd.com/job_array.html
# $SLURM_JOB_ID
# $SLURM_ARRAY_JOB_ID      = %A  ( i think)
# $SLURM_ARRAY_TASK_ID     = %a  ( i think)

# Read the cuda devices
cuda_device=0

echo 'Setting CUDA_VISIBLE_DEVICES to = '
echo $cuda_device


module purge
module load anaconda
source activate audio2022

pwd
echo Start job

# Copy data to local drive for max speed
#mkdir /tmp/$SLURM_JOB_ID                          # get a directory where you will put your data
#cp $WRKDIR/input.tar /tmp/$SLURM_JOB_ID           # copy tarred input files
#cd /tmp/$SLURM_JOB_ID

#trap "rm -rf /tmp/$SLURM_JOB_ID; exit" TERM EXIT  # set the trap: when killed or exits abnormally you clean up your stuff

#tar xf input.tar                                  # untar the files
#srun  input/*                                     # do the analysis, or what ever else
#tar cf output.tar output/*                        # tar output
#mv output.tar $WRKDIR/SOMEDIR

#srun script4experiment/seld_train_"$1".sh 0
#srun script4experiment/seld_train_triton_"$1".sh 0 $param
#srun script4experiment/seld_train_triton_"$1".sh 0 $param $2   #here we pass the random seed as parameter
srun scripts/experiments.sh 0 $param $1 $2 $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID   # args are: gpu_id, params_id, exp_group, random_seed, job_id, job_sub_id

echo End of job
conda deactivate

#seff %j

