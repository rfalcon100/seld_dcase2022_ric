#!/usr/bin/env bash
##SBATCH -p short
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu=3000
#SBATCH -o slurm/results-analysis-job-%A_%a.out
#SBATCH --array=0-30


# example
# >> sbatch scripts/run_evaluation_tables.sh 2021 table01 100
# or
# >> sbatch scripts/run_evaluation_tables.sh 2022 table01_2022 100
# or custom parameters:
# >> sbatch  --array 0-7 -t 00:55:00 scripts/run_evaluation_tables.sh 2022 table01_dcase2022 100
#
# Where the params are:
# dcase dataset (2021 or 2022)
# suffix for list_of_runs_XXXXXX.txt
# n_trails

# Useful variables when working with array jobs:
# https://slurm.schedmd.com/job_array.html
# $SLURM_JOB_ID
# $SLURM_ARRAY_JOB_ID      = %A  ( i think)
# $SLURM_ARRAY_TASK_ID     = %a  ( i think)

dcase=$1
fname=$2    # read fname via terminal
n_trials=$3

module purge
module load anaconda
source activate audio2022

pwd
echo Start job

# For DCASE2021:
if [ "$dcase" -eq "2021" ];
then
    echo Running Dcase2021 script
    srun python process_results_for_tables.py --mode 3 --filename $fname --n_trials $n_trials --chunk_id $SLURM_ARRAY_TASK_ID
fi

# For DCASE2022
if [ "$dcase" -eq "2022" ];
then
    echo Running Dcase2022 script
    srun python process_results_for_tables.py --mode 3 --filename $fname --dcase2022 --n_trials $n_trials --chunk_id $SLURM_ARRAY_TASK_ID
fi

# For DCASE2022
#srun python process_results_for_tables.py --mode 3 --filename $fname --dcase2022 --n_trials 100 --chunk_id $SLURM_ARRAY_TASK_ID

echo End of job
conda deactivate

#seff %j
