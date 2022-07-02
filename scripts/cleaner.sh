#!/usr/bin/env bash
# This a simple script to delete stuff when a job fails, or is not needed anymore.
# example
# >> ./scripts/cleaner.sh 12344678

pwd

## Read input parameters
job_id=$1

echo Deleting log, slurm output, results where id =
echo $job_id

rm -r ./logging/*$job_id*
rm ./slurm/*$job_id*


