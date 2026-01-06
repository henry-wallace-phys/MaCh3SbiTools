#!/bin/bash
#SBATCH --account=def-deborahh
#SBATCH --gpus=h100:1
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:0:0
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a_%N.out
#SBATCH --error=%x_%A_%a_%N.err
#SBATCH --job-name=mach3sbi

source /home/henryi/scratch/venvs/.venv_sbi/bin/activate
source /home/henryi/scratch/venvs/.venv_sbi/bin/setup.MaCh3.sh
source /home/henryi/scratch/venvs/.venv_sbi/bin/setup.MaCh3Tutorial.sh

# Define array of FIT_TYPE values
FIT_TYPES=("mechanistic_embedding" "truncated_proposal" "automatic_transform")

# Select FIT_TYPE based on array task ID
FIT_TYPE=${FIT_TYPES[$SLURM_ARRAY_TASK_ID]}

# Redirect output/error to files with FIT_TYPE name
exec > ${FIT_TYPE}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2> ${FIT_TYPE}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err

mkdir -p ${FIT_TYPE}_sbi_fit
OUTFILE=${FIT_TYPE}_sbi_fit/${FIT_TYPE}_posterior.pkl

mach3sbi --input_file /home/henryi/sft/MaCh3Tutorial/TutorialConfigs/FitterConfig.yaml \
        --output_file ${OUTFILE} \
        --mach3_type tutorial \
        --fit_type ${FIT_TYPE} \
        --autosave_interval 1 \
        --n_rounds 100 \
        --samples_per_round 10000 \
        --mach3_type tutorial
