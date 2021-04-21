#! /bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --partition cpu
#SBATCH --job-name=cpujob
#SBATCH --mem=100M
module load libs/ffmpeg/4.1
module load apps/ffmpeg/4.3
module load languages/anaconda3/2020.02-tflow-1.15
module load languages/anaconda3/2020.02-tflow-2.2.0
#module load libs/cudnn/10.1-cuda-10.0
module load CUDA
#module load libs/tensorflow/1.2

#cd $SLURM_SUBMIT_DIR
time python ./create_dataset.py
