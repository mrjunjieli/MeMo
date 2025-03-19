#!/bin/bash
#SBATCH -J 91 #Slurm job name

# Choose partition (queue) "gpu" or "gpu-share"
#SBATCH --partition=speech

#SBATCH --output=train.out
#SBATCH -N 1
#SBATCH --gres=gpu:2


# Go to the job submission directory and run your application
#module load cuda11.8/toolkit/11.8.0
#module load cuda12.1/toolkit/12.1.1
source activate py2.1

conda activate py2.1
# Execute applications in parallel


bash train.sh