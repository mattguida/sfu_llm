#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=gpgpudeeplearn
#SBATCH --ntasks=1
#SBATCH --job-name="topic_mock"
#SBATCH --cpus-per-task=4
#SBATCH -A punim0478
#SBATCH --mem=128G
#SBATCH -o "output/zero/out/topic_first_mock.out" #STDOUT
#SBATCH -e "output/zero/err/topic_first_mock.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

python3 /data/gpfs/projects/punim0478/guida/topic_stance_inferences/code/topic_inference.py --dtype=bfloat16

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

