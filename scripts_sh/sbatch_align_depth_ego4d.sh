#!/bin/bash
#SBATCH --job-name=Align_EGO                   # Job name
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem-per-cpu=16gb                          # Job memory request
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                       # Requesting 2 GPUs
#SBATCH --constraint=rtx6k|rtx8k|a6000
#SBATCH --array=0-7                       # array of length audio files
#SBATCH --exclude=gnodec1,gnodec2,gnodec3
#SBATCH --output=slurm/job.%A_%a.out # tell it to store the output console text to a file called job.<assigned job number>.out
# -------------------------------

# load conda
source /users/${USER}/.bashrc
conda activate py3d

# process mad audio
cd /work/yashsb/OSNOM-Lang/scripts/

for k in 8b47ac19-7c4f-47d2-b5d0-755b524b66b2 9f5253af-acc3-40ca-b8bf-7b931f875bd7 bff3d583-ca3b-44b8-9740-3b34c5a8d7a9 36bff607-9c5d-43f4-97d2-38ad79e5220c; do python3 extract_aligned_depth_ego4d.py --vid ${k} --n_chunks 8 --chunk_id $SLURM_ARRAY_TASK_ID ; done