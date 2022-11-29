#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem 30G
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --job-name=Test
#SBATCH --output=/work/smfrrohk/Masterarbeit/Experimente/%x/Logs/run-%j.log
#SBATCH --mail-user=frederik.rohkraehmer@tu-dortmund.de
#SBATCH --mail-type=ALL
#-------------------------------------

module purge
module load nvidia/cuda/11.1.1

source /home/smfrrohk/anaconda3/bin/activate
conda init bash
conda activate PhysicsLearner

cd /work/smfrrohk/Masterarbeit/PhysicsLearner
srun python main.py --name $SLURM_JOB_NAME