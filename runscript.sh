#!/bin/bash -l
#SBATCH -C cgpu01
#SBATCH -c 20
#SBATCH --mem=6G
#SBATCH --gres=gpu:2
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --job-name=hallo
#SBATCH --output=/work/smfrrohk/Masterarbeit/logs/hallo.log
#SBATCH --mail-user=frederik.rohkraehmer@tu-dortmund.de
#SBATCH --mail-type=ALL
#-------------------------------------

module purge
module load nvidia/cuda/11.1.1
module load gcc/11.1.0

source /home/smfrrohk/anaconda3/bin/activate
conda activate PhysicsLearner

cd /work/smfrrohk/Masterarbeit/PhysicsLearner
srun python main.py