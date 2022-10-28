#!/bin/bash -l
#SBATCH -C cgpu01
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --time=00:00:20
#SBATCH --job-name=test
#SBATCH --output=/work/smfrrohk/Masterarbeit/logs/test.log
#SBATCH --mail-user=frederik.rohkraehmer@tu-dortmund.de
#SBATCH --mail-type=ALL
#-------------------------------------

module purge
module load nvidia/cuda/11.1.1

source /home/smfrrohk/anaconda3/bin/activate
conda activate PhysicsLearner

cd /work/smfrrohk/Masterarbeit/PhysicsLearner
srun python main.py