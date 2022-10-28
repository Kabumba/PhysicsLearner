#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --time=00:15:00
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