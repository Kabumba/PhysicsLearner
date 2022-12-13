#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH --partition=med
#SBATCH --time=08:00:00
#SBATCH --job-name=Med4
#SBATCH --output=/work/smfrrohk/Masterarbeit/Logs/run-%j.log
#SBATCH --mail-user=frederik.rohkraehmer@tu-dortmund.de
#SBATCH --mail-type=ALL
#-------------------------------------

module purge
module load nvidia/cuda/11.1.1
module load gcc/9.2.0

source /home/smfrrohk/anaconda3/bin/activate
conda activate PhysicsLearner

cd /work/smfrrohk/Masterarbeit/PhysicsLearner
srun python main.py --name 4