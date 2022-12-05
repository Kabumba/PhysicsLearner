#!/bin/bash -l
#SBATCH -c 1
#SBATCH --mem 8G
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --job-name=Test
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
srun python umordnen.py