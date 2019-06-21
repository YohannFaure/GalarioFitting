#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=2400
#SBATCH --mail-user=faure.yohann@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/LastOpti.out
#SBATCH --error=results/LastOpti.err
#SBATCH --job-name=galopti

source ~/anaconda3/etc/profile.d/conda.sh
conda activate GalarioFitting
module load intel impi
cd ~/GalarioFitting

srun -n $SLURM_NTASKS python3 OptimizationGalarioMPI.py --nwalkers 560 --iterations 10 --suffix _last

conda deactivate
