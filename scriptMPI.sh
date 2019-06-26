#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --ntasks=60
#SBATCH --ntasks-per-node=30
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-user=faure.yohann@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/LastOptiMPI.out
#SBATCH --error=results/LastOptiMPI.err
#SBATCH --job-name=galmpi

ml Galario/1.2.1
ml impi
cd ~/GalarioFitting

srun -n $SLURM_NTASKS python OptimizationGalarioMPI.py --nwalkers 560 --iterations 1000 --suffix _MPI
