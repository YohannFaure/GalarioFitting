#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --ntasks=44
#SBATCH --ntasks-per-node=44
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=faure.yohann@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/LastOptiMPIC.out
#SBATCH --error=results/LastOptiMPIC.err
#SBATCH --job-name=galmpiC
echo test0

ml fosscuda/2019b Galario/1.2.1

ml impi

cd ~/GalarioFitting

srun -n 11 python OptimizationGalarioMPI.py --cuda --nwalkers 560 --iterations 1000 --suffix _MPIC
