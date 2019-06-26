#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --ntasks=44
#SBATCH --ntasks-per-node=44
#SBATCH --mem-per-cpu=4365
#SBATCH --mail-user=faure.yohann@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/LastOptiCUDA.out
#SBATCH --error=results/LastOptiCUDA.err
#SBATCH --job-name=galgpu
#SBATCH --gres=gpu:2


ml fosscuda/2019b Galario/1.2.1

cd ~/GalarioFitting

python OptimizationGalario.py --nwalkers 560 --iterations 1000 --suffix _CUDA2 --cuda --nthreads 11 --resume results/optimization/optigal_13_560_3000_CUDA.npy
