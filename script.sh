#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --ntasks=44
#SBATCH --ntasks-per-node=44
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-user=faure.yohann@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/LastOpti.out
#SBATCH --error=results/LastOpti.err
#SBATCH --job-name=galopti
source ~/anaconda3/etc/profile.d/conda.sh

conda activate test
cd ~/GalarioFitting

python3 OptimizationGalario.py --nwalkers 560 --iterations 3000 --suffix _last --nthreads 44

conda deactivate
