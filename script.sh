#!/bin/bash
#SBATCH --partition=largemem
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=2400
#SBATCH --mail-user=faure.yohann@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=results/LastOpti.out
#SBATCH --error=results/LastOpti.err
#SBATCH --job-name=galYF


source activate yfaure
module load intel impi
cd ~/yfaure/GalarioFitting

srun -n $SLURM_NTASKS python3 OptimizationGalarioMPI.py --nwalkers 560 --iterations 1000 --suffix _last --resume results/optimization/optigal_13_560_2000_14Jun_1.npy

source deactivate
