# GalarioFitting : fitting a Protoplanetary Disk

This package was designed to fit J1615 protoplanetary disk in the (u, v) plane, but is quite general and can be adapted quite easily to other disks.

## How to install the packages?
requirements :
1. Python3.7
2. Several modules : numpy, matplotlib, astropy, scipy, galario

### Import my repo
First, you should import my repository :
`git clone https://github.com/YohannFaure/GalarioFitting.git`

### Create a conda environment from a file (easy version)

Then you should [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (if you don't already have conda) and install python3.7 on a new environment.
If you don't know conda, see [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/).

```
cd GalarioFitting
conda env create -f CondaEnv.yml

conda activate GalarioFitting
git clone https://github.com/dfm/emcee.git
cd emcee
python3 setup.py install
cd ..
rm -rf emcee
conda deactivate
```

### Create a conda environment and install the packages (hard and often buggy version)

Then you should [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (if you don't already have conda) and install python3.7 on a new environment.
If you don't know conda, see [here](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/).

`conda create -n GalarioFitting python=3.7` (You can replace `GalarioFitting` by whatever you want.)

Then install all the necessary packages

```
conda install -n GalarioFitting numpy scipy matplotlib tqdm cython pytest
```

You then need the lattest version of Emcee (which might not be in the usual repos).
If you already have emcee installed with conda, you should uninstall it: `conda uninstall emcee`, and then reinstall it like so :

```
conda activate GalarioFitting
git clone https://github.com/dfm/emcee.git
cd emcee
python3 setup.py install
cd ..
rm -rf emcee
conda deactivate
```

Then you can install CUCA if you feel like it, but it's not necessary.


The last package is [galario](https://mtazzari.github.io/galario/install.html)

Fast version :
```
conda install -c conda-forge galario
```

And then you should be good to go!

## How to use it on a SLURM computing system (such as Leftraru)?

### installation
Log into your account and repeat the steps above.

#### If you've not imported the environment
Then you might want to use MPI instead of Multiprocessing, as it allows you to use more cores, more efficiently, and to use multiple nodes.

If you have not directly imported the environment, you need to install the proper packages.

```
conda activate GalarioFitting
conda install -c anaconda gcc
module load intel impi
export MPICC=`which mpicc`
conda install -c conda-forge openmpi
pip install mpi4py
python -c 'import mpi4py'
conda deactivate
```

If the `python -c 'import mpi4py'` line did not return an error you are good to go.

### Launching a script

To launch a script, the classical syntax is `sbatch /path/to/script`. It's easy, but the script must contain a detailed shebang and header, as follows :

```
#!/bin/bash
#SBATCH --partition=slims
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=2400
#SBATCH --mail-user=user@mail.ext
#SBATCH --mail-type=ALL
#SBATCH --output=GalLog.out
#SBATCH --error=GalLog.err

conda activate GalarioFitting
module load intel impi
cd ~/GalarioFitting

srun -n $SLURM_NTASKS python3 OptimizationGalarioMPI.py --nwalkers 560 --iterations 1000

conda deactivate
```

You should notice that `partitions` is the type of nodes tu use (general of slims on Leftraru), `ntasks` is the total number of threads to use, and `ntasks-per-node` is the number of cpus per node.

## What does each file do?

### `FunctionsModule.py`

It is a module with many usefull functions. Each function is described in the code.

### `TiltFinderVisibilities.py`

It finds the tilt of a Quasi-Gaussian image, *i.e.* the inclination and position angle of the image, as well as the center of the disk, using an emcee optimization.

The inc and pa will then be used to compute the visibilities with galario.

To use it, call `python3 TiltFinderVisibilities.py path/to/uvtable.txt nwalkers iterations nthreads`

Some options are available, described within the code.

### `STiltFinderSeed.py`

This is a seed file for the `TiltFinderVisibilities.py` program. It is meant to host the seed for the emcee optimization as well as its boundaries.

### `OptimizationGalario.py`

This program is meant to optimize a emcee fit in the image. It is quite custom and needs to be adapted, but it should be easy to modify.

The workings of the code are described in the code itself.

### `OptimizationGalarioMPI.py`
The same, with MPI support

### `MergeGalOpti.py`

Can be used to merge optimization files.

It can merge files named following this pattern :
filenameN.npy, ..., filenameM.npy
where N, ..., M are consecutive numbers, and filename is whatever you want, but all the same.

`python3 MergeGalOpti.py filename N M`

### `PlotEmcee.py`

Plots the emcee optimization and a cornerplot.

### `Legacy`

Random bits of code that migh be usefull one day.

## Contact and conditions

If you need help feel free to contact me
faure(dot)yohann(at)gmail(dot)com

You can branch and improve this code freely.
