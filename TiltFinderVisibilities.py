#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This finds the inclination and position angle of a disk based on a UV table, supposing it's gaussian enough.

output : inc,pa,dRA,dDec
"""

##### Import modules
import numpy as np
import matplotlib.pyplot as plt
import math
##### galario
from galario.double import get_image_size, chi2Profile # computes the image size required from the (u,v) data , computes a chi2
from galario import deg, arcsec # for conversions
##### Emcee
from emcee import EnsembleSampler
from multiprocessing import Pool

##### Because we don't want each thread to use multiple core for numpy computation. That forces the use of a proper multithreading
import os
os.environ["OMP_NUM_THREADS"] = "1"

##### Define a gaussian profile
def GaussianProfile(f0, sigma):
    """ Gaussian brightness profile.
    """
    return( f0 * np.exp(-(0.5/(sigma**2.))*(R**2.) ))

##### Define a conversion to translate the data for galario.double.chi2Profile
def convertp(p):
        f0, sigma, inc, PA, dRA, dDec = p
        return(10.**f0, sigma*arcsec, inc*deg, PA*deg, dRA*arcsec, dDec*arcsec)

##### define the cost functions
def lnpriorfn(p):
    if np.any(p<p_range[:,0]) or np.any(p>p_range[:,1]):
        return(-np.inf)
    return(0.0)

def lnpostfn(p):
    """ Log of posterior probability function """
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return -np.inf
    # unpack the parameters
    f0, sigma, inc, PA, dRA, dDec = convertp(p)
    # compute the model brightness profile
    f = GaussianProfile(f0, sigma)
    chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w, inc=inc, PA=PA, dRA=dRA, dDec=dDec)
    return(-0.5 * chi2)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("location", help = "File to use as image data.",type = str)
    parser.add_argument("nwalkers", help = "Number of walkers",type = int)
    parser.add_argument("iterations", help = "number of iterations",type = int)
    parser.add_argument("nthreads", help = "Number of cpu threads",type = int)
    parser.add_argument("--suffix", help = "Suffix to file name.",type = str,default='')
    parser.add_argument("--resume", help = "File to resume training",type = str,default=None)
    parser.add_argument("--seed",help='You can put a seed in a .py file, and use it.',type=str, default=None)
    parser.add_argument("--split",help='If you want to split your workload in n parts, enter n here.',type=int, default=None)
    args = parser.parse_args()
    if args.seed :
        exec(open(args.seed).read())
    else :
        ##### parameter space domain
        p0 = np.array([10., 0.1, 0.1, 140., 0.1, 0.1]) #  2 parameters for the model + 4 (inc, PA, dRA, dDec)
        p_range = np.array([[1., 100.],    #f0
            [2e-5, 2e5],   #sigma
            [-1., 91.],  #inc
            [-1., 181.], #pa
            [-2., 2.],  #dra
            [-2., 2.]])  #ddec

    ##### radial grid parameters, fixed, YOU CAN AND SHOULD MANUALLY CHANGE THAT
    Rmin = 1e-6  # arcsec
    dR = 0.0006    # arcsec
    nR = int(2./dR)
    dR *= arcsec
    Rmin*=arcsec
    ##### Define a mesh for the space
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)

    ##### load data
    u, v, Re, Im, w = np.require(np.loadtxt(args.location, unpack=True), requirements='C')

    ##### for conversion
    #wavelength = 0.00087  # [m]
    #u /= wavelength
    #v /= wavelength

    ##### get size of the image
    nxy, dxy = get_image_size(u, v, verbose=False) # Number of pixel, width of a pixel in rad

    ##### define emcee parameters
    ndim       = 6                           # number of dimensions
    nwalkers   = args.nwalkers               # number of walkers
    nthreads   = args.nthreads               # CPU threads that emcee should use
    iterations = args.iterations             # total number of MCMC steps

    if args.resume:
        samples,_,_,_=np.load(args.resume,allow_pickle=True)
        pos=samples[:,-1,:]
    else :
        ##### initialize the walkers with an ndim-dimensional Gaussian ball
        pos = np.array([(1. + 1e-4*np.random.random(ndim))*p0 for i in range(nwalkers)])

    if args.split :
        n=args.split
        m=iterations//n
        lastm=iterations%n
        for i in range(n):
            with Pool(processes=nthreads) as pool:
                sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,pool=pool)
                pos, prob, state = sampler.run_mcmc(pos, m, progress=True)

            samples=sampler.chain
            #To save the data.
            np.save("results/optimization/optigal_{}_{}_{}{}_split{}.npy".format(ndim, nwalkers, iterations, args.suffix,i),(samples,p_range[:,0],p_range[:,1],[r"$f_0$", r"$\sigma$", r"$i$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]))
        if lastm!=0:
            with Pool(processes=nthreads) as pool:
                sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,pool=pool)
                pos, prob, state = sampler.run_mcmc(pos, lastm, progress=True)

            samples=sampler.chain
            #To save the data.
            np.save("results/optimization/optigal_{}_{}_{}{}_split{}.npy".format(ndim, nwalkers, iterations, args.suffix,n),(samples,p_range[:,0],p_range[:,1],[r"$f_0$", r"$\sigma$", r"$i$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]))

    else :
    ##### execute the MCMC
        with Pool(processes=nthreads) as pool:
            sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,pool=pool)
            pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)

        samples=sampler.chain
        #To save the data.
        np.save("results/optimization/optigal_{}_{}_{}{}.npy".format(ndim, nwalkers, iterations, args.suffix),(samples,p_range[:,0],p_range[:,1],[r"$f_0$", r"$\sigma$", r"$i$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]))
    print(np.mean(pos,axis=0),'\n',np.std(pos,axis=0),'\n',np.percentile(pos,[.25,.5,.75],axis=0))
    print('(inc,pa)=',np.mean(pos,axis=0)[2:4])
