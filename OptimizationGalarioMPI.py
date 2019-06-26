#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the needed functions in order to make some EMCEE and classical optimisation, such as models
You might need to adapt it to open your own file.

run it like that :

python3 OptimizationGalario.py --nwalkers 28 --iterations 10 --nthreads 4 --suffix test --split 2

python3 OptimizationGalario.py
'''
##### Import modules
import numpy as np
import math
import scipy
from galario import deg, arcsec # for conversions
import schwimmbad
import sys


from galario.double import get_image_size, chi2Profile, sampleProfile
cuda=False

##### Emcee
from emcee import EnsembleSampler

import os
os.environ["OMP_NUM_THREADS"] = "1"


##### Define the parameters of the mesh
Rmin = 1e-6  # arcsec
dR = 0.0008    # arcsec
nR = int(1.5/dR)
dR *= arcsec
Rmin*=arcsec

np.random.seed(15)

##### Define a mesh for the space
R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
u, v, Re, Im, w = np.require(np.loadtxt('uvtable2.txt', unpack=True), requirements='C')

##### Define inc and PA, as well as dRA and dDec
##### Those values come from TiltFinderVisibilities
igauss,inc,PA,dRA,dDec=10.420409422990984, 0.8062268497358551, 2.5555283969130116, 4.727564265115043e-08, -2.6084635508588672e-08
#convert
inc,PA,dRA,dDec=inc*deg,PA*deg,dRA*arcsec,dDec*arcsec

##### Define the diferent profiles
def GaussianProfile(f0, sigma):
    """ Gaussian brightness profile.
    """
    return(f0 * np.exp(-(0.5*((sigma)**-2.))*(R**2.) ))

def GaussianRing(amplitude, width, center):
    """
    This makes a gaussian ring centered on (xc,yc), elliptic with semi-axis a and b, and rotation theta.
    """
    # compute gaussian
    return( amplitude * np.exp(  ( -.5*(width**-2.) ) * ((R-center)**2.) ) ) 

def PowerGaussianRing(i0,sig,gam,center):
    """
    gaussian time a power of R
    """
    f = (i0 * ((R / sig)**gam)) * np.exp(-((R-center)**2.) / (2. * sig**2.))
    return(f)


##### Just to define which to multiply by arcsec and which to power10
##### You should adapt this to your code, this is for the J1615 Model I used
power10=np.array([0,2,5,7,10])
ListOfParams=np.arange(0,13,1)
#compute the mask
mask = np.ones(ListOfParams.shape,dtype=bool)
mask[power10]=np.zeros(power10.shape)
timesarcsec=ListOfParams[mask]

def pre_conversion(p):
    """converts the values for galario, so you can read arcsec"""
    pout=np.zeros(p.shape)
    pout[power10]=10**p[power10]
    pout[timesarcsec]=arcsec*p[timesarcsec]
    return(pout)

def ModelJ1615(p):
    """J1615 modelisation. I just added one gaussian and 4 gaussian rings"""
    f0_0, sigma_0,amplitude_1, width_1, center_1,f0_2,sigma0_2,amplitude_3, width_3, center_3,amplitude_4, width_4, center_4 = pre_conversion(p)
    return(GaussianProfile(f0_0, sigma_0)+
        GaussianRing(amplitude_1, width_1, center_1)+
        GaussianProfile(f0_2,sigma0_2)+
        GaussianRing(amplitude_3, width_3, center_3)+
        GaussianRing(amplitude_4, width_4, center_4))

##### define the displayed labels
labels=['f0_0', 'sigma_0',
        'amplitude_1', 'width_1', 'center_1',
        'f0_2', 'sigma_2',
        'amplitude_3', 'width_3', 'center_3',
        'amplitude_4', 'width_4', 'center_4']

##### define the cost functions
def lnpriorfn(p):
    if np.any(p<p_range[:,0]) or np.any(p>p_range[:,1]):
        return(-np.inf)
    return(0.0)

def lnpostfn(p):
    """ Log of posterior probability function using galario"""
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return -np.inf
    # compute the model brightness profile
    f = ModelJ1615(p)
    chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w, inc=inc, PA=PA, dRA=dRA, dDec=dDec)
    return(-0.5 * chi2)

def chi2compute(ModelVal,Re,Im,w):
    return( np.sum( ((np.imag(ModelVal)-Im)**2.+(np.real(ModelVal)-Re)**2.)*w  ) )

def lnpostfnbis(p):
    """ Log of posterior probability function using less galario (faster)"""
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return(-np.inf)
    # compute the model brightness profile
    f = ModelJ1615(p)
    ModelVal=sampleProfile(f,Rmin,dR,nxy,dxy,u,v)
    chi2 = chi2compute(ModelVal,Re,Im,w)
    return(-0.5 * chi2)

def tominimize(p):
    """ Only for the classical optimization process """
    print(p)
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return(-np.inf)
    # compute the model brightness profile
    f = ModelJ1615(p)
    ModelVal=sampleProfile(f,Rmin,dR,nxy,dxy,u,v)
    chi2 = chi2compute(ModelVal,Re,Im,w)
    return(chi2/1000000)


##### Different values obtained with a classical optimization process
##### use these lines to compute such values. You need to define p_range before.
#p0 = (a seed)
#p0=scipy.optimize.fmin_slsqp(tominimize,p0,bounds=p_range)

p0list=np.array([
        [10.92464736, 0.01183856,
        10.30642221,  0.09737238,  0.16015055,
        9.95183693,   0.42739781,
        8.41536481,   0.08035405,  0.74988449,
        8.459264  ,   0.09994191,  0.65280675]
        ,
        [10.96789008, 0.01155506,
        10.29265437,  0.09478848,  0.16229442,
        9.98834487,   0.4406268 ,
        8.45186003,   0.08843546,  0.74950883,
        8.49680396,   0.06784508,  0.671583]
        ,
        [1.10833490e+01,1.02650502e-02,
        1.02918828e+01, 1.02526895e-01,  1.59372502e-01,
        9.95803536e+00, 4.36283306e-01,
        8.42835610e+00, 1.02136030e-01,  7.45828481e-01,
        8.47980262e+00, 9.30403301e-02,  7.15890398e-01]
        ,
        [1.10777223e+01,9.76108116e-03,
        1.02946656e+01, 9.51628786e-02, 1.61276380e-01,
        9.98591971e+00, 4.39358011e-01,
        8.45076133e+00, 8.50551811e-02, 7.52851430e-01,
        8.49338800e+00, 6.67571321e-02, 6.61434890e-01]
        ,
        [1.10777223e+01,9.76107989e-03,
        1.02946656e+01, 9.51628802e-02, 1.61276382e-01,
        9.98591971e+00, 4.39358013e-01,
        8.45076133e+00, 8.50551813e-02, 7.52851430e-01,
        8.49338800e+00, 6.67571322e-02, 6.61434890e-01]
        ,
        [10.92464738, 0.01183863,
        10.30642216,  0.09737231,  0.16015049,
        9.9518369 ,   0.42739774,
        8.41536481,   0.08035404,  0.74988449,
        8.459264  ,   0.0999419 ,  0.65280675]
        ,
        [1.10777223e+01,9.76107989e-03,
        1.02946656e+01, 9.51628802e-02, 1.61276382e-01,
        9.98591971e+00, 4.39358013e-01,
        8.45076133e+00, 8.50551813e-02, 7.52851430e-01,
        8.49338800e+00, 6.67571322e-02, 6.61434890e-01]
        ,
        [1.15e+01,5.76107989e-03,
        1.025e+01, 9.44e-02, 1.645e-01,
        10.05e+00, 4.5e-01,
        8.73e+00, 8.5e-02, 7.32e-01,
        8.6e+00, 3.e-02, 6.2e-01]
        ])

p_range=np.array([
    [10.6,15.],
        [0.000001,0.016],#
    [10.,10.5],
        [0.07,0.12],
            [0.1,0.2],#
    [9.,11.],
        [0.1,0.9],
    [8.,9.],
        [0.01,0.13],
            [0.7,1.],#
    [8.2,10.],
        [0.01,0.13],
            [0.6,.8]
    ])

##### Different other ways of obtaining classical optimizations
#p0=scipy.optimize.minimize(tominimize,p0,method='Nelder-Mead')
#p0=scipy.optimize.minimize(tominimize,p0,method='Nelder-Mead')
#p0=scipy.optimize.minimize(tominimize,p0,method='Powell')
#p0=scipy.optimize.minimize(tominimize,p0,method='CG')
#p0=scipy.optimize.fmin_slsqp(tominimize,p0,bounds=p_range)



if __name__=='__main__':
    ##### Argparser
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("--nwalkers", help = "Number of walkers",type = int,default='280')
    parser.add_argument("--iterations", help = "number of iterations",type = int,default='2000')
    parser.add_argument("--nthreads", help = "Number of cpu threads",type = int,default='20')
    parser.add_argument("--suffix", help = "Suffix to file name.",type = str,default='')
    parser.add_argument("--resume", help = "File to resume training",type = str,default=None)
    parser.add_argument("--split",help='If you want to split your workload in n parts, enter n here.',type=int, default=None)
    parser.add_argument("--cuda",help='If you want to use CUDA',action='store_true')
    args = parser.parse_args()
    ##### galario, store is cuda is OK on that machine.
    if args.cuda:
        try:
            from galario.double_cuda import get_image_size, chi2Profile, sampleProfile # computes the image size required from the (u,v) data , computes a chi2
            cuda=True
            print('cuda ON')
        except :
            from galario.double import get_image_size, chi2Profile, sampleProfile
            cuda=False
            print('cuda OFF')
    else :
        print('cuda OFF')
    ##### Get the size of the image
    nxy, dxy = get_image_size(u, v, verbose=False)

    ##### define emcee parameters
    ndim       = 13                          # number of dimensions
    nwalkers   = args.nwalkers               # number of walkers
    nthreads   = args.nthreads               # CPU threads that emcee should use
    iterations = args.iterations             # total number of MCMC steps

    ##### Check if you want to resume something
    if args.resume:
        samples,_,_,_=np.load(args.resume,allow_pickle=True)
        pos=samples[:,-1,:]
    else :
        ##### initialize the walkers with an ndim-dimensional ball
        pos1 = np.array([(1. + 1.e-2*np.random.random(ndim))*p0list[i%len(p0list)] for i in range(nwalkers//2)])
        pos2 = np.transpose([np.random.uniform(p_range[i,0],p_range[i,1],nwalkers//2+nwalkers%2) for i in range(ndim)])
        pos=np.concatenate((pos1,pos2),axis=0)
    ##### If we want to split the data
    if args.split :
        n=args.split
        m=iterations//n
        lastm=iterations%n
        ##### launch the mcmc
        for i in range(n):
            with schwimmbad.MPIPool() as pool:
                sampler = EnsembleSampler(nwalkers, ndim, lnpostfnbis,pool=pool)
                pos, prob, state = sampler.run_mcmc(pos, m, progress=True)
            # save each part

            samples=sampler.chain
            #To save the data.
            np.save("results/optimization/optigal_{}_{}_{}{}_split{}.npy".format(ndim, nwalkers, iterations, args.suffix, i),(samples,p_range[:,0],p_range[:,1],labels))
            print(i,' out of ',n)
        if lastm!=0:
            with schwimmbad.MPIPool() as pool:
                sampler = EnsembleSampler(nwalkers, ndim, lnpostfnbis,pool=pool)
                pos, prob, state = sampler.run_mcmc(pos, lastm, progress=True)

            samples=sampler.chain
            #To save the data.
            np.save("results/optimization/optigal_{}_{}_{}{}_split{}.npy".format(ndim, nwalkers, iterations, args.suffix,n),(samples,p_range[:,0],p_range[:,1],labels))
    ##### If we don't want to split
    else :
        ##### execute the MCMC
        with schwimmbad.MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = EnsembleSampler(nwalkers, ndim, lnpostfnbis,pool=pool)
            pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)
            pool.close()
            sys.exit(0)
        samples=sampler.chain
        #To save the data.
        np.save("results/optimization/optigal_{}_{}_{}{}.npy".format(ndim, nwalkers, iterations, args.suffix),(samples,p_range[:,0],p_range[:,1],labels))

