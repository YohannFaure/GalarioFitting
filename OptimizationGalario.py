#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the needed functions in order to make some EMCEE and classical optimisation, such as models
You need to adapt it to open your own file.
'''
##### Import modules
import numpy as np
import matplotlib.pyplot as plt
import math
##### galario
from galario.double_cuda import get_image_size, chi2Profile # computes the image size required from the (u,v) data , computes a chi2
from galario import deg, arcsec # for conversions
##### Emcee
from emcee import EnsembleSampler
from multiprocessing import Pool

##### Because we don't want each thread to use multiple core for numpy computation. That forces the use of a proper multithreading
#import os
#os.environ["OMP_NUM_THREADS"] = "1"


Rmin = 1e-6  # arcsec
dR = 0.0006    # arcsec
nR = int(2./dR)
dR *= arcsec
Rmin*=arcsec
##### Define a mesh for the space
R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
u, v, Re, Im, w = np.require(np.loadtxt('uvtable2.txt', unpack=True), requirements='C')
nxy, dxy = get_image_size(u, v, verbose=False)


#lastsamples,_,_,_=np.load('results/optimization/optigal_6_60_6300_merged.npy',allow_pickle=True)
#lastsamples=lastsamples[:,-500:,:].reshape((-1,6))


#igauss,wgauss,inc,PA,dRA,dDec=np.mean(lastsamples[:,:], axis=0)
#siginc,sigPA,sigdRA,sigdDec=np.std(lastsamples[:,2:], axis=0)


igauss,wgauss,inc,PA,dRA,dDec=10.420409422990984, 0.3862295668301837, 0.8062268497358551, 2.5555283969130116, 4.727564265115043e-08, -2.6084635508588672e-08
inc,PA,dRA,dDec=inc*deg,PA*deg,dRA*arcsec,dDec*arcsec


##### Define a gaussian profile
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
    Facchini Ring, refer to eq 1 in arXiv 1905.09204.
    """
    f = (i0 * ((R / sig)**gam)) * np.exp(-((R-center)**2.) / (2. * sig**2.))
    return(f)


##### Just to define which to multiply by arcsec and which to power10
power10=np.array([0,2,6,9,12])
ListOfParams=np.arange(0,15,1)
mask = np.ones(ListOfParams.shape,dtype=bool)
mask[power10]=np.zeros(power10.shape)
timesarcsec=ListOfParams[mask]

def pre_conversion(p):
    pout=np.zeros(p.shape)
    pout[power10]=10**p[power10]
    pout[timesarcsec]=arcsec*p[timesarcsec]
    return(pout)

def ModelJ1615(p):
    f0_0, sigma_0,i0_1,sig_1,gam_1,center_1,amplitude_2, width_2, center_2,amplitude_3, width_3, center_3,amplitude_4, width_4, center_4=pre_conversion(p)
    return(GaussianProfile(f0_0, sigma_0)+
        PowerGaussianRing(i0_1,sig_1,gam_1,center_1)+
        GaussianRing(amplitude_2, width_2, center_2)+
        GaussianRing(amplitude_3, width_3, center_3)+
        GaussianRing(amplitude_4, width_4, center_4))

#p0 = np.array([igauss*0.97,0.02,
#    igauss*.98,0.1,3.,0.12,
#    igauss*.95,.5,0.15,
#    igauss*.93,.05,.5,
#    igauss*.93,.05,.7])

labels=['f0_0', 'sigma_0',
        'i0_1','sig_1','gam_1','center_1',
        'amplitude_2', 'width_2', 'center_2',
        'amplitude_3', 'width_3', 'center_3',
        'amplitude_4', 'width_4', 'center_4']

#p_range=np.transpose(np.array((p0/100,p0*100)))

p_range=np.array([
    [8.,15.],
        [-0.,0.02],#
    [8.,12.],
        [0.05,0.15],
        [0.1,5.],
        [-0.,0.4],#
    [1.,20.],
        [0.1,1.],
        [-0.,0.3],#
    [5.,15.],
        [0.01,0.3],
        [0.4,1.2],#
    [2.,12.],
        [-0.,0.1],
        [-0.,1.2]
    ])


p0=np.mean(p_range,axis=1)
#plt.plot(f);plt.plot(ModelJ1615(p0));plt.show()


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
    # compute the model brightness profile
    f = ModelJ1615(p)
    chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w, inc=inc, PA=PA, dRA=dRA, dDec=dDec)
    return(-0.5 * chi2)


nwalkers=150
ndim=15
nthreads=4
iterations=100

pos = np.array([(1. + 1.e-1*np.random.random(ndim))*p0 for i in range(nwalkers)])

with Pool(processes=nthreads) as pool:
    sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,pool=pool)
    pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)

np.save('results/optimization/optigal_{}_{}_{}'.format(ndim,nwalkers,iterations),(samples,p_range[:,0],p_range[:,1],labels))

#np.save('firsttestopti.npy',sampler.chain)
