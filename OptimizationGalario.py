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
import scipy
##### galario
from galario.double import get_image_size, chi2Profile, sampleProfile # computes the image size required from the (u,v) data , computes a chi2
from galario import deg, arcsec # for conversions
##### Emcee
from emcee import EnsembleSampler
from multiprocessing import Pool

##### Because we don't want each thread to use multiple core for numpy computation. That forces the use of a proper multithreading
#import os
#os.environ["OMP_NUM_THREADS"] = "1"


Rmin = 1e-6  # arcsec
dR = 0.0008    # arcsec
nR = int(1.5/dR)
dR *= arcsec
Rmin*=arcsec
##### Define a mesh for the space
R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
u, v, Re, Im, w = np.require(np.loadtxt('uvtable2.txt', unpack=True), requirements='C')


#x=np.random.rand(len(uf))<0.01

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
power10=np.array([0,2,5,8,11])
ListOfParams=np.arange(0,14,1)
mask = np.ones(ListOfParams.shape,dtype=bool)
mask[power10]=np.zeros(power10.shape)
timesarcsec=ListOfParams[mask]

def pre_conversion(p):
    pout=np.zeros(p.shape)
    pout[power10]=10**p[power10]
    pout[timesarcsec]=arcsec*p[timesarcsec]
    return(pout)

def ModelJ1615(p):
    f0_0, sigma_0,i0_1,sig_1,center_1,amplitude_2, width_2, center_2,amplitude_3, width_3, center_3,amplitude_4, width_4, center_4=pre_conversion(p)
    return(GaussianProfile(f0_0, sigma_0)+
        #PowerGaussianRing(i0_1,sig_1,gam_1,center_1)+
        GaussianRing(i0_1, sig_1, center_1)+
        GaussianRing(amplitude_2, width_2, center_2)+
        GaussianRing(amplitude_3, width_3, center_3)+
        GaussianRing(amplitude_4, width_4, center_4))

labels=['f0_0', 'sigma_0',
        'i0_1','sig_1', 'center_1',
        'amplitude_2', 'width_2', 'center_2',
        'amplitude_3', 'width_3', 'center_3',
        'amplitude_4', 'width_4', 'center_4']

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

def chi2compute(ModelVal,Re,Im,w):
    return( np.sum( ((np.imag(ModelVal)-Im)**2.+(np.real(ModelVal)-Re)**2.)*w  ) )


def lnpostfnbis(p):
    """ Log of posterior probability function """
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return(-np.inf)
    # compute the model brightness profile
    f = ModelJ1615(p)
    ModelVal=sampleProfile(f,Rmin,dR,nxy,dxy,u,v)
    chi2 = chi2compute(ModelVal,Re,Im,w)
    return(-0.5 * chi2)

def tominimize(p):
    """ Log of posterior probability function """
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return(-np.inf)
    # compute the model brightness profile
    f = ModelJ1615(p)
    ModelVal=sampleProfile(f,Rmin,dR,nxy,dxy,u,v)
    chi2 = chi2compute(ModelVal,Re,Im,w)
    return(chi2)

p0=[10.85096006,  0.01146675,
    10.35811935,  0.09957152,  0.15475334,
    9.85595757,  0.38882761,  0.24746409,
    8.44414471,  0.06197724,  0.52520985,
    8.4625868 ,  0.18377439, 0.55165571]


p_range=np.array([
    [8.,15.],
        [-0.,0.02],#
    [10.,11.],
        [0.05,0.15],
    #    [0.1,5.],
        [0.1,0.2],#
    [9.,11.],
        [0.3,0.5],
        [0.1,0.4],#
    [7.,11.],
        [0.01,0.25],
        [0.5,0.9],#
    [8.,11.],
        [0.1,0.25],
        [0.5,0.8]
    ])


#a=scipy.optimize.minimize(tominimize,p0,method='Nelder-Mead')
#a=scipy.optimize.minimize(tominimize,a['x'],method='Nelder-Mead')
#a=scipy.optimize.minimize(tominimize,p0,method='Powell')
#a=scipy.optimize.minimize(tominimize,p0,method='CG')
#a=scipy.optimize.minimize(tominimize,p0,method='BFGS',bounds=p_range)

#p0=a[x]

nwalkers=280
ndim=14
nthreads=20
iterations=1000

pos = np.array([(1. + 2.e-1*np.random.random(ndim))*p0 for i in range(nwalkers)])


import os
os.environ["OMP_NUM_THREADS"] = "1"

with Pool(processes=nthreads) as pool:
    sampler = EnsembleSampler(nwalkers, ndim, lnpostfnbis,pool=pool)
    pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)

samples=sampler.chain

np.save('results/optimization/optigal_{}_{}_{}'.format(ndim,nwalkers,iterations),(samples,p_range[:,0],p_range[:,1],labels))

#np.save('firsttestopti.npy',sampler.chain)
