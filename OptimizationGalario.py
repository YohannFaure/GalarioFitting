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

#samples,_,_,_=np.load('results/optimization/optigal_14_280_2000.npy',allow_pickle=True)


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
    print(p)
    lnp = lnpriorfn(p)
    if not np.isfinite(lnp):
        return(-np.inf)
    # compute the model brightness profile
    f = ModelJ1615(p)
    ModelVal=sampleProfile(f,Rmin,dR,nxy,dxy,u,v)
    chi2 = chi2compute(ModelVal,Re,Im,w)
    return(chi2/1000000)

p0list=np.array([
        [10.92464736, 0.01183856,
        10.30642221,  0.09737238,  0.16015055,
        9.95183693,   0.42739781,  0.1173729 ,
        8.41536481,   0.08035405,  0.74988449,
        8.459264  ,   0.09994191,  0.65280675]
        ,
        [10.96789008, 0.01155506,
        10.29265437,  0.09478848,  0.16229442,
        9.98834487,   0.4406268 ,  0.07194621,
        8.45186003,   0.08843546,  0.74950883,
        8.49680396,   0.06784508,  0.671583  ]
        ,
        [1.10833490e+01,1.02650502e-02,
        1.02918828e+01, 1.02526895e-01,  1.59372502e-01,
        9.95803536e+00, 4.36283306e-01,  9.71339541e-02,
        8.42835610e+00, 1.02136030e-01,  7.45828481e-01,
        8.47980262e+00, 9.30403301e-02,  7.15890398e-01]
        ,
        [1.10777223e+01,9.76108116e-03,
        1.02946656e+01, 9.51628786e-02, 1.61276380e-01,
        9.98591971e+00, 4.39358011e-01, 7.61627194e-02,
        8.45076133e+00, 8.50551811e-02, 7.52851430e-01,
        8.49338800e+00, 6.67571321e-02, 6.61434890e-01]
        ,
        [1.10777223e+01,9.76107989e-03,
        1.02946656e+01, 9.51628802e-02, 1.61276382e-01,
        9.98591971e+00, 4.39358013e-01, 7.61627207e-02,
        8.45076133e+00, 8.50551813e-02, 7.52851430e-01,
        8.49338800e+00, 6.67571322e-02, 6.61434890e-01]
        ,
        [10.92464738, 0.01183863,
        10.30642216,  0.09737231,  0.16015049,
        9.9518369 ,   0.42739774,  0.11737283,
        8.41536481,   0.08035404,  0.74988449,
        8.459264  ,   0.0999419 ,  0.65280675]
        ,
        [1.10777223e+01,9.76107989e-03,
        1.02946656e+01, 9.51628802e-02, 1.61276382e-01,
        9.98591971e+00, 4.39358013e-01, 7.61627207e-02,
        8.45076133e+00, 8.50551813e-02, 7.52851430e-01,
        8.49338800e+00, 6.67571322e-02, 6.61434890e-01]
        ])

p_range=np.array([
    [10.6,11.7],
        [0.006,0.016],#
    [10.,10.5],
        [0.07,0.12],
            [0.1,0.2],#
    [9.,11.],
        [0.39,0.47],
            [0.03,0.15],#
    [8.,9.],
        [0.05,0.13],
            [0.7,0.8],#
    [8.2,8.8],
        [0.03,0.13],
            [0.6,0.8]
    ])


#a=scipy.optimize.minimize(tominimize,p0,method='Nelder-Mead')
#a=scipy.optimize.minimize(tominimize,a['x'],method='Nelder-Mead')
#a=scipy.optimize.minimize(tominimize,p0,method='Powell')
#a=scipy.optimize.minimize(tominimize,p0,method='CG')

#####
#p0=scipy.optimize.fmin_slsqp(tominimize,p0,bounds=p_range)

nwalkers=280
ndim=14
nthreads=20
iterations=2000

pos = np.array([(1. + 1.e-2*np.random.random(ndim))*p0list[i%len(p0list)] for i in range(nwalkers)])
#pos=samples[:,-1,:]

import os
os.environ["OMP_NUM_THREADS"] = "1"

with Pool(processes=nthreads) as pool:
    sampler = EnsembleSampler(nwalkers, ndim, lnpostfnbis,pool=pool)
    pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)

samples=sampler.chain

np.save('results/optimization/optigal_{}_{}_{}TEST'.format(ndim,nwalkers,iterations),(samples,p_range[:,0],p_range[:,1],labels))

#np.save('firsttestopti.npy',sampler.chain)
