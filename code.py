#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This is just a galario test.

In order to use galario and this test script, you need Anaconda.

How to install :
1) Install anaconda
2) activate your conda console by typing
conda activate
3) in the console type :

conda install (module)

with (module) being replaced by :
    numpy
    astropy
    galario
    multiprocessing

4) Install emcee 3 by doing this :
conda uninstall emcee
git clone https://github.com/dfm/emcee.git
cd emcee
python3 setup.py install

You are good to go :D
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
#import corner
from multiprocessing import Pool


##### Because we don't want each thread to use multiple core for numpy computation. That forces the use of a proper multithreading
import os
os.environ["OMP_NUM_THREADS"] = "1"


##### load data
u, v, Re, Im, w = np.require(np.loadtxt("uvtable2.txt", unpack=True), requirements='C')

##### for conversion
#wavelength = 0.00087  # [m]
#u /= wavelength
#v /= wavelength

##### get size of the image
nxy, dxy = get_image_size(u, v, verbose=False) # Number of pixel, width of a pixel in rad

##### radial grid parameters, fixed
Rmin = 1e-6  # arcsec
dR = 0.0006    # arcsec
nR = int(2/dR)
dR *= arcsec
Rmin*=arcsec
##### Define a mesh for the space
R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)

##### Define a gaussian profile
def GaussianProfile(f0, sigma):
    """ Gaussian brightness profile.
    """
    return( f0 * np.exp(-(0.5/(sigma**2.))*(R**2.) ))

##### parameter space domain
p0 = np.array([10.5, 0.35, 40., 140., 0.1, 0.1]) #  2 parameters for the model + 4 (inc, PA, dRA, dDec)
p_range = np.array([[10, 11],    #f0
            [0.25, 0.5],   #sigma
            [30., 50.],  #inc
            [120., 180.], #pa
            [-2., 2.],  #dra
            [-2., 2.]])  #ddec

##### define emcee parameters
ndim = len(p_range)           # number of dimensions
nwalkers   = 60               # number of walkers
nthreads   = 16                # CPU threads that emcee should use
iterations = 200             # total number of MCMC steps

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

##### initialize the walkers with an ndim-dimensional Gaussian ball
pos = np.array([(1. + 1e-4*np.random.random(ndim))*p0 for i in range(nwalkers)])

##### execute the MCMC
with Pool(processes=nthreads) as pool:
    sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,pool=pool)
    pos, prob, state = sampler.run_mcmc(pos, iterations, progress=True)

#samples,_,_,_=np.load('optigal_6_60_5000.npy')
#pos=samples[:,-1,:]

samples = np.concatenate((samples,sampler.chain),axis=1)
#samples=sampler.chain
#To save the data.
iterations=6000
np.save("optigal_{}_{}_{}.npy".format(ndim, nwalkers, iterations),(samples,p_range[:,0],p_range[:,1],["$f_0$", "$\sigma$", r"$i$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]))

########## Plotting the result :

"""
##### Get the shape of the plot
nwalkers,iterations,ndims = samples.shape
ncols = 3
nrows = 2

##### labeling
labels=[r"$f_0$", r"$\sigma$", r"$Inc$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]

##### Make a figure
fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(15, 10), sharex=True)
for i in range(ndims):
    ax = axes.flatten()[i]
    _=ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
    _=ax.set_xlim(0, iterations)
    _=ax.set_ylabel(labels[i])
#    _=ax.yaxis.set_label_coords(-0.1, 0.5)
#    _=ax.plot([0,iterations],[p_range[i,0],p_range[i,0]])
#    _=ax.plot([0,iterations],[p_range[i,1],p_range[i,1]])

_=ax.set_xlabel('iterations')
plt.tight_layout()
#plt.savefig('test.png',dpi=600)
plt.show()




##### This would be for a cornerplot
#Define the labels
labels=["$f_0$", "$\sigma$", r"$i$", r"PA", r"$\Delta$RA", r"$\Delta$Dec"]
#Reshape the array (we want all the converged iterations to be taken into account)
cornering=(samples[:,-3000:,:].reshape((-1,ndims)))


#Plot
fig = corner.corner(cornering, quantiles=[0.16, 0.50, 0.84],labels=labels,show_titles=True,label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8)
fig.show()
fig.savefig("triangle_example.png",dpi=600)
"""
