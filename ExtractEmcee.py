#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to open and plot the result of an EMCEE optimization, especially one made using OptimizationGalario.py
"""
from OptimizationGalario import *
from PlotModule import *
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

##### Galario Model
location = 'results/optimization/optigal_13_560_3000_CUDA.npy'
#get model parameters
values = extractvalues(location)

#get the profile
RadialModel = ModelJ1615(values[1])
#get the visibilities
visibilitiesModel=ComputeVisibilities(RadialModel)
#Re and Im, and weights (uniform)
ReModel,ImModel=ReandIm(visibilitiesModel)
wmodel=np.ones(len(w))

##### Plotting
# Data Binning
r1,m1,e1=Binning(Re, UVPlaneRadius, w, 1000)
r2,m2,e2=Binning(ReModel, UVPlaneRadius, w, 1000)
r3,m3,e3=Binning(Im, UVPlaneRadius, w, 1000)
r4,m4,e4=Binning(ImModel, UVPlaneRadius, w, 1000)

# Plot config
fig, axes=plt.subplots(2,sharex=True)
ax=axes.flatten()

ax[0].errorbar(r1,m1,e1,fmt=' ',label='Exerimental data',color='k',ecolor='k',capsize=2,elinewidth=.1,markeredgewidth=1)
ax[0].plot(r2,m2,label='Model')

ax[1].errorbar(r3,m3,e3,fmt=' ',label='Exerimental data',color='k',ecolor='k',capsize=2,elinewidth=.1,markeredgewidth=1)
ax[1].plot(r3,m3,label='Model')


ax[0].grid(which='both')
ax[0].grid(which='minor',alpha=0.2)
ax[0].grid(which='major',alpha=0.7)
ax[0].legend()
ax[0].set_title('Real part')
ax[0].set_xscale('log')

ax[1].grid(which='both')
ax[1].grid(which='minor',alpha=0.2)
ax[1].grid(which='major',alpha=0.7)
ax[1].legend()
ax[1].set_title('Imaginary part')
ax[1].set_xscale('log')


ax[1].set_xlabel(r'$(u,v)$ distance')
ax[1].set_ylabel(r'Visibility value')

fig.set_size_inches((4,8))
plt.tight_layout()
fig.savefig('results/UVProfile.pdf')
plt.close()


##### Plotting the radial profile
end=1200

fig=plt.figure()
plt.plot(R[:end]/arcsec,RadialModel[:end])
plt.yscale('log')
plt.grid(which='both')
plt.xlabel('Distance to the star (arcsec).')
plt.ylabel('Intensity (arbitrary).')
plt.title('Radial Profile Model.')
plt.savefig('results/RadialProfile.pdf')

'''
##### Image plane
samplesImage,_,_,_=np.load('../DiskFitting/results/optimization/opti_37_300_1000part40.npy')
valuesImage=extractvalues(samplesImage)

ablist=[13,14,20,21,27,28]
rlist=(ValuesImage[:,[13,20,27]]+ValuesImage[:,[14,21,28]])/2
'''
#####
