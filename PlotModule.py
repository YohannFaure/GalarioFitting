#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from OptimizationGalario import *

UVPlaneRadius=np.sqrt(u**2+v**2)

def BinnedRadius(radius,BinsNumber):
    maxbin=np.max(radius)*(1+.1/BinsNumber)
    binsize=maxbin/BinsNumber
    bins=(radius//binsize).astype(int)
    return(bins)

def BinnedValues(vals,bins):
    binslist=[[] for i in range(np.max(bins)+1)]
    for i,val in enumerate(vals):
        binslist[bins[i]].append(val)
    return(binslist)

def weight_std(vals,w):
    n=len(vals)
    mean=np.mean(vals)
    return((1/(n-1))*math.sqrt(np.sum((vals-mean)**2*w)))

def BinningMeansAndErrors(ValsBinList,WeightsBinList):
    Errors=[]
    Means=[]
    NoneIndex=[]
    for i,bin in enumerate(ValsBinList):
        if len(bin)>1:
            Errors.append(weight_std(ValsBinList[i],WeightsBinList[i]))
            Means.append(np.median(bin))
        else:
            Errors.append(None)
            Means.append(None)
            NoneIndex.append(i)
    return(np.array(np.delete(Means,NoneIndex)),np.array(list(np.delete(Errors,NoneIndex).astype(list))),np.array(NoneIndex))

def Binning(vals,radius,weights,BinsNumber):
    radtemp=BinnedRadius(radius,BinsNumber)
    r=np.arange(0,1,1/BinsNumber)*np.max(radius)
    r+=r[1]/2
    ValsBinList=BinnedValues(vals,radtemp)
    WeightsBinList=BinnedValues(weights,radtemp)
    m,e,NoneIndex=BinningMeansAndErrors(ValsBinList,WeightsBinList)
    r=np.delete(r,NoneIndex)
    e=np.transpose(e)
    return(r,m,e)

from galario.double import get_image_size, chi2Profile, sampleProfile

def extractvalues(location,lconv=500,percentiles=[0.15,0.5,0.85]):
    samples,thetaminbis,thetamaxbis,labels=np.load(location,allow_pickle=True)
    l1,l2,l3=samples.shape
    samples_converged=samples[:,-lconv:,:].reshape(lconv*l1,l3)
    values=np.percentile(samples_converged,percentiles,axis=0)
    return(values)


nxy, dxy = get_image_size(u, v, verbose=False)

def ComputeVisibilities(RadialModel):
    return(sampleProfile(RadialModel,Rmin,dR,nxy,dxy,u,v))

def ReandIm(x):
    return(np.real(x),np.imag(x))


