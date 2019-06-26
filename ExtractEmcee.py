from OptimizationGalario import *


location = 'results/optimization/optigal_13_560_3000_CUDA.npy'

def extractvalues(location,lconv=500,percentiles=[0.15,0.5,0.85]):
    samples,thetaminbis,thetamaxbis,labels=np.load(location,allow_pickle=True)
    l1,l2,l3=samples.shape
    samples_converged=samples[:,-lconv:,:].reshape(lconv*l1,l3)
    values=np.percentile(samples_converged,percentiles,axis=0)
    return(values)

values = extractvalues(location)

RadialModel = ModelJ1615(values[1])


samplesImage,_,_,_=np.load('../DiskFitting/results/optimization/opti_37_300_1000part40.npy')
valuesImage=extractvalues(samplesImage)

ablist=[13,14,20,21,27,28]
rlist=(ValuesImage[:,[13,20,27]]+ValuesImage[:,[14,21,28]])/2

