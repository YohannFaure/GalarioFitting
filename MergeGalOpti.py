#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
just merges some files
suppose I want to merge optigal0.npy, optigal1.npy, optigal2.npy and optigal3.npy :

python3 MergeGalOpti optigal 0 3

it will be saved in optigalmerged.npy
"""
import numpy as np

def Merge(prefix,start,end):
    MergedOpti,val1,val2,val3=np.load('{}{}.npy'.format(prefix,start))
    for i in range(start+1,end+1):
        MergedOpti=np.concatenate((MergedOpti,np.load('{}{}.npy'.format(prefix,i))[0]),axis=1)
    np.save('{}merged.npy'.format(prefix),(MergedOpti,val1,val2,val3))


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'arguments')
    parser.add_argument("prefix", help = "Prefix to use.",type = str)
    parser.add_argument("start", help = "starting point.",type = int)
    parser.add_argument("end", help = "ending point.",type = int)
    args = parser.parse_args()
    Merge(args.prefix,args.start,args.end)
