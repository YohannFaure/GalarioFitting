#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In the seed you should define p0 as the seed, and p_range as the range for each parameter.

example :

p0 = np.array([10.5, 0.35, 40., 140., 0.1, 0.1]) #  2 parameters for the model + 4 (inc, PA, dRA, dDec)

p_range = np.array([
            [10, 11],    #f0
            [0.25, 0.5],   #sigma
            [30., 50.],  #inc
            [120., 180.], #pa
            [-2., 2.],  #dra
            [-2., 2.]])  #ddec



"""

p0 = np.array([10.5, 0.35, 40., 140., 0.1, 0.1]) #  2 parameters for the model + 4 (inc, PA, dRA, dDec)

p_range = np.array([[10., 11.],    #f0
            [0.25, 0.5],   #sigma
            [30., 50.],  #inc
            [120., 180.], #pa
            [-2., 2.],  #dra
            [-2., 2.]])  #ddec

if __name__=='__main__':
    print('Seed has been imported')
