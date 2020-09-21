# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:11:17 2020

@author: long
"""


import numpy as np
from numpy import genfromtxt
import torch

def extract_info():
    #designed based on the structure of the input csv
    trainRef = genfromtxt('/mnt/sdc1/daicwoz/train_split_Depression_AVEC2017.csv', usecols = (0,1), encoding = "UTF-8", dtype = str, delimiter = ',', skip_header = 1)
    testRef = genfromtxt('/mnt/sdc1/daicwoz/dev_split_Depression_AVEC2017.csv', usecols = (0,1), encoding = "UTF-8", dtype = str, delimiter = ',', skip_header = 1)
    return trainRef, testRef
    
def main():
    trainRef, testRef = extract_info()
    for patient in trainRef:
        print(patient[0], float(patient[1]))


if __name__ == '__main__':
    main()   
