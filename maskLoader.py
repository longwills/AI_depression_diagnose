# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:57:14 2020

@author: long
"""

import librosa
import librosa.display
import numpy as np
from numpy import genfromtxt
import torch
import os

def process_mask(session):
    session=str(session)
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_masksA.pt'):        
        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (0,1,2), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant')
        startStamps = torch.from_numpy(np.asfarray(transcript[patient_indice[0],0])).double()
        endStamps = torch.from_numpy(np.asfarray(transcript[patient_indice[0],1])).double()
        y, sr = librosa.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_AUDIO.wav', sr=16000)
        #print(y.shape)
        
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        nstamps = startStamps.shape[0]
        nframes = mel_spect.shape[1]
        timesA = []
        for i in range(0, nframes):
            timesA.append(512/sr * i)
        timesA = torch.DoubleTensor(timesA)
        timesV = np.loadtxt("/mnt/sdc1/daicwoz/"+session+"_P/"+session+"_CLNF_features3D.txt", comments="#", skiprows = (1), delimiter=",", unpack=False)
        timesV = torch.from_numpy(timesV[:,1]).double()
        #print(timesA)
        #print(timesV)
        masksA = []
        masksV = []
        for j in range(nstamps):
            masksA.append((timesA>startStamps[j])* (timesA<endStamps[j]))
            masksV.append((timesV>startStamps[j])* (timesV<endStamps[j]))
        masksA = torch.stack(masksA).double()
        masksV = torch.stack(masksV).double()
        masksA = masksA.T/torch.sum(masksA, 1)
        masksV = masksV.T/torch.sum(masksV, 1)
        torch.save(masksA, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_masksA.pt')
        torch.save(masksV, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_masksV.pt') 
    else:
        masksA = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_masksA.pt')
        masksV = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_masksV.pt')
    return masksA, masksV
    


def main():
    masksA, masksV = process_mask(300)
    print(masksA.shape)
    print(torch.sum(masksA, 0))
    print(masksV.shape)
    print(torch.sum(masksV, 0))



if __name__ == '__main__':
    main()
