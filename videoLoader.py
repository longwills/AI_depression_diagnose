# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:54:04 2020

@author: long
"""

import numpy as np
import torch
import os
from numpy import genfromtxt

import matplotlib.pyplot as plt
import maskLoader

def load_2D(filename):
    lines = np.loadtxt(filename, comments="#", skiprows = (1), delimiter=",", unpack=False)
    lines = np.delete(lines, [0,1,2,3], axis=1)
    lines_x = lines[:,0:68]
    lines_y = lines[:,68:136]
    tensor_x = torch.from_numpy(lines_x)
    tensor_y = torch.from_numpy(lines_y)
    tensor_x = tensor_x.double()
    tensor_y = tensor_y.double()
    #print("tensor_x_size: ",tensor_x.shape," tensor_y_size: ",tensor_y.shape)
    return tensor_x, tensor_y

def load_3D(filename):
    lines = np.loadtxt(filename, comments="#", skiprows = (1), delimiter=",", unpack=False)
    lines = np.delete(lines, [0,1,2,3], axis=1)
    lines_x = lines[:,0:68]
    lines_y = lines[:,68:136]
    lines_z = lines[:,136:204]
    tensor_x = torch.from_numpy(lines_x)
    tensor_y = torch.from_numpy(lines_y)
    tensor_z = torch.from_numpy(lines_z)
    tensor_x = tensor_x.double()
    tensor_y = tensor_y.double()
    tensor_z = tensor_z.double()
    #print("tensor_x_size: ",tensor_x.shape," tensor_y_size: ",tensor_y.shape," tensor_z_size: ",tensor_z.shape)
    return tensor_x, tensor_y, tensor_z

def process_video(session):
    session=str(session)
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_x.pt'):
        tensor_V2D_x, tensor_V2D_y = load_2D("/mnt/sdc1/daicwoz/"+session+"_P/"+session+"_CLNF_features.txt")
        tensor_V3D_x, tensor_V3D_y, tensor_V3D_z = load_3D("/mnt/sdc1/daicwoz/"+session+"_P/"+session+"_CLNF_features3D.txt")

        torch.save(tensor_V2D_x, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V2D_x.pt')
        torch.save(tensor_V2D_y, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V2D_y.pt')
        torch.save(tensor_V3D_x, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_x.pt')
        torch.save(tensor_V3D_y, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_y.pt')
        torch.save(tensor_V3D_z, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_z.pt')
    else:
        tensor_V2D_x = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V2D_x.pt')
        tensor_V2D_y = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V2D_y.pt')
        tensor_V3D_x = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_x.pt')
        tensor_V3D_y = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_y.pt')
        tensor_V3D_z = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_z.pt')
    
    return tensor_V2D_x, tensor_V2D_y, tensor_V3D_x, tensor_V3D_y, tensor_V3D_z

def process_video_segments(session):
    session=str(session)
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_seg.pt'):

        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (0,1,2), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant')
        startStamps = torch.from_numpy(np.asfarray(transcript[patient_indice[0],0])).double()
        endStamps = torch.from_numpy(np.asfarray(transcript[patient_indice[0],1])).double()
        nstamps = startStamps.shape[0]

        tensor_V3D_x, tensor_V3D_y, tensor_V3D_z = load_3D("/mnt/sdc1/daicwoz/"+session+"_P/"+session+"_CLNF_features3D.txt")
        tensor_V3D = torch.cat([tensor_V3D_x, tensor_V3D_y, tensor_V3D_z], dim=1)
        nframes = tensor_V3D.shape[0]

        V_list = []
        for j in range(nstamps):
            V = []
            for i in range(nframes):
                if (i*1/30 > startStamps[j] and i*1/30 < endStamps[j]):
                    V.append(tensor_V3D[i])
            if len(V) > 0:
                V = torch.stack(V)
            else:
                V = torch.zeros(1, 204)
            V_list.append(V)

        torch.save(V_list, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_seg.pt')

    else:
        V_list = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_V3D_seg.pt')

    
    return V_list

def main():

    V = process_video(300)
    #masksA, masksV = maskLoader.process_mask(300)
    
    plt.plot(V[2])
    print("tensor: ", V[2].shape)

    #Vp = torch.matmul(masksV.T, V[2])
    #plt.plot(Vp)
    #print("tensor: ", Vp.shape)

    V_list = process_video_segments(476)
    #print(len(V_list), V_list[0].shape, V_list[-1].shape)
    for V in V_list:
        print(V.shape)


if __name__ == '__main__':
    main()
