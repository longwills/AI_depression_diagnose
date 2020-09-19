# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:54:04 2020

@author: long
"""

import numpy as np
import torch
import os

def load_2D(filename):
    lines = np.loadtxt(filename, comments="#", skiprows = (1), delimiter=",", unpack=False)
    lines = np.delete(lines, [0,1,2,3], axis=1)
    lines_x = lines[:,0:68]
    lines_y = lines[:,68:136]
    tensor_x = torch.from_numpy(lines_x)
    tensor_y = torch.from_numpy(lines_y)
    tensor_x = tensor_x.double()
    tensor_y = tensor_y.double()
    print("tensor_x_size: ",tensor_x.shape," tensor_y_size: ",tensor_y.shape)
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
    print("tensor_x_size: ",tensor_x.shape," tensor_y_size: ",tensor_y.shape," tensor_z_size: ",tensor_z.shape)
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

def main():
    tensor_featuresCombined = process_video(300)
    print("tensor: ", tensor_featuresCombined[0].shape, tensor_featuresCombined[1].shape, tensor_featuresCombined[2].shape, tensor_featuresCombined[3].shape, tensor_featuresCombined[4].shape)


if __name__ == '__main__':
    main()
