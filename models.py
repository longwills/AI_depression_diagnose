# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:20:47 2020

@author: long
"""

import torch
from torch import nn

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
                                      
        super(CausalConv1d, self).__init__()
        
        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size-1)*dilation
        
        # modules:
        self.conv1d = nn.utils.weight_norm(torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=stride,
                                      padding=(kernel_size-1)*dilation,
                                      dilation=dilation), name = 'weight')

    def forward(self, seq):

        conv1d_out = self.conv1d(seq)
        # remove k-1 values from the end:
        return conv1d_out[:,:,:-(self.padding)]

class CFNN(nn.Module):

    def __init__(self):
        super(CFNN, self).__init__()

        self.conv_unit_video = nn.Sequential(
            CausalConv1d(204 ,128, kernel_size=5, stride=1, dilation=1),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=2),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=4),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=8),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=16),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=32),
            nn.ReLU(),
            #nn.Dropout(0.1),
            #CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=64),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=128),
            #nn.ReLU(),
            #nn.Dropout(0.1), 
        )

        self.conv_unit_audio = nn.Sequential(
            CausalConv1d(80 ,128, kernel_size=5, stride=1, dilation=1),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=2),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=4),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=8),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=16),
            nn.ReLU(),
            #nn.Dropout(0.1),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=32),
            nn.ReLU(),
            #nn.Dropout(0.1),
            #CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=64),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=128),
            #nn.ReLU(),
            #nn.Dropout(0.1), 
        )

        self.lstm_unit_transcript = nn.LSTM(300, 64, 3, batch_first=True, bidirectional=True, dropout=0.02)       

        # flatten
        # fully connected (fc) unit
        self.fc_unit = nn.Sequential(
            nn.Linear(128*2, 256),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.Linear(1024, 2048),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.Linear(2048, 4096),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )
        

        #self.criteon = nn.CrossEntropyLoss() #crossEntropyLoss has built-in softmax function
        self.criteon = nn.BCEWithLogitsLoss()

    def forward(self, inputs):

        (batchsz, V, A, L) = inputs

        V = torch.transpose(V, 1, 2)
        Vr = V
        Vr = torch.flip(Vr, [2])
        V = torch.cat([V,Vr], 2) #make it bidirectional
        V = self.conv_unit_video(V)
        V = torch.transpose(V, 1, 2)
        length = V.size(1)
        V = V[: ,0, :]
        V = V.reshape(batchsz, 128)

        A = torch.transpose(A, 1, 2)
        Ar = A
        Ar = torch.flip(Ar, [2])
        A = torch.cat([A,Ar], 2)
        A = self.conv_unit_audio(A)
        A = torch.transpose(A, 1, 2)
        length = A.size(1)
        A = A[: ,0, :]
        A = A.reshape(batchsz, 128)

        L = self.lstm_unit_transcript(L)[0]
        L = L[: ,0, :]
        length = L.size(1)
        L = L.reshape(batchsz, 128)

        '''             
        #print(x1.shape, x2.shape, x3.shape)

        t1 = torch.unsqueeze(V, 1)
        t2 = torch.unsqueeze(A, 1)
        t3 = torch.unsqueeze(L, 1)

        #attention
        M12 = torch.matmul(torch.transpose(t1, 1, 2), t2)
        M23 = torch.matmul(torch.transpose(t2, 1, 2), t3)
        M31 = torch.matmul(torch.transpose(t3, 1, 2), t1)
        O12 = somx(M12)
        O23 = somx(M23)
        O31 = somx(M31)
        v1 = torch.matmul(M12, torch.transpose(t1, 1, 2))
        v2 = torch.matmul(M23, torch.transpose(t3, 1, 2))
        v3 = torch.matmul(M31, torch.transpose(t2, 1, 2))
        v1 = torch.squeeze(torch.transpose(v1, 1, 2), 1)
        v2 = torch.squeeze(torch.transpose(v2, 1, 2), 1)
        v3 = torch.squeeze(torch.transpose(v3, 1, 2), 1)

        x = torch.cat([v1, v2, v3], dim=1)
        '''

        z1 = torch.div(V+A+L, 3)
        z2 = V*A*L

        x = torch.cat([z1, z2], dim=1)
        logits = self.fc_unit(x)

        return logits
