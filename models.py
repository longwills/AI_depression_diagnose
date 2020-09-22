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
       
        self.conv_unit_vision = nn.Sequential(
            CausalConv1d(204 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),  
        )
        
        self.conv_unit_voice = nn.Sequential(
            CausalConv1d(80 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(4, stride=4),
            nn.ReLU(),
            nn.Dropout(0.5),
        )        

        self.conv_unit_transcript = nn.Sequential(
            CausalConv1d(300 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=1),
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
        ) 

        self.conv_unit = nn.Sequential(
            CausalConv1d(284 ,128, kernel_size=5, stride=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=8),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=16),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=64),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            CausalConv1d(128 ,128, kernel_size=5, stride=1, dilation=512),
            nn.ReLU(),
            nn.Dropout(0.5),
 
        ) 


        # flatten
        # fully connected (fc) unit
        self.fc_unit = nn.Sequential(
            nn.Linear(2*128, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1)
        )
        

        #self.criteon = nn.CrossEntropyLoss() #crossEntropyLoss has built-in softmax function
        self.criteon = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        
        somx = nn.Softmax(dim=1)


        (batchsz, x1, x2, x3) = inputs
        
        
        x1 = torch.transpose(x1, 1, 2)
        x1 = self.conv_unit_vision(x1)
        x1 = torch.transpose(x1, 1, 2)
        length = x1.size(1)
        #x1 = x1[: ,length//2, :]
        x1 = torch.sum(x1, 1).squeeze(1)/length
        
        x2 = torch.transpose(x2, 1, 2)
        x2 = self.conv_unit_voice(x2)
        x2 = torch.transpose(x2, 1, 2)
        length = x2.size(1)
        #x2 = x2[: ,length//2, :]
        x2 = torch.sum(x2, 1).squeeze(1)/length

        x3 = torch.transpose(x3, 1, 2)
        x3 = self.conv_unit_transcript(x3)
        x3 = torch.transpose(x3, 1, 2)
        length = x3.size(1)
        #x3 = x3[: ,length//2, :]
        x3 = torch.sum(x3, 1).squeeze(1)/length
        
        
        #print(x1.shape, x2.shape, x3.shape)
        x1 = x1.reshape(batchsz, 1*128)
        x2 = x2.reshape(batchsz, 1*128)
        x3 = x3.reshape(batchsz, 1*128)
        t1 = torch.unsqueeze(x1, 1)
        t2 = torch.unsqueeze(x2, 1)
        t3 = torch.unsqueeze(x3, 1)

        z1 = torch.div(x1+x2+x3, 3)
        z2 = x1*x2*x3

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

        #x = torch.cat([v1, v2, v3], dim=1)
        x = torch.cat([z1, z2], dim=1)

        logits = self.fc_unit(x)

        return logits
