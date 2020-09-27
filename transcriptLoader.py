# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:57:14 2020

@author: long
"""


import numpy as np
from numpy import genfromtxt
import torch
import sentenceEmbedding
import wordEmbedding
import os

def process_transcript(session):
    session=str(session)
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT.pt'):        
        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (2,3), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant')
        transcript = transcript[patient_indice[0],1]
        #use fastText to do word embedding
        embeddingsS = sentenceEmbedding.embed(transcript)
        embeddingsS = embeddingsS.double()
        torch.save(embeddingsS, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT.pt') 
    else:
        embeddingsS = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT.pt')
    return embeddingsS
    
def process_transcript_segments(session):
    session=str(session)
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT_seg.pt'):        
        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (2,3), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant')
        transcript = transcript[patient_indice[0],1]
        #use fastText to do word embedding
        
        embeddingsW = []
        for sentence in transcript:
            sentence = [sentence]
            embeddingsW.append(wordEmbedding.embed(sentence).double())
        
        torch.save(embeddingsW, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT_seg.pt')

    else:
        embeddingsW = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT_seg.pt')
    return embeddingsW

def main():
    L = process_transcript(300)
    print("embeddings shape: ", L.shape)

    L_list = process_transcript_segments(476)
    print(len(L_list), L_list[0].shape, L_list[-1].shape)


if __name__ == '__main__':
    main()
