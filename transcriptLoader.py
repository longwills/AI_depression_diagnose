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
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT_s.pt'):        
        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (2,3), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant')
        transcript = transcript[patient_indice[0],1]
        #use fastText to do word embedding
        embeddings = sentenceEmbedding.embed(transcript)
        #embeddings = wordEmbedding.embed(transcript)
        embeddings = embeddings.double()
        torch.save(embeddings, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT_s.pt') 
    else:
        embeddings = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_TRANSCRIPT_s.pt')
    return embeddings
    


def main():
    L = process_transcript(301)
    print("embeddings shape: ", L.shape)


if __name__ == '__main__':
    main()
