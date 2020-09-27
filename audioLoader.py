import librosa
import librosa.display
import numpy as np
import torch
import os
from numpy import genfromtxt

import matplotlib.pyplot as plt
import maskLoader

def process_audio(session):

    session = str(session)
    
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_AUDIO.pt'):
        y, sr = librosa.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_AUDIO.wav', sr=16000)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        tensor_AU = torch.DoubleTensor(mel_spect)
        #tensor_AU = torch.log(tensor_AU)
        tensor_AU = torch.transpose(tensor_AU, 0, 1)
        print("tensorsize: "+str(tensor_AU.shape))
        torch.save(tensor_AU, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_AUDIO.pt')
    else:
        tensor_AU = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_AUDIO.pt')

    return tensor_AU

def process_audio_segments(session):

    session = str(session)
    
    if not os.path.isfile('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_AUDIO_seg.pt'):
        transcript = genfromtxt('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_TRANSCRIPT.csv', usecols = (0,1,2), encoding = "UTF-8", dtype = str, delimiter = '\t', skip_header = 1)
        patient_indice = np.where(transcript == 'Participant')
        startStamps = torch.from_numpy(np.asfarray(transcript[patient_indice[0],0])).double()
        endStamps = torch.from_numpy(np.asfarray(transcript[patient_indice[0],1])).double()
        nstamps = startStamps.shape[0]

        y, sr = librosa.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_AUDIO.wav', sr=16000)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        tensor_AU = torch.DoubleTensor(mel_spect)
        #tensor_AU = torch.log(tensor_AU)
        tensor_AU = torch.transpose(tensor_AU, 0, 1)
        nframes = tensor_AU.shape[0]
        print("tensorsize: "+str(tensor_AU.shape))

        A_list = []
        for j in range(nstamps):
            A = []
            for i in range(nframes):
                if (i*512/16000 > startStamps[j] and i*512/16000 < endStamps[j]):
                    A.append(tensor_AU[i])
            if len(A) > 0:
                A = torch.stack(A)
            else:
                A = torch.zeros(1, 80)
            A_list.append(A)


        torch.save(A_list, '/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_AUDIO_seg.pt')
    else:
        A_list = torch.load('/mnt/sdc1/daicwoz/'+session+'_P/'+session+'_encoded_AUDIO_seg.pt')

    return A_list

def main():
    A = process_audio(300)
    #masksA, masksV = maskLoader.process_mask(300)
    
    plt.plot(A)
    print("tensor: ", A.shape)

    #A = torch.matmul(masksA.T, A)
    #plt.plot(A)
    #print("tensor: ", A.shape)

    A_list = process_audio_segments(300)
    print(len(A_list), A_list[0].shape, A_list[-1].shape)




if __name__ == '__main__':
    main()

