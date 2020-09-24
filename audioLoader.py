import librosa
import librosa.display
import numpy as np
import torch
import os

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

def main():
    tensor_featuresCombined = process_audio(300)
    print("tensor: ", tensor_featuresCombined.shape)
    print("element: ", tensor_featuresCombined[3, 2])


if __name__ == '__main__':
    main()

