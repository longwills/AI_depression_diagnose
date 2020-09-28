import torch
from torch import nn, optim

import nltk
nltk.download('punkt')

from torch.utils.data import DataLoader
import maskLoader
import audioLoader
import videoLoader
import transcriptLoader
import metaInfoLoader
import models
from models import CFNN

import matplotlib.pyplot as plt

def normalize_tensor(inputT):    
    mean = torch.mean(inputT)
    dev = torch.std(inputT)
    outputT = (inputT-mean)/dev
    return outputT

def normalize_tensors(inputList):  
    #normalize tensors in the list with the mean and deviation of the first element 
    #print(inputList[0].shape) 
    mean = torch.mean(inputList[0])
    dev = torch.std(inputList[0])
    outputList = []
    for inputT in inputList:
        #print(inputT.shape)
        outputT = (inputT-mean)/dev
        outputList.append(outputT)
    return outputList

def pad_pack(x, x_len):
    x_len_sorted, idx_sort = torch.sort(x_len, descending=True)  #np.sort(x_len)[::-1], np.argsort(-x_len)
    idx_unsort = torch.argsort(idx_sort)
    x = torch.index_select(x, 0, idx_sort)
    x_packed = nn.utils.rnn.pack_padded_sequence(x, x_len_sorted, batch_first = True)
    
    return x_packed

def read_sample(Reference):

    label_list = []
    V_list = []
    A_list = []
    L_list = []

    V_map = dict()
    A_map = dict()
    L_map = dict()
   
    for patient in Reference:
        session = patient[0]
        print("preparing: ", session)
        if int(session) > 500:
            break
        if int(session) == 396 or int(session) == 432 or int(session) == 367:  #problematic session
            continue

        V = videoLoader.process_video_segments(session)
        A = audioLoader.process_audio_segments(session)
        L = transcriptLoader.process_transcript_segments(session)
        
        label = float(patient[1])
        label = torch.DoubleTensor([label])
        label_sublist = [label]*len(L)

        V_list = V_list + V
        A_list = A_list + A
        L_list = L_list + L
        label_list = label_list + label_sublist

        

    print(len(label_list), len(V_list), len(A_list), len(L_list))
 

    return (label_list, V_list, A_list, L_list)

def prepare_sample():

    #get meta information reference like train/test separation and labels
    trainRef, testRef = metaInfoLoader.extract_info()
 
    #read input V, A, L and labels
    daic_train = read_sample(trainRef)
    daic_test = read_sample(testRef)
    trainLen = len(daic_train[0])
    testLen = len(daic_test[0])
    
    label_list = daic_train[0] + daic_test[0]
    V_list = daic_train[1] + daic_test[1]
    A_list = daic_train[2] + daic_test[2]
    L_list = daic_train[3] + daic_test[3]

    #normalize V, A, L (before padding)
    V_list = normalize_tensors(V_list)
    A_list = normalize_tensors(A_list)
    L_list = normalize_tensors(L_list)

    #pad sequences of patients to make them have same length
    V_tensor = torch.nn.utils.rnn.pad_sequence(V_list, batch_first=True )
    A_tensor = torch.nn.utils.rnn.pad_sequence(A_list, batch_first=True )
    L_tensor = torch.nn.utils.rnn.pad_sequence(L_list, batch_first=True )
    print(V_tensor.shape)
    print(A_tensor.shape)
    print(L_tensor.shape)

    #segment sequence to make it shorter
    V_tensor = V_tensor[:,0:500,:]
    A_tensor = A_tensor[:,0:500,:]
    L_tensor = L_tensor[:,0:10,:]
   
    #make final output list
    sample = []   
    nPatient = len(label_list)
    for ip in range(0, nPatient):
        pair = (V_tensor[ip], A_tensor[ip], L_tensor[ip], label_list[ip])
        sample.append(pair)
    daic_train = sample[:trainLen]
    daic_test = sample[-testLen:]

    return daic_train, daic_test

    
    

def main():
    
    print("start running")
 
    #read V, A, L from daicwoz dataset into lists
    daic_train, daic_test = prepare_sample()       
    print('train length:', len(daic_train), 'test length:', len(daic_test))

    #make datasets with given batch size
    batchsz_train = 128
    batchsz_test = 128    
    daic_train = DataLoader(daic_train, batch_size=batchsz_train, shuffle=True, drop_last=True)
    daic_test = DataLoader(daic_test, batch_size=batchsz_test, shuffle=True, drop_last=True)
    V, A, L, label = iter(daic_train).next()
    #print('V, A, L shape:', V.shape, A.shape, L.shape, 'label:', label.shape)

    #initialize model
    device = torch.device('cuda')
    model = CFNN().to(device)
    model = model.double()
    criteon = nn.BCEWithLogitsLoss().to(device)
    sm = nn.Sigmoid().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(model)

    #training iteration
    for epoch in range(200):
        model.train()
        for batchidx, (V, A, L, label) in enumerate(daic_train):
            V, A, L, label = V.to(device), A.to(device), L.to(device), label.to(device)

            inputs = (batchsz_train, V, A, L)
            label = torch.squeeze(label)
            logits = model(inputs)
            logits = torch.squeeze(logits)
            loss = criteon(logits,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del inputs

        print(epoch, loss.item())
        pred = sm(logits)


        model.eval()    
        with torch.no_grad():
               # test
            total_correct = 0
            total_num = 0
            total_positive = 0
            total_true = 0
            total_true_positive = 0
            for V, A, L, label in daic_test:
                V, A, L, label = V.to(device), A.to(device), L.to(device), label.to(device)

                inputs = (batchsz_test, V, A, L)
                label = torch.squeeze(label)               
                logits = model(inputs)
                logits = torch.squeeze(logits)
                                
                pred = sm(logits)

                total_correct += torch.le(torch.abs(pred-label), 0.5).float().sum().item()
                total_num += V.size(0)
                total_positive += torch.gt(pred, 0.5).float().sum().item()
                total_true += torch.gt(label, 0.5).float().sum().item()
                total_true_positive += (torch.gt(pred, 0.5) * torch.gt(label, 0.5)).float().sum().item()

                del inputs

            acc = total_correct / total_num
            print("test accuracy: ", acc, " precision: ", total_true_positive,"/",total_positive, " recall: ", total_true_positive,"/",total_true)

    #save model    
    torch.save(model, "model/model.pt")

if __name__ == '__main__':
    main()

