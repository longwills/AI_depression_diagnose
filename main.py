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

def normalize_features(inputT):    
    mean = torch.mean(inputT)
    dev = torch.std(inputT)
    outputT = (inputT-mean)/dev
    return outputT

def normalize_tensors(inputList):  
    #normalize tensors in the list with the mean and deviation of the first element  
    mean = torch.mean(inputList[0])
    dev = torch.std(inputList[0])
    outputList = []
    for inputT in inputList:
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
    features_list = []
    #V_list = []
    #A_list = []
    #L_list = []
   
    for patient in Reference:
        session = patient[0]
        print("preparing: ", session)
        if int(session) > 310:
            break
        if int(session) == 396 or int(session) == 432 or int(session) == 367:  #problematic session
            continue
        if int(session) == 420 or int(session) == 434 or int(session) == 402:  #needs investigation
            continue
        label = float(patient[1])
        label = torch.DoubleTensor([label])
        label_list.append(label)

        V = videoLoader.process_video(session)
        V = torch.cat([V[2], V[3], V[4]], dim=1) #combine 3-d video features
        A = audioLoader.process_audio(session)
        L = transcriptLoader.process_transcript(session)
        masksA, masksV = maskLoader.process_mask(session)
        #print(torch.sum(masksA), torch.sum(masksV))
        plt.plot(A)
        
        #average video and audio frames within each sentence
        V = torch.matmul(masksV.T, V)
        A = torch.matmul(masksA.T, A)
        plt.plot(A)
        
        features = torch.cat([V, A, L], dim=1) #combine all features
        features_list.append(features)
        #V_list.append(V)
        #A_list.append(A)
        #L_list.append(L)
 

    return (label_list, features_list)

def prepare_sample():

    #get meta information reference like train/test separation and labels
    trainRef, testRef = metaInfoLoader.extract_info()
 
    #read input features and labels
    daic_train = read_sample(trainRef)
    daic_test = read_sample(testRef)
    trainLen = len(daic_train[0])
    testLen = len(daic_test[0])
    
    label_list = daic_train[0] + daic_test[0]
    features_list = daic_train[1] + daic_test[1]

    #normalize features (before padding)
    features_list = normalize_tensors(features_list)

    #pad sequences of patients to make them have same length
    features_tensor = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True )

    #segment sequence to make it shorter
    #features_tensor = features_tensor[:,0:100,:]

    print("features_tensor: ", features_tensor)

    
    #make final output list
    sample = []   
    nPatient = len(label_list)
    for ip in range(0, nPatient):
        pair = (features_tensor[ip], label_list[ip])
        sample.append(pair)
    daic_train = sample[:trainLen]
    daic_test = sample[-testLen:]

    return daic_train, daic_test

    
    

def main():
    
    print("start running")
 
    #read features from daicwoz dataset into lists
    daic_train, daic_test = prepare_sample()       
    print('train length:', len(daic_train), 'test length:', len(daic_test))

    #make datasets with given batch size
    batchsz_train = 1
    batchsz_test = 5    
    daic_train = DataLoader(daic_train, batch_size=batchsz_train, shuffle=True, drop_last=True)
    daic_test = DataLoader(daic_test, batch_size=batchsz_test, shuffle=True, drop_last=True)
    features, label = iter(daic_train).next()
    print('features shape:', features.shape, 'label:', label.shape)

    #initialize model
    device = torch.device('cuda')
    model = CFNN().to(device)
    model = model.double()
    criteon = nn.BCEWithLogitsLoss().to(device)
    sm = nn.Sigmoid().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    print(model)

    #training iteration
    for epoch in range(3):
        model.train()
        for batchidx, (features, label) in enumerate(daic_train):
            features, label = features.to(device), label.to(device)

            inputs = (batchsz_train, features)
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
            for features, label in daic_test:
                features, label = features.to(device), label.to(device)

                inputs = (batchsz_test, features)
                label = torch.squeeze(label)               
                logits = model(inputs)
                logits = torch.squeeze(logits)
                                
                pred = sm(logits)

                total_correct += torch.le(torch.abs(pred-label), 0.5).float().sum().item()
                total_num += features.size(0)
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

