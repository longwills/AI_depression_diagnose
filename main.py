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
    V2D_x_list = []
    V2D_y_list = []
    V3D_x_list = []
    V3D_y_list = []
    V3D_z_list = []
    A_list = []
    L_list = []
   
    for patient in Reference:
        session = patient[0]
        print("preparing: ", session)
        if int(session) > 310:
            break
        if int(session) == 396 or int(session) == 432 or int(session) == 367:  #problematic session
            continue
        label = float(patient[1])
        label = torch.DoubleTensor([label])
        label_list.append(label)

        V = videoLoader.process_video(session)
        A = audioLoader.process_audio(session)
        L = transcriptLoader.process_transcript(session)
        masksA, masksV = maskLoader.process_mask(session)
        print(A.shape, V[0].shape, L.shape)

        #average video and audio frames within each sentence
        V = torch.matmul(masksV.T, V[0]), torch.matmul(masksV.T, V[1]), torch.matmul(masksV.T, V[2]), torch.matmul(masksV.T, V[3]), torch.matmul(masksV.T, V[4])
        A = torch.matmul(masksA.T, A)
        print(A.shape, V[0].shape, L.shape)

        V2D_x_list.append(V[0])
        V2D_y_list.append(V[1])
        V3D_x_list.append(V[2])
        V3D_y_list.append(V[3])
        V3D_z_list.append(V[4])
        A_list.append(A)
        L_list.append(L)
 

    return (label_list, V2D_x_list, V2D_y_list, V3D_x_list, V3D_y_list, V3D_z_list, A_list, L_list, masksA, masksV)

def prepare_sample():

    #get meta information reference like train/test separation and labels
    trainRef, testRef = metaInfoLoader.extract_info()
 
    #read input features and labels
    daic_train = read_sample(trainRef)
    daic_test = read_sample(testRef)
    trainLen = len(daic_train[0])
    testLen = len(daic_test[0])
    
    label_list = daic_train[0] + daic_test[0]
    V2D_x_list = daic_train[1] + daic_test[1]
    V2D_y_list = daic_train[2] + daic_test[2]
    V3D_x_list = daic_train[3] + daic_test[3]
    V3D_y_list = daic_train[4] + daic_test[4]
    V3D_z_list = daic_train[5] + daic_test[5]
    A_list = daic_train[6] + daic_test[6]
    L_list = daic_train[7] + daic_test[7]
    masksA = daic_train[8] + daic_test[8]
    masksV = daic_train[9] + daic_test[9]

    #normalize features (before padding)
    V2D_x_list = normalize_tensors(V2D_x_list)
    V2D_y_list = normalize_tensors(V2D_y_list)
    V3D_x_list = normalize_tensors(V3D_x_list)
    V3D_y_list = normalize_tensors(V3D_y_list)
    V3D_z_list = normalize_tensors(V3D_z_list)
    A_list = normalize_tensors(A_list)
    L_list = normalize_tensors(L_list)

    #pad sequences of patients to make them have same length
    V2D_x_tensor = torch.nn.utils.rnn.pad_sequence(V2D_x_list, batch_first=True )
    V2D_y_tensor = torch.nn.utils.rnn.pad_sequence(V2D_y_list, batch_first=True )
    V3D_x_tensor = torch.nn.utils.rnn.pad_sequence(V3D_x_list, batch_first=True )
    V3D_y_tensor = torch.nn.utils.rnn.pad_sequence(V3D_y_list, batch_first=True )
    V3D_z_tensor = torch.nn.utils.rnn.pad_sequence(V3D_z_list, batch_first=True )
    A_tensor = torch.nn.utils.rnn.pad_sequence(A_list, batch_first=True )
    L_tensor = torch.nn.utils.rnn.pad_sequence(L_list, batch_first=True )

    #combine dimensions into one vector for vision
    V_tensor = torch.cat([V3D_x_tensor, V3D_y_tensor, V3D_z_tensor], dim=2)

    #segment sequence to make it shorter (to be optimized)
    V_tensor = V_tensor[:,0:4096,:]
    A_tensor = A_tensor[:,0:4096,:]
    L_tensor = L_tensor[:,0:512,:]

    print("V_tensor: ", V_tensor)
    print("A_tensor: ", A_tensor)
    print("L_tensor: ", L_tensor)
    
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
 
    #read features from daicwoz dataset into lists
    daic_train, daic_test = prepare_sample()       
    print('train length:', len(daic_train), 'test length:', len(daic_test))

    #make datasets with given batch size
    batchsz_train = 16
    batchsz_test = 5    
    daic_train = DataLoader(daic_train, batch_size=batchsz_train, shuffle=True, drop_last=True)
    daic_test = DataLoader(daic_test, batch_size=batchsz_test, shuffle=True, drop_last=True)
    V, A, L, label = iter(daic_train).next()
    print('vision shape:', V.shape, 'voice shape:', A.shape, 'transcript:', L.shape, 'label:', label.shape)

    #initialize model
    device = torch.device('cuda')
    model = CFNN().to(device)
    model = model.double()
    criteon = nn.BCEWithLogitsLoss().to(device)
    sm = nn.Sigmoid().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    print(model)

    #training iteration
    for epoch in range(500):
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

