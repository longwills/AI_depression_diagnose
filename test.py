import torch
from torch import nn, optim

import nltk

from torch.utils.data import DataLoader
import metaInfoLoader
from main import prepare_sample

def main():
    
    print("start running")
    daic_train, daic_test = prepare_sample()
             
    batchsz = 1
    
    daic_test = DataLoader(daic_test, batch_size=1, shuffle=True, drop_last=True)

    
    device = torch.device('cuda')
    model = torch.load("model/model.pt").to(device)
    model = model.double()

    sm = nn.Sigmoid().to(device)

    print(model)

    model.eval()    
    with torch.no_grad():
        # test
        total_correct = 0
        total_num = 0
        total_positive = 0
        total_true = 0
        total_true_positive = 0
        for x1, x2, x3, label in daic_test:
            x1, x2, x3, label = x1.to(device), x2.to(device), x3.to(device), label.to(device)
            label = torch.squeeze(label) 
            inputs = (batchsz, x1, x2, x3)              
            logits = model(inputs)
            logits = torch.squeeze(logits)
                
            pred = sm(logits)
            print("pred: ", pred, " label: ", label)
                
            total_correct += torch.le(torch.abs(pred-label), 0.5).float().sum().item()
            total_num += x1.size(0)
            total_positive += torch.gt(pred, 0.5).float().sum().item()
            total_true += torch.gt(label, 0.5).float().sum().item()
            total_true_positive += (torch.gt(pred, 0.5) and torch.gt(label, 0.5)).float().sum().item()
                
            del inputs

        print("total_positive: ", total_positive, "total_true: ", total_true, "total_true_positive: ", total_true_positive)
        acc = total_correct / total_num
        precision = total_true_positive / total_positive
        recall = total_true_positive / total_true
        F1_score = 2./(1./precision + 1./recall)
        print("test accuracy: ", acc, "precision: ", precision, "recall: ", recall, "F1_score: ", F1_score)
        


if __name__ == '__main__':
    main()

