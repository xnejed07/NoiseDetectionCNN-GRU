###############################################################################
#   Example code 1                                                            #
#   Train CNN GRU model on dataset from two reviewers                         #
#   Test model on dataset from third reviewer                                 #
#   This should be done for each reviewer -> cross validation of results      #
###############################################################################

import torch
from model import *
from dataset import *
from statistics import *
from torch.utils.data import DataLoader

# create training and validation dataset
# split_reviewer(reviewer_id) function split dataset by reviewers
# results should be evaluated for all reviewers i.e. 1,2,3
dataset_fnusa_train,dataset_fnusa_valid = Dataset('./DATASET_FNUSA/').split_reviewer(1)

NWORKERS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN = DataLoader(dataset=dataset_fnusa_train,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)

VALID = DataLoader(dataset=dataset_fnusa_valid,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)




if __name__ == "__main__":
    model = NN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()
    statistics = Statistics()

    for epoch in range(5):
        model.train()
        for i,(x,t) in enumerate(TRAIN):
            optimizer.zero_grad()
            x = x.to(DEVICE).float()
            t = t.to(DEVICE).long()
            y = model(x)
            J = loss(input=y[:,-1,:],target=t)
            J.backward()
            optimizer.step()

            if i%50==0:
                print('EPOCH:{}\tITER:{}\tLOSS:{}'.format(str(epoch).zfill(2),
                                                          str(i).zfill(5),
                                                          J.data.cpu().numpy()))

        # evaluate results for validation test
        model.eval()
        for i,(x,t) in enumerate(VALID):
            x = x.to(DEVICE).float()
            t = t.to(DEVICE).long()
            y = model(x)
            statistics.append(target=t,logits=y[:,-1,:])
        statistics.evaluate()


