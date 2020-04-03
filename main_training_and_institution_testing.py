##################################################################################
#   Example code 2                                                               #
#   Train CNN GRU model on dataset from one hospital and test on second hospital #
#   This should be done for each hospital -> cross validation of results         #
##################################################################################

import torch
from model import *
from dataset import *
from statistics import *
from torch.utils.data import DataLoader


# Create training and testing datasets
# remove powerline noise class since Europe and USA use different powerline frequencies(50Hz and 60Hz respectively)
# network is not able to generalize to data from different class which was not used in training set
# if additional hospital is used with same powerline frequency as training set
# then powerline noise class should not be removed
dataset_fnusa_train= Dataset('./DATASET_FNUSA/').remove_powerline_noise_class()
dataset_mayo_test = Dataset('./DATASET_MAYO/').remove_powerline_noise_class()

NWORKERS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN = DataLoader(dataset=dataset_fnusa_train,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)

TEST = DataLoader(dataset=dataset_mayo_test,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)




if __name__ == "__main__":
    # we training model for classification to 3 classes i.e. normal(physiological), noise, pathological
    model = NN(NOUT=3).to(DEVICE)
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

        model.eval()
        for i,(x,t) in enumerate(TEST):
            x = x.to(DEVICE).float()
            t = t.to(DEVICE).long()
            y = model(x)
            statistics.append(target=t,logits=y[:,-1,:])
        statistics.evaluate()


