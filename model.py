import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self,NFILT=256,NOUT=4):
        super(NN,self).__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(200,3),padding=(0,1),bias=False)
        self.bn0 = nn.BatchNorm2d(NFILT)
        self.gru = nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(128,NOUT)



    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze().permute(0,2,1)
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc1(x)
        return x