import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform_

class preNet(nn.Module):
    def __init__(self,seq_net,struct_net,fcnn):
        super(preNet,self).__init__()
        self.seq_net = seq_net
        self.struct_net = struct_net
        self.fcnn = fcnn
    def forward(self,X_seq,X_struct):
        out_seq = self.seq_net(X_seq).flatten(1,-1)
        out_struct = self.struct_net(X_struct).flatten(1,-1)
        out = torch.cat((out_seq,out_struct),dim=1)
        out = self.fcnn(out)
        return out
    def kaiming_init(self):
        for module in self.modules():
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                kaiming_uniform_(module.weight,nonlinearity="relu")
        return self

class preNet_conv(nn.Module):
    def __init__(self,mid_channels,out_channels,dropout,device=None):
        super(preNet_conv,self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=mid_channels,
                kernel_size=(5,1)
                )
        self.conv2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(3,1)
                )
        self.bn1 = nn.BatchNorm2d(mid_channels,device=device)
        self.bn2 = nn.BatchNorm2d(out_channels,device=device)
        self.drop = nn.Dropout(dropout)
    def forward(self,X):
        X = X.unsqueeze(1)
        X = F.max_pool2d(F.relu(self.bn1(self.conv1(X))),2)
        X = F.max_pool2d(F.relu(self.bn2(self.conv2(X))),2)
        X = self.drop(X)
        return X

class preNet_linear(nn.Module):
    def __init__(self,in_features,dropout,device=None):
        super(preNet_linear,self).__init__()
        self.linear1 = nn.Linear(in_features,100,device=device)
        self.bn1 = nn.BatchNorm1d(100,device=device)
        self.linear2 = nn.Linear(100,50,device=device)
        self.bn2 = nn.BatchNorm1d(50,device=device)
        self.linear3 = nn.Linear(50,1,device=device)
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(self.bn1(x))
        x = self.linear2(self.drop(x))
        x = F.relu(self.bn2(x))
        x = self.linear3(self.drop(x))
        x = torch.sigmoid(x)
        return x
