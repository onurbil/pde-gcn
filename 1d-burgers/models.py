
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import Callable


class ANN_Model(nn.Module):
    
    def __init__(self,input_feat, hidden, output_feat):
        super(ANN_Model, self).__init__()


        self.dense1 = nn.Linear(input_feat,hidden)
        self.dense2 = nn.Linear(hidden,hidden)
        self.dense3 = nn.Linear(hidden,hidden)
        self.dense4 = nn.Linear(hidden,hidden)
        self.dense5 = nn.Linear(hidden,hidden)
        self.dense6 = nn.Linear(hidden,hidden)
        self.dense7 = nn.Linear(hidden,hidden)
        self.dense8 = nn.Linear(hidden,output_feat)



    def forward(self, x):
        
        x= self.dense1(x)
        x = torch.tanh(x)

        x= self.dense2(x)
        x = torch.tanh(x)

        x= self.dense3(x)
        x = torch.tanh(x)

        x= self.dense4(x)
        x = torch.tanh(x)

        x= self.dense5(x)
        x = torch.tanh(x)

        x= self.dense6(x)
        x = torch.tanh(x)

        x= self.dense7(x)
        x = torch.tanh(x)
          
        x= self.dense8(x)
        
        return x


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)



class GCNLayer(nn.Module):
    def __init__(self, g:dgl.DGLGraph, in_feats:int, out_feats:int, activation:Callable[[torch.Tensor], torch.Tensor],
                 dropout:int, bias:bool=True):
        super().__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.resconv = nn.Conv1d(in_feats,out_feats,1)

 
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
            
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        
        res=h
        
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization (by square root of src degree):
        h = h * self.g.ndata['norm']
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # normalization (by square root of dst degree):
        h = h * self.g.ndata['norm']
        # bias:
        if self.bias is not None:
            h = h + self.bias
        
        
        res = res.transpose(0, 1)
        res = torch.unsqueeze(res,0)
        res = self.resconv(res)   
        res = torch.squeeze(res,0)
        res = res.transpose(0, 1)
        
        h = h + res
        
        if self.activation:
            h = self.activation(h)

        return h
        


class GCN(nn.Module):
    def __init__(self, g:dgl.DGLGraph, in_feats:int, hidden_feats:int,
                 out_feats:int, activation:Callable, dropout:int, bias=True):
        super().__init__()

        activation = F.tanh
        dropout = 0.
        
        self.gcn11 = GCNLayer(g, in_feats, hidden_feats, activation, dropout, bias=True)

        self.conv10 = nn.Conv1d(hidden_feats,hidden_feats,1)
        self.conv11 = nn.Conv1d(hidden_feats,hidden_feats,1)
        self.conv12 = nn.Conv1d(hidden_feats,hidden_feats,1)

        self.dense = nn.Linear(hidden_feats,out_feats)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)


    def forward(self, x):
        
        x = self.gcn11(x)

        x = x.transpose(0, 1)
        x = torch.unsqueeze(x,0)
        x = self.conv10(x)
        x = F.tanh(x)
        x = self.conv11(x)
        x = F.tanh(x)
        x = self.conv12(x)
        x = F.tanh(x)
        x = torch.squeeze(x,0)
        x = x.transpose(0, 1)
        
        x = self.dense(x)

        return x


class Ensemble(nn.Module):

    def __init__(self, model1, model2, hidden_units):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.fc1 = nn.Linear(2, hidden_units, bias=False)
        self.fc2 = nn.Linear(hidden_units, 1, bias=False)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)

                
    def forward(self, x):
                
        out1 = self.model1(x)
        out2 = self.model2(x)

        out = torch.cat((out1, out2), dim=1)    

        out = self.fc1(out)
        out = F.tanh(out)
        out = self.fc2(out)

        return out