import torch
import torch.nn as nn
from torch.autograd import Variable 
import math
import numpy as np



class rnn_LSTM_variable_multitask_special(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            batch_size,
            device,
            model_name='lstm',
            pretrained=False,
            return_last_only=True,
            in_layer_sizes=None,
            out_layer_sizes=None,
            dropout=0.1,
            orth_gain=1.41,
            ntask=1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.return_last_only = return_last_only
        self.use_embeddings_for_in = None
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_layers = 1
        self.hidden_size = 700
        self.device = device
        self.rnn = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.relu = nn.ReLU(0.25)
        self.dropout = nn.Dropout()
        self.fc_1 = nn.Linear(self.hidden_size, 512)
        self.fc_interim = nn.Linear(512, 512)
        # self.fc_interim2 = nn.Linear(512, 128)
        self.fc_subnets = []
        self.fc_finals = []
        self.ntask = ntask
        for i in range(self.ntask):
            # if i==0:
            #     self.fc_subnets += [nn.Linear(3, 32).to(self.device)]
            #     self.fc_finals += [nn.Linear(32, self.output_dim[i], 1).to(self.device)]
            # else:
            self.fc_subnets += [nn.Linear(512, 128).to(self.device)]
            self.fc_finals += [nn.Linear(128, self.output_dim[i], 1).to(self.device)]

        self.normLayer = nn.LayerNorm(self.hidden_size)
        
        
    def forward(self, x, x_age, output_attentions=False):
        input = x.to(self.device)
        h0 = Variable(torch.nn.init.kaiming_normal_(torch.zeros(self.num_layers, x.data.shape[0], self.hidden_size))).to(self.device) #hidden state
        c0 = Variable(torch.nn.init.kaiming_normal_(torch.zeros(self.num_layers, x.data.shape[0], self.hidden_size))).to(self.device) #internal state
        out_, (hn, cn) = self.rnn(input, (h0, c0))
        # hn = hn.view(-1, self.hidden_size)
        hn = hn[self.num_layers-1, :, :].view(-1, self.hidden_size)
        hn = self.normLayer(hn)
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_interim(out) #first Dense
        out = self.relu(out)
        out = self.dropout(out)
        outs = []
        
        out1_ = self.fc_finals[1](self.fc_subnets[1](out))
        out1 = nn.Sigmoid()(out1_).to(self.device)
        
        out2_ = self.fc_finals[2](self.fc_subnets[2](out))
        out2 = nn.Sigmoid()(out2_).to(self.device)
        
        out3_ = self.fc_finals[3](self.fc_subnets[3](out))
        out3 = nn.Sigmoid()(out3_).to(self.device)
        
        # concat_ADNC = torch.cat([out1_, out2_, out3_], dim=1)
        out0 = self.fc_finals[0](self.fc_subnets[0](out))
        out0 = nn.Sigmoid()(out0).to(self.device)
        
        outs += [out0, out1, out2, out3]
    
        for task in range(4, self.ntask):
            outs += [nn.Sigmoid()(self.fc_finals[task](self.fc_subnets[task](out))).to(self.device)]
        return outs

