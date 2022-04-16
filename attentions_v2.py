import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init



class Attention(nn.Module):
    def __init__(self, input_size, output_size, cuda=False):
        super(Attention,self).__init__()

        self.cuda = cuda
        self.bilinear_layer = nn.Bilinear(input_size,input_size,1)
        self.softmax = nn.Softmax(dim=1)
        if self.cuda:
            self.bilinear_layer.cuda()
            self.softmax.cuda()
    def forward(self, Ws):
        """
        Measuring relations between all the dimensions
        """
    


        num_dims = len(Ws)

        attention_matrix = torch.empty((num_dims,num_dims),dtype=torch.float)
        if self.cuda:
            attention_matrix = attention_matrix.cuda()
        for i,wi in enumerate(Ws):
            for j,wj in enumerate(Ws):
                attention_matrix[i,j] = torch.sum(self.bilinear_layer(wi,wj))
        
        attention_matrix_softmax = self.softmax(attention_matrix)


        return attention_matrix_softmax