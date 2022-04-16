import torch
import torch.nn as nn
from torch.autograd import Variable

from aggregators_sparse import MeanAggregator
from attentions_v2 import Attention
import time


### self-attentive model
class PointWiseFeedforward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedforward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SelfAttentiveModel(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SelfAttentiveModel, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.attention_layers = torch.nn.MultiheadAttention(self.hidden_units,
                                                            self.num_heads,
                                                            self.dropout_rate)

        self.fwd_layer = PointWiseFeedforward(hidden_units, dropout_rate)

        self.attention_layernorms = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.forward_layernorms = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

    def forward(self, node_embeddings):
        node_embeddings = torch.transpose(node_embeddings, 0, 1)
        Q = self.attention_layernorms(node_embeddings)
        mha_output, _ = self.attention_layers(Q, node_embeddings, node_embeddings)
        mha_output = Q + mha_output
        mha_output = torch.transpose(mha_output, 0, 1)
        mha_output = self.forward_layernorms(mha_output)
        node_output = self.fwd_layer(mha_output)
        node_output = self.last_layernorm(node_output)
        return node_output


"""
Modules for aggregating embeddings from other dimensions
"""


class DimAggregator(nn.Module):
    """
    Aggregate a node's embedding from the other dimensions

    """

    def __init__(self, num_dims, cuda=False, gcn=False, input_size=5, output_size=3, dropout_rate=0, alpha=0.5,
                 add_self=0):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(DimAggregator, self).__init__()
        self.cuda = cuda
        self.gcn = gcn
        self.num_dims = num_dims
        self.input_size = input_size
        self.output_size = output_size

        if self.cuda:
            self.linear_layers_for_dims = nn.ModuleList(
                [nn.Linear(input_size, output_size, bias=False).cuda() for i in range(num_dims)])
        else:
            self.linear_layers_for_dims = nn.ModuleList(
                [nn.Linear(input_size, output_size, bias=False) for i in range(num_dims)])
        self.mean_agg = MeanAggregator(self.cuda, self.gcn)
        self.attention_layer = Attention(input_size, output_size, self.cuda)
        if self.cuda:
            self.act = nn.ELU().cuda()
        else:
            self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        if self.cuda:
            self.alpha = torch.tensor(alpha).cuda().item()
        else:
            self.alpha = alpha
        self.add_self = add_self
        ### self-attentive layer

        self.r1_matrix = nn.Parameter(torch.nn.init.normal_(torch.empty(self.input_size, self.input_size)))
        self.r2_matrix = nn.Parameter(torch.nn.init.normal_(torch.empty(self.input_size, self.input_size)))
        self.r3_matrix = nn.Parameter(torch.nn.init.normal_(torch.empty(self.input_size, self.input_size)))
        # self.r_all = nn.Parameter(torch.nn.init.normal_(torch.empty(self.input_size, self.input_size)))
        self.self_attention_layer = SelfAttentiveModel(self.input_size, 4, 0)
        self.combine_layer = Combine(num_dims * input_size, input_size)




    def forward(self, features, nodes, to_neighs_dims, num_samples=10):
        """
        nodes --- list of nodes in a batch
        to_neighs_dims --- dictionary: dim to list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """

        agg_within_dims = [self.mean_agg(features, nodes, to_neighs_dims[i], num_samples) for i in range(self.num_dims)]
        input_node_embedding = torch.cat((agg_within_dims[0], agg_within_dims[1], agg_within_dims[2]), 1)
        input_node_embedding = input_node_embedding.reshape(-1, self.num_dims, self.input_size)
        # input_r1 = torch.matmul(input_node_embedding[:, 0, :], torch.mm(self.r1_matrix, self.r_all)).unsqueeze(1)
        # input_r2 = torch.matmul(input_node_embedding[:, 1, :], torch.mm(self.r2_matrix, self.r_all)).unsqueeze(1)
        # input_r3 = torch.matmul(input_node_embedding[:, 2, :], torch.mm(self.r3_matrix, self.r_all)).unsqueeze(1)

        input_r1 = torch.matmul(input_node_embedding[:, 0, :], self.r1_matrix).unsqueeze(1)
        input_r2 = torch.matmul(input_node_embedding[:, 1, :], self.r2_matrix).unsqueeze(1)
        input_r3 = torch.matmul(input_node_embedding[:, 2, :], self.r3_matrix).unsqueeze(1)
        input_node_embedding = torch.cat([input_r1, input_r2, input_r3], dim=1)
        # print(input_node_embedding[-1][0] * self.r1_matrix)
        mha_output = self.self_attention_layer.forward(input_node_embedding)
        mha_output = mha_output.reshape(-1, self.num_dims * self.input_size)
        final_output = self.combine_layer(mha_output)
        return final_output


class Combine(nn.Module):
    """
    Combine embeddings from different dimensions to generate a general embedding
    """

    def __init__(self, input_len=6, output_size=3, cuda=False, dropout_rate=0):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(Combine, self).__init__()
        self.cuda = cuda
        self.input_len = input_len
        self.output_size = output_size
        self.linear_layer = nn.Linear(self.input_len, self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        # print('input_len type: ',type(input_len))
        # print('output_size type : ',type(output_size))
        self.act = nn.ELU()
        if self.cuda:
            self.linear_layer.cuda()
            self.act.cuda()

    def forward(self, dim_embs):

        emb_combine = self.linear_layer(dim_embs)
        emb_combine_act = self.act(emb_combine)
        # emb_combine_act_drop = self.dropout(emb_combine_act)

        return emb_combine_act
