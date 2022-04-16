import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.sparse as sp 

import random
import time
import numpy as np
"""
Set of modules for aggregating embeddings of neighbors.
"""

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    # def __init__(self, sparse):
    #     super(SparseMM, self).__init__()
    #     self.sparse = sparse

    @staticmethod
    def forward(self, sparse, dense):
        self.sparse = sparse
        return torch.mm(self.sparse, dense)

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return None, grad_input


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        #self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, features, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        test_time = False
        _set = set
        start1 = time.time()
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        end1 = time.time()
        if test_time:
            print('get neighbors: ', str(end1-start1))

        start2 = time.time()

        unique_nodes_list = list(set.union(*samp_neighs))



        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        end2 = time.time()
        if test_time:
            print('get nodes: ', str(end2-start2))
        #print('Test1:',len(samp_neighs))
        #print('Test2:',len(unique_nodes_list))
        start3 = time.time()
    #    mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]

        values = [1.0]*len(row_indices)
        mask_sparse = sp.coo_matrix((values,(row_indices,column_indices)))
        mask_sparse_normalized = normalize(mask_sparse)
        mask = sparse_mx_to_torch_sparse_tensor(mask_sparse_normalized)


        if self.cuda:
            mask = mask.cuda()
        #node id starts from 1, so -1 to get the corresponding embedding
        unique_nodes_list = np.array(unique_nodes_list)-1

        if self.cuda:
            embed_matrix = features[torch.LongTensor(unique_nodes_list).cuda()]
        else:
            embed_matrix = features[torch.LongTensor(unique_nodes_list)]
       # print(embed_matrix.shape)
        end4 = time.time()
        if test_time:
            print('pepare for emb : ', str(end4-start3))  ##changed

        start5 = time.time()

        # to_feats = mask.mm(embed_matrix)

        to_feats = SparseMM.apply(mask, embed_matrix)
        end5= time.time()


        if test_time:
            print('get emb time: ', str(end5-start5))
        return to_feats
