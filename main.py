import pandas as pd
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F

import data_lib
from dmgcn import DMGCN, CoupledRNN
from evaluation import Evaluation

import os


use_cuda = True
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda" if use_cuda else "cpu")
print("Use cuda or not: ", use_cuda)


feature_size = 128
epochs = 10
input_size = 128
output_sizes = [128, 128]
learning_rate = 0.001
time_threshold = 3600
dis_threshold = 100

# read data
# train_data_dim = pd.read_csv("train_data_dim.csv")
# train_data_sorted = pd.read_csv("train_data_sorted.csv")
# test_data_dim = pd.read_csv("test_data_dim.csv")
# test_data_sorted = pd.read_csv("test_data_sorted.csv")
# baltimore data
train_data_dim = pd.read_csv("train_data_new_dim_no_duplicate.csv")
train_data_sorted = pd.read_csv("train_data_sorted_new_no_duplicate.csv")
test_data_dim = pd.read_csv("test_data_new_dim_no_duplicate.csv")
test_data_sorted = pd.read_csv("test_data_sorted_new_no_duplicate.csv")

# dc data

# train_data_dim = pd.read_csv("dc_data8/dc8_train_data_new_dim_no_duplicate.csv")
# train_data_sorted = pd.read_csv("dc_data8/dc8_train_data_sorted_new_no_duplicate.csv")
# test_data_dim = pd.read_csv("dc_data8/dc8_test_data_new_dim_no_duplicate.csv")
# test_data_sorted = pd.read_csv("dc_data8/dc8_test_data_sorted_new_no_duplicate.csv")


print("Train data: \n", train_data_sorted)

print("Train data: ", len(train_data_sorted))
print("Test data: ", len(test_data_sorted))
max_loc_id_train = max(train_data_sorted["loc_id"])
max_loc_id_test = max(test_data_sorted["loc_id"])
max_id = max(max_loc_id_test, max_loc_id_train)
# id starts from 0, so +1

print("Number of nodes: ", max_id + 1)

# + another 1 to represent stop token
num_nodes = max_id + 1 + 1

num_user = len(np.unique(train_data_sorted["user_id"])) + len(np.unique(test_data_sorted["user_id"]))
print("Number of users: ", num_user)
num_loc = max_id - num_user + 1
print("Number of locations: ", num_loc)

unique_dim = np.unique(train_data_dim["dim"])
print("Dim: ", unique_dim)

num_dim = len(unique_dim)

whole_data = pd.concat([train_data_dim, test_data_dim], ignore_index=True)
data = data_lib.Data(whole_data, max_id)
to_neigh_dim = data.to_neigh_dims
#print(to_neigh_dim[0][5828])
# 5828

# set seed
#torch.manual_seed(4)

# trying
seed = 0
print("Seed: ", seed)
torch.manual_seed(seed)


# init emb matrix and model
initial_embedding = nn.Parameter(F.normalize(torch.rand(feature_size), dim=0))
emb_matrix = initial_embedding.repeat(num_nodes, 1).to(device)

dmgcn_model = DMGCN(num_dim, input_size, output_sizes, feature_size, device, use_cuda)
dmgcn_model.to(device)
rnn_model = CoupledRNN(feature_size, num_user, num_loc, time_threshold, dis_threshold, device)
rnn_model.to(device)
loss_func = nn.NLLLoss().to(device)
optimizer = optim.Adam(dmgcn_model.parameters(), lr=learning_rate, weight_decay=1e-5)

model_evaluation = Evaluation(loss_func, test_data_sorted, device)

# init user loc sequence
user_sequence = train_data_sorted["user_id"].values
loc_sequence = train_data_sorted["loc_id"].values
next_loc_sequence = train_data_sorted["next_loc"].values
time_diff_sequence = train_data_sorted["time_diff"].values
dis_diff_sequence = train_data_sorted["disdiff"].values


# init tbatch timestamp
timestamp_sequence = train_data_sorted["location_at"].values
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 200

# check test work or not
# print("Testing...")
# mean_recall, mean_mrr = model_evaluation.eval(dmgcn_model, rnn_model, data, num_user, emb_matrix, loss_func)
# print("Mean recall: ", mean_recall, "Mean mrr: ", mean_mrr)

# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_loc = {}
cached_tbatches_gt = {}
cached_tbatches_time_diff = {}
cached_tbatches_dis_diff = {}


best_recall = 0
best_mrr = 0
for epoch in range(epochs):
    print_cnt = 0

    print("Epoch:", epoch)
    dmgcn_model.train()
    rnn_model.train()
    optimizer.zero_grad()
    loss = 0
    total_loss = 0
    data_lib.reinitialize_tbatches()
    tbatch_start_time = None
    tbatch_to_insert = -1

    # the index for GNN input
    graph_start_idx = 0
    graph_end_idx = 0

    for idx in range(len(train_data_sorted)):

        if is_first_epoch:
            # READ INTERACTION J
            userid = user_sequence[idx]
            locationid = loc_sequence[idx]
            groud_truth = next_loc_sequence[idx]
            time_diff = time_diff_sequence[idx]
            dis_diff = dis_diff_sequence[idx]

            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
            tbatch_to_insert = max(data_lib.tbatchid_user[userid], data_lib.tbatchid_loc[locationid]) + 1
            data_lib.tbatchid_user[userid] = tbatch_to_insert
            data_lib.tbatchid_loc[locationid] = tbatch_to_insert

            data_lib.current_tbatches_user[tbatch_to_insert].append(userid)
            data_lib.current_tbatches_loc[tbatch_to_insert].append(locationid)
            data_lib.current_tbatches_gt[tbatch_to_insert].append(groud_truth)
            data_lib.current_tbatches_time_diff[tbatch_to_insert].append(time_diff)
            data_lib.current_tbatches_dis_diff[tbatch_to_insert].append(dis_diff)

        timestamp = timestamp_sequence[idx]
        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        if timestamp - tbatch_start_time >= tbatch_timespan:
            tbatch_start_time = timestamp  # RESET START TIME FOR THE NEXT TBATCHES

            # run GNN model
            graph_end_idx = idx
            graph_user = user_sequence[graph_start_idx:graph_end_idx]
            graph_loc = loc_sequence[graph_start_idx:graph_end_idx]
            graph_gt = next_loc_sequence[graph_start_idx:graph_end_idx]

            all_dims, dims_count, source_to_neigh_dims, target_to_neigh_dims = data.generate_dmgcn_input(graph_user,
                                                                                                         graph_loc)
            output_user_emb, output_loc_emb = dmgcn_model.forward(emb_matrix, all_dims, dims_count, graph_user,
                                                                  source_to_neigh_dims, graph_loc, target_to_neigh_dims)

            emb_matrix[graph_user, :] = output_user_emb
            emb_matrix[graph_loc, :] = output_loc_emb
            
            graph_start_idx = graph_end_idx
            # GNN end here


            # ITERATE OVER ALL T-BATCHES
            if not is_first_epoch:
                data_lib.current_tbatches_user = cached_tbatches_user[timestamp]
                data_lib.current_tbatches_loc = cached_tbatches_loc[timestamp]
                data_lib.current_tbatches_gt = cached_tbatches_gt[timestamp]
                data_lib.current_tbatches_time_diff = cached_tbatches_time_diff[timestamp]
                data_lib.current_tbatches_dis_diff = cached_tbatches_dis_diff[timestamp]


            for i in range(len(data_lib.current_tbatches_user)):

                if is_first_epoch:

                    data_lib.current_tbatches_user[i] = torch.LongTensor(data_lib.current_tbatches_user[i])
                    data_lib.current_tbatches_loc[i] = torch.LongTensor(data_lib.current_tbatches_loc[i])
                    data_lib.current_tbatches_gt[i] = torch.LongTensor(data_lib.current_tbatches_gt[i])
                    data_lib.current_tbatches_time_diff[i] = torch.LongTensor(data_lib.current_tbatches_time_diff[i])
                    data_lib.current_tbatches_dis_diff[i] = torch.LongTensor(data_lib.current_tbatches_dis_diff[i])

                tbatch_userids = data_lib.current_tbatches_user[i].to(device)# Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_locids = data_lib.current_tbatches_loc[i].to(device) # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_gts = data_lib.current_tbatches_gt[i].to(device)
                tbatch_time_diff = data_lib.current_tbatches_time_diff[i].to(device)
                tbatch_dis_diff = data_lib.current_tbatches_dis_diff[i].to(device)


                tbatch_user_emb = emb_matrix[tbatch_userids, :]
                tbatch_loc_emb = emb_matrix[tbatch_locids, :]

                # all loc emb matrix to calculate prediction scores
                score, output_user_emb, output_loc_emb = rnn_model(tbatch_user_emb, tbatch_loc_emb, emb_matrix,
                                                                   tbatch_time_diff, tbatch_dis_diff)

                emb_matrix[tbatch_userids, :] = output_user_emb
                emb_matrix[tbatch_locids, :] = output_loc_emb

                # minius num_user to change loc idx to start from 0
                #output_sampled = output[:, tbatch_gts.view(-1) - num_user]

                loss += loss_func(score, tbatch_gts.view(-1) - num_user)



            # BACKPROPAGATE ERROR AFTER END OF T-BATCH
            if print_cnt > 0 and print_cnt % 10 == 0:
                print("Loss: ", loss.item() / len(graph_user))
            print_cnt += 1
            #total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            output_user_emb.detach_()  # Detachment is needed to prevent double propagation of gradient
            output_loc_emb.detach_()
            emb_matrix.detach_()
            # # REINITIALIZE
            if is_first_epoch:
                cached_tbatches_user[timestamp] = data_lib.current_tbatches_user
                cached_tbatches_loc[timestamp] = data_lib.current_tbatches_loc
                cached_tbatches_gt[timestamp] = data_lib.current_tbatches_gt
                cached_tbatches_time_diff[timestamp] = data_lib.current_tbatches_time_diff
                cached_tbatches_dis_diff[timestamp] = data_lib.current_tbatches_dis_diff

                data_lib.reinitialize_tbatches()
                tbatch_to_insert = -1

    is_first_epoch = False  # as first epoch ends here
    #print("\n\nTotal loss in this epoch = %f" % (total_loss))
    print("Testing...")

    mean_recall, mean_mrr = model_evaluation.eval(dmgcn_model, rnn_model, data, num_user, emb_matrix, loss_func)
    print("Mean recall: ", mean_recall, "Mean mrr: ", mean_mrr)
    if mean_mrr > best_mrr:
        best_mrr = mean_mrr
    if mean_recall > best_recall:
        best_recall = mean_recall
    print("Best recall: {}, mrr: {} ".format(best_recall, best_mrr))

    emb_matrix = initial_embedding.repeat(num_nodes, 1).to(device)















