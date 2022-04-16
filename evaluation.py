import numpy as np
import torch
import metric
from collections import defaultdict


def reinitialize_tbatches_test():
    global current_tbatches_user_test, current_tbatches_loc_test, current_tbatches_gt_test, current_tbatches_time_diff_test, current_tbatches_dis_diff_test
    global tbatchid_user_test, tbatchid_loc_test

    current_tbatches_user_test = defaultdict(list)
    current_tbatches_loc_test = defaultdict(list)
    current_tbatches_gt_test = defaultdict(list)
    current_tbatches_time_diff_test = defaultdict(list)
    current_tbatches_dis_diff_test = defaultdict(list)
    #current_tbatches_interaction_dim =defaultdict(list)

    tbatchid_user_test = defaultdict(lambda: -1)

    # the latest tbatch a location is in
    tbatchid_loc_test = defaultdict(lambda: -1)


class Evaluation(object):
    def __init__(self, loss_func, test_data_sorted, device):

        self.loss_func = loss_func
        self.test_data_sorted = test_data_sorted
        self.device = device

    def eval(self, dmgcn_model, rnn_model, data,num_user, features, loss_func):
        dmgcn_model.eval()
        rnn_model.eval()
        prt_cnt_test = 0
        loss = 0
        recalls = 0
        mrrs = 0

        recall_list = []
        mrr_list = []

        user_sequence = self.test_data_sorted["user_id"].values
        loc_sequence = self.test_data_sorted["loc_id"].values
        next_loc_sequence = self.test_data_sorted["next_loc"].values
        timestamp_sequence_test = self.test_data_sorted["location_at"].values
        time_diff_sequence = self.test_data_sorted["time_diff"].values
        dis_diff_sequence = self.test_data_sorted["disdiff"].values

        timespan_test = timestamp_sequence_test[-1] - timestamp_sequence_test[0]
        tbatch_timespan_test = timespan_test / 30
        reinitialize_tbatches_test()

        tbatch_start_time_test = None
        tbatch_to_insert_test = -1



        # the index for GNN input
        graph_start_idx = 0
        graph_end_idx = 0

        for test_idx in range(len(self.test_data_sorted)):
            userid = user_sequence[test_idx]
            locationid = loc_sequence[test_idx]
            groud_truth = next_loc_sequence[test_idx]
            time_diff = time_diff_sequence[test_idx]
            dis_diff = dis_diff_sequence[test_idx]


            # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
            tbatch_to_insert_test = max(tbatchid_user_test[userid], tbatchid_loc_test[locationid]) + 1
            tbatchid_user_test[userid] = tbatch_to_insert_test
            tbatchid_loc_test[locationid] = tbatch_to_insert_test
            #print(current_tbatches_user_test[tbatch_to_insert_test])
            current_tbatches_user_test[tbatch_to_insert_test].append(userid)
            current_tbatches_loc_test[tbatch_to_insert_test].append(locationid)
            current_tbatches_gt_test[tbatch_to_insert_test].append(groud_truth)
            current_tbatches_time_diff_test[tbatch_to_insert_test].append(time_diff)
            current_tbatches_dis_diff_test[tbatch_to_insert_test].append(dis_diff)

            timestamp_test = timestamp_sequence_test[test_idx]

            if tbatch_start_time_test is None:
                tbatch_start_time_test = timestamp_test

            # print("timestamp_test", timestamp_test)
            # print("tbatch_start_time_test", tbatch_start_time_test)
            # print("tbatch_timespan_test", tbatch_timespan_test)
            # print("Res", timestamp_test - tbatch_start_time_test)
            if timestamp_test - tbatch_start_time_test >= tbatch_timespan_test:
                tbatch_start_time_test = timestamp_test  # RESET START TIME FOR THE NEXT TBATCHES
                #print("Running model")

                # run GNN model
                graph_end_idx = test_idx
                graph_user = user_sequence[graph_start_idx:graph_end_idx]
                graph_loc = loc_sequence[graph_start_idx:graph_end_idx]
                graph_gt = next_loc_sequence[graph_start_idx:graph_end_idx]

                all_dims, dims_count, source_to_neigh_dims, target_to_neigh_dims = data.generate_dmgcn_input(graph_user,
                                                                                                             graph_loc)
                output_user_emb, output_loc_emb = dmgcn_model.forward(features, all_dims, dims_count, graph_user,
                                                                      source_to_neigh_dims, graph_loc,
                                                                      target_to_neigh_dims)

                features[graph_user, :] = output_user_emb
                features[graph_loc, :] = output_loc_emb

                graph_start_idx = graph_end_idx
                # GNN end here
                for i in range(len(current_tbatches_user_test)):

                    tbatch_userids = torch.LongTensor(current_tbatches_user_test[i]).to(self.device)  # Recall "lib.current_tbatches_user[i]" has unique elements
                    tbatch_locids = torch.LongTensor(current_tbatches_loc_test[i]).to(self.device)# Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_gts = torch.LongTensor(current_tbatches_gt_test[i]).to(self.device)
                    tbatch_time_diff = torch.LongTensor(current_tbatches_time_diff_test[i]).to(self.device)  # Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_dis_diff = torch.LongTensor(current_tbatches_dis_diff_test[i]).to(self.device)


                    tbatch_user_emb = features[tbatch_userids, :].to(self.device)
                    tbatch_loc_emb = features[tbatch_locids, :].to(self.device)

                    score, output_user_emb, output_loc_emb = rnn_model(tbatch_user_emb, tbatch_loc_emb, features,
                                                                       tbatch_time_diff, tbatch_dis_diff)

                    features[tbatch_userids, :] = output_user_emb
                    features[tbatch_locids, :] = output_loc_emb

                    loss += loss_func(score, tbatch_gts.view(-1) - num_user)

                    recall, mrr = metric.evaluate(score, tbatch_gts.view(-1) - num_user)
                    recalls += recall

                    mrrs += mrr


                output_user_emb.detach_()  # Detachment is needed to prevent double propagation of gradient
                output_loc_emb.detach_()
                features.detach_()
                # reinitialize
                reinitialize_tbatches_test()
                tbatch_to_insert_test = -1
                if prt_cnt_test > 0 and prt_cnt_test % 10 == 0:

                    print("Loss: ", loss.item() / len(graph_user))
                    print("Recall: ", recalls/len(graph_user), "Mrr: ", mrrs/len(graph_user))
                prt_cnt_test += 1

                loss = 0
                recall_list.append(recalls / len(graph_user))
                mrr_list.append(mrrs / len(graph_user))

        mean_recall = np.mean(recall_list)
        mean_mrr = np.mean(mrr_list)

        return mean_recall, mean_mrr

