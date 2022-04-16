import numpy as np
from collections import defaultdict


class Data:
    def __init__(self, data, max_id):
        self.dataset = data.values
        self.length = np.shape(data)[0]
        self.all_dims = np.unique(self.dataset[:, 2])
        self.num_nodes = max_id + 1
        self.to_neigh_dims = self.get_to_neigh_dims()

    def get_to_neigh_dims(self):
        to_neigh_dims = dict()
        for i in range(self.length):
            source = int(self.dataset[i, 0])
            target = int(self.dataset[i, 1])
            dimension = int(self.dataset[i, 2])
            if dimension not in to_neigh_dims:
                to_neigh_dims[dimension] = dict()
                for node in range(self.num_nodes):
                    to_neigh_dims[dimension][node] = set([node])

            to_neigh_dims[dimension][source].add(target)
            to_neigh_dims[dimension][target].add(source)
        print('to_neigh_dims constructed')
        return to_neigh_dims

    def generate_dmgcn_input(self, tbatch_userids, tbatch_locids):
        source_to_neigh_dims = dict()
        target_to_neigh_dims = dict()
        dims_count = []
        for dim in self.all_dims:
            source_to_neigh_dims[dim] = [self.to_neigh_dims[dim][source.item()] for source in tbatch_userids]
            target_to_neigh_dims[dim] = [self.to_neigh_dims[dim][target.item()] for target in tbatch_locids]

            dims_count.append(len(source_to_neigh_dims[dim]))

        return self.all_dims, dims_count, source_to_neigh_dims, target_to_neigh_dims


def reinitialize_tbatches():
    global current_tbatches_user, current_tbatches_loc, current_tbatches_gt, current_tbatches_time_diff, current_tbatches_dis_diff
    global tbatchid_user, tbatchid_loc

    current_tbatches_user = defaultdict(list)
    current_tbatches_loc = defaultdict(list)
    current_tbatches_gt = defaultdict(list)
    current_tbatches_time_diff = defaultdict(list)
    current_tbatches_dis_diff = defaultdict(list)
    #current_tbatches_interaction_dim =defaultdict(list)

    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a location is in
    tbatchid_loc = defaultdict(lambda: -1)



