import random
import copy
import collections
import numpy as np
from torch.utils.data.sampler import Sampler

class PKSampler(Sampler):
    """
    Randomly samples P identities, then K instances of each identity.
    Ensures every batch is balanced with shape (P * K).
    """
    def __init__(self, data_source, p, k):
        self.data_source = data_source
        self.p = p # Number of PIDs per batch
        self.k = k # Number of images per PID
        self.batch_size = p * k
        
        self.index_dic = collections.defaultdict(list)
        
        for index in range(len(data_source)):
             pid = data_source.new_pids[index]
             self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        self.length = len(self.pids)

    def __iter__(self):
        batch_idxs_dict = collections.defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid]
            if len(idxs) < self.k:
                idxs = np.random.choice(idxs, size=self.k, replace=True)
            
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.k:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.p:
            selected_pids = random.sample(avai_pids, self.p)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length * self.k