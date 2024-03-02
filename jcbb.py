import numpy as np
import os
import torch
from copy import copy

import time

class JCBB():
    def __init__(self, cosine_similarities, R_matrices):
        self.cosine_similarities = cosine_similarities
        self.R_matrices = R_matrices

        self.num_detected = self.cosine_similarities.shape[0]
        self.num_memory   = self.cosine_similarities.shape[1]

        self.leaf_nodes = []
        
        if self.num_detected > self.num_memory:
            raise NotImplementedError

    # get all assignment paths
    def get_assignments(self, to_be_assigned=None):
        if to_be_assigned == None:
            to_be_assigned = [i for i in range(self.num_detected)]
        
        self.leaf_nodes = []
        JCBBNode(self, 
                 indices_left=to_be_assigned, 
                 similarity_sum=0, depth=0, 
                 path=[], choices_left=[i for i in range(self.num_memory)])
        self.sort_assignments()
        return self.leaf_nodes

    def get_all_subset_assignments(self, min_length=1):
        # generate the power set of all assignments
        def powerset(s, l):
            x = len(s)
            ps = []
            for i in range(1 << x):
                subset = [s[j] for j in range(x) if (i & (1 << j))]
                if len(subset) >= l:
                    ps.append(subset)
            return ps
        
        self.leaf_nodes = []
        for subset in powerset([i for i in range(self.num_detected)], min_length):
            JCBBNode(self, 
                 indices_left=subset, 
                 similarity_sum=0, depth=0, 
                 path=[], choices_left=[i for i in range(self.num_memory)])
        
        self.sort_assignments()
        return self.leaf_nodes 

    
    def sort_assignments(self):
        self.leaf_nodes = sorted(self.leaf_nodes, key=lambda x: x[1], reverse=True)

"""
Node in the JCBB tree
tree : the jcbb tree object that manages the calculation
indices_left : the indices left to be assigned
depth : i
assignment : j
path : array of assignments before this (index is i, value is j)
choices_left : list of indices left to assign

calculate the product of similarities as we go down (add log costs)
"""
def JCBBNode(tree, indices_left, similarity_sum, depth, path, choices_left):
    if len(indices_left) == 0:
        tree.leaf_nodes.append([path, similarity_sum/len(path)])
        return
    
    # recurse through 
    for c in choices_left:
        child_path = copy(path)

        # let this child be assignment (indices_left[0], c), 
        #       remove c from choices
        #       remove indices_left[0] from indices_left
        child_path.append([indices_left[0], c])
        child_choices_left = [d for d in choices_left if d != c]
        
        # add the cosine similarity of this assignment to the cost (indices)
        new_sum = similarity_sum + np.log(tree.cosine_similarities[indices_left[0]][c])
        
        JCBBNode(
            tree,
            indices_left[1:],
            new_sum,
            depth+1,
            child_path,
            child_choices_left
        )

if __name__ ==  '__main__':
    np.random.seed(seed=42)
    sim = np.random.uniform(size=(4,4))

    start = time.time()

    j = JCBB(sim, None)

    paths = j.get_all_subset_assignments(min_length=1)

    # paths = j.get_assignments(to_be_assigned=[1,2,3])
    # paths = j.get_assignments()
    for p in paths[:20]:
        print(p[0],p[1], sep="|", end="\n")
        # print()

    # # print(paths)
    # print("Time taken: ", time.time()-start)

    # print(sim)
