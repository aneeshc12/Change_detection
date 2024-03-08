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
            if l < 1:
                l = 1
            
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

    """
    Get the top `assignments_per_length` assignments out of all subsets, sorted by normalised cos sim
    """
    def get_candidate_assignments(self, min_length=1, max_length=1e3, assignments_per_length=4):
        min_length = max(1,min_length)
        max_length = min(self.num_detected, max_length)
        self.get_all_subset_assignments(min_length)

        candidate_assns = []
        for length in range(min_length, max_length + 1):
            cnt_len = [assn for assn in self.leaf_nodes if len(assn[0]) == length]
            cnt_len = sorted(cnt_len, key=lambda x: x[1], reverse=True)
            candidate_assns += cnt_len[:assignments_per_length]
        
        return candidate_assns
    
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
    print("call JCBBNode block2 ", depth)

    to_be_assigned = indices_left[0]
    remaining = indices_left[1:]
    # indices_left.pop(0)
    for c in choices_left:
        # let this child be assignment (indices_left[0], c), 
        #       remove c from choices
        #       remove indices_left[0] from indices_left
        # print("ind: ", indices_left)
        child_path = path + [[to_be_assigned, c]]
        child_choices_left = [d for d in choices_left if d != c]
        
        # add the cosine similarity of this assignment to the cost (indices)
        new_sum = similarity_sum + np.log(tree.cosine_similarities[to_be_assigned][c])
        print("about to call ", depth)
        
        JCBBNode(
            tree,
            remaining,
            new_sum,
            depth+1,
            child_path,
            child_choices_left
        )
    return

if __name__ ==  '__main__':
    np.random.seed(seed=42)
    sim = np.random.uniform(size=(4,4))

    start = time.time()

    j = JCBB(sim, np.random.uniform(size=(4,40,3,3)))

    paths = j.get_assignments()

    # paths = j.get_assignments(to_be_assigned=[1,2,3])
    # paths = j.get_assignments()
    # for p in paths[:20]:
    #     print(p[0],p[1], sep="|", end="\n")
    #     print()

    print(paths)
    print("Time taken: ", time.time()-start)

    # print(sim)
