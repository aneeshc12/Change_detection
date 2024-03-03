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
    def get_assignments(self):
        self.leaf_nodes = []
        JCBBNode(self, self.num_detected, 0, 0, [], [i for i in range(self.num_memory)])
        self.sort_assignments()
        return self.leaf_nodes
    
    def sort_assignments(self):
        self.leaf_nodes = sorted(self.leaf_nodes, key=lambda x: x[1], reverse=True)

"""
Node in the JCBB tree
tree : the jcbb tree object that manages the calculation
max_depth : max depth to traverse to (num objects)
depth : i
assignment : j
path : array of assignments before this (index is i, value is j)
choices_left : list of indices left to assign

calculate the product of similarities as we go down (add log costs)
"""
def JCBBNode(tree, max_depth, similarity_sum, depth, path, choices_left):
    if depth == max_depth:
        tree.leaf_nodes.append([path, similarity_sum/max_depth])
        return
    
    # recurse through 
    for c in choices_left:
        # add the cosine similarity of this assignment to the cost
        new_sum = similarity_sum + np.log(tree.cosine_similarities[depth][c])

        child_path = copy(path)

        # let this child be assignment (depth+1, c), remove c from choices
        child_path.append(c)
        child_choices_left = [d for d in choices_left if d != c]
        
        JCBBNode(
            tree,
            max_depth,
            new_sum,
            depth+1,
            child_path,
            child_choices_left
        )

if __name__ ==  '__main__':
    sim = np.random.uniform(size=(4,20))

    start = time.time()

    j = JCBB(sim, None)
    paths = j.get_assignments()
    # for p in paths:
    #     for k, v in p.items():
    #         print(k,v, sep="|", end=" ")
    #     print()

    print("Time taken: ", time.time()-start)

    # print(sim)

    # print(paths)