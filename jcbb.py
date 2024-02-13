import numpy as np
import os
import torch
from copy import copy

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
        JCBBNode(self, self.num_detected, 0, [], [i for i in range(self.num_memory)])
        return self.leaf_nodes

"""
Node in the JCBB tree
tree : the jcbb tree object that manages the calculation
max_depth : max depth to traverse to (num objects)
depth : i
assignment : j
path : array of assignments before this (index is i, value is j)
choices_left : list of indices left to assign
"""
def JCBBNode(tree, max_depth, depth, path, choices_left):
    if depth == max_depth:
        tree.leaf_nodes.append(path)
        return
    
    # recurse through 
    for c in choices_left:
        child_path = copy(path)
        
        # let this child be assignment (depth+1, c), remove c from choices
        child_path.append(c)
        child_choices_left = [d for d in choices_left if d != c]
        
        JCBBNode(
            tree,
            max_depth,
            depth+1,
            child_path,
            child_choices_left
        )

if __name__ ==  '__main__':
    j = JCBB(np.zeros((3,4)), None)
    paths = j.get_assignments()
    # for p in paths:
    #     for k, v in p.items():
    #         print(k,v, sep="|", end=" ")
    #     print()

    print(paths)