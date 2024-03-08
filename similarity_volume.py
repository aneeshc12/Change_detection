import numpy as np
import torch
import os, sys, time
import itertools


class SimVolume():
    def __init__(self, cosine_similarities) -> None:
        # e x m x 1
        aug = np.ones((cosine_similarities.shape[0], cosine_similarities.shape[1] + 1))
        aug[:, :-1] = cosine_similarities

        self.aug = aug

    """
    # construct an m x m x... x m (e times) volume, where each entry is the product of all
    # cosine similarities row-wise.
    # To query the total of assignment (0,3), (1,4), (2,0),
    #   retrieve vol[3, 4, 0]

    To query an assignment with missing indices, like (0,3), (2, 4), (3,0)
    replace all missing indices with a -1
        eg. vol[3, -1, 4, 0]
    """
    def construct_volume(self):
        if self.aug.shape[0] < 2:
            print("Too few detected embs")
            return self.aug

        # prepare main volume
        volume = np.einsum('i,j', self.aug[0], self.aug[1])
        for i, row in enumerate(self.aug[2:]):
            # print(i, row)
            vb = volume.shape
            volume = np.einsum('...i,j', volume, row)
            # print(f"{vb} -> {volume.shape}\n")

        # disallow repeats
        e = -np.inf * np.ones_like(volume)
        for comb in itertools.combinations([i for i in range(volume.shape[0])], volume.shape[0]):
            
            e[comb,:] = 1
        
        rep_volume = volume * e

        return volume, rep_volume
    



# use to check volume
def test_vol(vol, cs, verbose=False):
    num = cs.shape[0]
    indices = tuple([np.random.randint(0,num) for i in range(num)])

    prod = 1
    for n, i in enumerate(indices):
        prod *= cs[n,i]

    if verbose:
        print(f"Indices: {indices}")
        print(f"Volume says: {vol[indices]}")
        print(f"Product: {prod}")

    return vol[indices] == prod

# use for volume
def test_missing(vol, cs, verbose=False):
    num = cs.shape[0]
    indices = [np.random.randint(0,num) for i in range(num)]

    random_missing_idx = np.random.randint(0,num)
    indices[random_missing_idx] = -1
    indices = tuple(indices)

    prod = 1
    for n, i in enumerate(indices):
        if i == -1:
            continue
        prod *= cs[n,i]

    if verbose:
        print(f"Indices: {indices}")
        print(f"Missing assingment: {random_missing_idx}")
        print(f"Volume says: {vol[indices]}")
        print(f"Product should be zero: {prod}")

    return vol[indices] == prod

# use for rep vol
def test_repeated(rep_vol, cs, verbose=False):
    num = cs.shape[0]
    indices = tuple([np.random.randint(0,num) for i in range(num)])

    if len(set(indices)) == len(indices):
        prod = -np.inf
    else:
        prod = 1
        for n, i in enumerate(indices):
            prod *= cs[n,i]

    if verbose:
        print(f"Indices: {indices}")
        print(f"Volume says: {rep_vol[indices]}")
        print(f"Product: {prod}")

    return rep_vol[indices] == prod
    
if __name__ == "__main__":
    cs = np.array([i for i in range(5)])
    cs = cs.reshape(-1,1) + cs.reshape(1,-1)
    
    sv = SimVolume(cs)

    vol, rep_vol = sv.construct_volume()

    # print(cs)
    # print(sv.aug)
    # print(vol.shape)

    test_missing(vol, cs, True)

    for i in range(100):
        assert(test_vol(vol, cs, False))
    print("Basic test passed")

    for i in range(100):
        assert(test_missing(vol, cs, False))
    print("Missing assignment test passed")

    for i in range(100):
        assert(test_repeated(rep_vol, cs, True))
    print("Repeated assignment test passed")