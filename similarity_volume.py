import numpy as np
import torch
import os, sys, time
import itertools
import warnings

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
        start = time.time()

        if self.aug.shape[0] < 2:
            print("Too few detected embs")
            return self.aug

        # prepare main volume
        # einsum_prompt = ""
        # for i, row in enumerate(self.aug[:-1]):
        #     einsum_prompt += chr(i + 97) + ","
        # einsum_prompt += chr(len(self.aug) + 97 - 1)
        # print("prompt: ", einsum_prompt)

        # t = [row for row in self.aug]
        # volume = np.einsum(einsum_prompt, *t)

        volume = np.einsum('i,j', self.aug[0], self.aug[1])
        for i, row in enumerate(self.aug[2:]):
            # print(i, row)
            vb = volume.shape
            volume = np.einsum('...i,j', volume, row)
            # print(f"einsum {i} done in {time.time() - start} seconds")
            # print(f"{vb} -> {volume.shape}\n")


        # disallow repeats
        e = -np.inf * np.ones_like(volume)

        vol_dim = len(volume.shape)

        # unassigned may be repeated

        # print(f"e constructed at {time.time() - start}")

        # set unique assignments to 1
        for comb in itertools.permutations([i for i in range(volume.shape[0] - 1)], vol_dim):
            e[comb] = 1

            for j in range(1, 1 << vol_dim):
                c = list(comb)
                unassigned = [k for k in range(vol_dim) if j & 1 << k]
                for u in unassigned:
                    c[u] = -1
                # print(c)

                e[tuple(c)] = 1
        
        # print(f"unique at {time.time() - start}")

        # apply mask, fix all 0 * np.infs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            rep_volume = volume * e
            # print(f"mask at {time.time() - start}")
            rep_volume[np.isnan(rep_volume)] = -np.inf
            # print(f"fill at {time.time() - start}")

        return volume, rep_volume
    
    """
    Simpler volume, no space for rearrangements, all given objects are given an assignment
    """
    def construct_volume_choose_e(self, chosen_e):
        assert len(chosen_e) <= self.aug.shape[0]

        volume = np.einsum('i,j', self.aug[chosen_e[0]], self.aug[chosen_e[1]])
        for i, row_num in enumerate(chosen_e[2:]):
            # print(i, row)
            row = self.aug[row_num]
            vb = volume.shape
            volume = np.einsum('...i,j', volume, row)
            # print(f"einsum {i} done in {time.time() - start} seconds")
            # print(f"{vb} -> {volume.shape}\n")
        return volume
    
    # def get_minimum(self, chosen_e, vol):



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

    if len(set(indices)) != len(indices):
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

# use for rep vol
def test_repeated_missing(rep_vol, cs, verbose=False):
    num = cs.shape[0]
    indices = [np.random.randint(0,num) for i in range(num)]

    random_missing_idx = np.random.randint(0,num)
    indices[random_missing_idx] = -1
    indices = tuple(indices)


    if len(set(indices)) != len(indices):
        prod = -np.inf
    else:
        prod = 1
        for n, i in enumerate(indices):
            if i == -1:
                continue
            prod *= cs[n,i]

    if verbose:
        print(f"Indices: {indices}")
        print(f"Volume says: {rep_vol[indices]}")
        print(f"Product: {prod}")

    return rep_vol[indices] == prod

def test_repeated_multiple_missing(rep_vol, cs, verbose=False):
    num = cs.shape[0]
    indices = [np.random.randint(0,num) for i in range(num)]

    random_missing_idx1 = np.random.randint(0,num)
    random_missing_idx2 = np.random.randint(0,num)
    indices[random_missing_idx1] = -1
    indices[random_missing_idx2] = -1
    indices = tuple(indices)

    if len(set(indices))+1 != len(indices) and random_missing_idx1 != random_missing_idx2:
        prod = -np.inf
    elif len(set(indices)) != len(indices) and random_missing_idx1 == random_missing_idx2:
        prod = -np.inf
    else:
        prod = 1
        for n, i in enumerate(indices):
            if i == -1:
                continue
            prod *= cs[n,i]

    if verbose:
        print(f"Indices: {indices}")
        print(f"Volume says: {rep_vol[indices]}")
        print(f"Product: {prod}")

    return rep_vol[indices] == prod

def get_topK_inplace(vol, K):
    topK = []
    for i in range(K):
        ind = np.unravel_index(np.argmax(rep_vol, axis=None), rep_vol.shape)
        topK.append(ind)
        vol[ind] = -np.inf
    return topK

if __name__ == "__main__":

    cs = np.array([i for i in range(20)])
    cs2 = np.array([i for i in range(3)])
    cs = cs2.reshape(-1,1) + cs.reshape(1,-1)
    
    sv = SimVolume(cs)

    start = time.time()
    vol, rep_vol = sv.construct_volume()
    print(f"Volume constructed, dim: {rep_vol.shape}")

    time_taken = time.time() - start
    print(f"in {time_taken} seconds")
    start = time.time()

    ind = np.unravel_index(np.argmax(rep_vol, axis=None), rep_vol.shape)
    print(ind, rep_vol[ind])#, rep_vol[49,48,47])

    print(get_topK_inplace(rep_vol,10))

    # v2 = sv.construct_volume_choose_e([1,2,3])
    # print(v2.shape)

    exit(0)

    for i in range(100):
        assert(test_vol(vol, cs, False))
    print("Basic test passed")

    for i in range(100):
        assert(test_missing(vol, cs, False))
    print("Missing assignment test passed")

    for i in range(100):
        assert(test_repeated(rep_vol, cs, False))
    print("Repeated assignment test passed")

    for i in range(100):
        assert(test_repeated_missing(rep_vol, cs, False))
    print("Repeated missing assignment test passed")

    for i in range(100):
        assert(test_repeated_multiple_missing(rep_vol, cs, False))
    print("Repeated multiple missing assignment test passed")
    
    """
    Indices: (2, 4, 3, 0, -1)
    Volume says: 150.0
    Product: 1200
    """

    time_taken = time.time() - start
    print(f"Tests completed in {time_taken} seconds")