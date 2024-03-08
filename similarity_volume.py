import numpy as np
import torch
import os, sys, time


class SimVolume():
    def __init__(self, cosine_similarities) -> None:
        # e x m x 1
        aug = np.zeros((cosine_similarities.shape[0], cosine_similarities.shape[1] + 1))
        aug[:, :-1] = cosine_similarities

        self.aug = aug

    def construct_volume(self):
        if self.aug.shape[0] < 2:
            print("Too few detected embs")
            return self.aug

        volume = np.einsum('i,j', self.aug[0], self.aug[1])
        for i, row in enumerate(self.aug[2:]):
            # print(i, row)
            vb = volume.shape
            volume = np.einsum('...i,j', volume, row)
            # print(f"{vb} -> {volume.shape}\n")

        return volume
    
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
    
if __name__ == "__main__":
    cs = np.array([i for i in range(5)])
    cs = cs.reshape(-1,1) + cs.reshape(1,-1)
    
    sv = SimVolume(cs)

    vol = sv.construct_volume()

    # print(cs)
    # print(sv.aug)
    # print(vol.shape)

    for i in range(100):
        assert(test_vol(vol, cs, True))