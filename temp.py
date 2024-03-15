import pickle

with open("/scratch/vineeth.bhat/results/vineeth-1/results.pkl", "rb") as f:
    print(pickle.load(f))