import numpy as np
from scipy.special import logsumexp

def score_max(sim_matrix):
    return sim_matrix.max(axis=1)

def score_topk(sim_matrix, k=5):
    return np.sort(sim_matrix, axis=1)[:, -k:].mean(axis=1)

