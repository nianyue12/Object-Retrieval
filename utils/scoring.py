import numpy as np
from scipy.special import logsumexp

def score_max(sim_matrix):
    return sim_matrix.max(axis=1)

def score_topk(sim_matrix, k=5):
    return np.sort(sim_matrix, axis=1)[:, -k:].mean(axis=1)

def score_margin(sim_matrix):
    top2 = np.partition(sim_matrix, -2, axis=1)[:, -2:]
    return top2[:, -1] - top2[:, -2]

def score_energy(sim_matrix):
    return logsumexp(sim_matrix, axis=1)