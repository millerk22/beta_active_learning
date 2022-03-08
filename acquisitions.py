import numpy as np
from scipy.special import softmax


def beta_var(A, candidate_set):
    a0 = A.sum(axis=1)
    a = (A * A).sum(axis=1)
    return ((1. - a/(a0**2.))/(1. + a0))[candidate_set]

def unc(u, candidate_set):
    u_sort = np.sort(u[candidate_set])
    return 1. - (u_sort[:,-1] - u_sort[:,-2]) # smallest margin acquisition function

def uncsftmax(u):
    s = softmax(u[candidate_set], axis=1)
    u_sort = np.sort(s)
    return 1. - (u_sort[:,-1] - u_sort[:,-2]) # smallest margin

def uncdist(u):
    '''
    Straightforward Euclidean distance to current pseudolabel
    '''
    one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[candidate_set], axis=1)]
    return  np.linalg.norm((u[candidate_set] - one_hot_predicted_labels), axis=1)

def uncsftmaxnorm(u):
    '''
    Project onto simplex and then Euclidean distance to current pseudolabel
    '''
    u_probs = softmax(u[candidate_set], axis=1)
    one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[candidate_set], axis=1)]
    return np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)

def uncnorm(u):
    return 1. - np.linalg.norm(u[candidate_set], axis=1)


def random(u):
    return np.random.rand(u.shape[0])
