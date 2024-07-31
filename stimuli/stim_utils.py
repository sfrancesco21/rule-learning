import numpy as np
from scipy.stats import multivariate_normal, chi2
import random
import copy
from scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform
from random import shuffle, sample
import scipy.io

def load_mat(filepath):
    """
    Load a .mat file and return its contents as a dictionary.

    Parameters:
    filepath (str): The path to the .mat file.

    Returns:
    dict: A dictionary containing the variables from the .mat file.
    """
    mat_contents = scipy.io.loadmat(filepath)
    
    # Remove metadata entries
    mat_contents = {key: value for key, value in mat_contents.items() if not key.startswith('__')}
    
    return mat_contents

def generate_test(n):
    stims = []
    labels = []
    for i in range(1,7):
        for j in range(1,7):
            stims += n*[[i, j]]

    random.shuffle(stims)
    for s in stims:
        if np.abs(s[0] - s[1]) > 1:
            label = 1
        else:
            label = 0
        labels += [label]

    return stims, labels

def generate_training_interleaved(n):
    configs = [
        [1, 6], 
        [1, 4],
        [2, 5], 
        [2, 3], 
        [2, 2], 
        [3, 6], 
        [3, 2], 
        [3, 1], 
        [4, 5], 
        [4, 4], 
        [4, 2], 
        [5, 6], 
        [5, 4],
        [5, 3], 
        [5, 2], 
        [6, 6], 
        [6, 3],
        [6, 2]
    ]
    stims = []
    labels = []
    for c in configs:
        stims += n*[c]
    random.shuffle(stims)
    for s in stims:
        if np.abs(s[0] - s[1]) > 1:
            label = 1
        else:
            label = 0
        labels += [label]

    return stims, labels

def generate_training_blocked(n):    

    # Initialize variables
    counter = 1
    pos = []
    conf = []
    outcome = []

    # Fill pos, conf, and outcome arrays
    for r in range(1, 7):
        for l in range(1, 7):
            pos.append([r, l])
            conf.append(counter)
            outcome.append(abs(r - l) <= 1)
            counter += 1

    s = copy.deepcopy(pos)
    o = copy.deepcopy(outcome)
    pos = np.array(pos)
    conf = np.array(conf)
    outcome = np.array(outcome)

    # Select specific indices
    selected_indices = list(np.array([3, 8, 9, 11, 12, 14, 17, 18, 19, 22, 23, 26, 27, 28, 31, 33, 35, 36]) - 1)

    pos = pos[selected_indices, :]
    conf = conf[selected_indices]
    outcome = outcome[selected_indices]

    tobefound = True
    selected = []

    # Main while loop
    while tobefound:
        idx = sample(range(len(conf)), 3)
        selected_pos = pos[idx, :]
        
        if (np.all(pdist(selected_pos) > 1.4) and 
            0 < sum(outcome[idx]) <= 2):
            selected.extend(conf[idx])
            pos = np.delete(pos, idx, axis=0)
            conf = np.delete(conf, idx)
            outcome = np.delete(outcome, idx)

        if len(conf) == 0:
            tobefound = False

    # Repetition and order generation
    counter = 1
    repetition = n
    order = []

    for t in range(1, 17, 3):
        temp = np.tile(selected[t-1:t+2], (repetition, 1)).flatten()
        shuffled_temp = temp.tolist()
        shuffle(shuffled_temp)
        order.extend(shuffled_temp)

    # Convert to numpy array if needed
    order = np.array(order) - 1

    stims = []
    labels = []

    for i in order:
        stims.append(s[i])
        labels.append(1-o[i])

    return stims, labels