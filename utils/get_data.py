import glob
import os
import scipy
import numpy as np

def config_dict():
    counter = 0
    conf = {}
    label = {}
    for r in range(1, 7):
        for l in range(1, 7):
            counter += 1
            conf[counter] = [r, l]
            label[counter] = int(abs(r - l) <= 1)
    return [conf, label]
            

def read_mat_files(base_dir='../Results'):
    conf = config_dict()
    data = {}
    for N in range(1, 126):
        folder_name = f'sbj{N}'
        folder_path = os.path.join(base_dir, folder_name)
        training_pattern = os.path.join(folder_path, f'sbj{N}Training*')
        test_pattern     = os.path.join(folder_path, f'sbj{N}Test*')
        
        training_files = glob.glob(training_pattern)
        test_files     = glob.glob(test_pattern)
        if training_files and test_files:
            training_file = training_files[0]
            test_file = test_files[0]
            training_data = scipy.io.loadmat(training_file)
            test_data = scipy.io.loadmat(test_file)
            data[N] = {
                'training' : {
                    'trialN' : [training_data['Results'][0][0][0][0][i][0][0][0] for i in range(training_data['Results'][0][0][0][0].shape[0])],
                    'stimuli' : np.array([conf[0][training_data['Results'][0][0][0][0][i][1][0][0]] for i in range(training_data['Results'][0][0][0][0].shape[0])]),
                    'labels' : np.array([conf[1][training_data['Results'][0][0][0][0][i][1][0][0]] for i in range(training_data['Results'][0][0][0][0].shape[0])]),
                    'outcome' : [training_data['Results'][0][0][0][0][i][2][0][0] for i in range(training_data['Results'][0][0][0][0].shape[0])], 
                    'action' : np.array([training_data['Results'][0][0][0][0][i][4][0][0] for i in range(training_data['Results'][0][0][0][0].shape[0])]),
                    'trialLost' : np.array([training_data['Results'][0][0][0][0][i][6][0][0] for i in range(training_data['Results'][0][0][0][0].shape[0])])
                },
                'test' : {
                    'trialN' : [test_data['Results'][0][0][0][0][i][0][0][0] for i in range(test_data['Results'][0][0][0][0].shape[0])],
                    'stimuli' : np.array([conf[0][test_data['Results'][0][0][0][0][i][1][0][0]] for i in range(test_data['Results'][0][0][0][0].shape[0])]),
                    'labels' : np.array([conf[1][test_data['Results'][0][0][0][0][i][1][0][0]] for i in range(test_data['Results'][0][0][0][0].shape[0])]),
                    'outcome' : [test_data['Results'][0][0][0][0][i][2][0][0] for i in range(test_data['Results'][0][0][0][0].shape[0])], 
                    'action' : np.array([test_data['Results'][0][0][0][0][i][4][0][0] for i in range(test_data['Results'][0][0][0][0].shape[0])]),
                    'trialLost' : np.array([test_data['Results'][0][0][0][0][i][6][0][0] for i in range(test_data['Results'][0][0][0][0].shape[0])])
                }
            }
            #data[N] = {'training': training_data, 'test': test_data}
        else:
            print(f"Files for {folder_name} not found in {base_dir}.")
    return data


def participant_accuracy_by_phase(data):
    """
    Returns participant accuracy for training and test, excluding trialLost==1.
    Assumes:
      - data[phase]["labels"] are correct (0/1)
      - data[phase]["action"] are participant actions (0/1)
      - data[phase]["trialLost"] is 0/1
    """
    out = {}
    for phase in ("training", "test"):
        lab = np.asarray(data[phase]["labels"]).astype(int)
        act = np.asarray(data[phase]["action"]).astype(int)
        lost = np.asarray(data[phase]["trialLost"]).astype(int)

        keep = (lost == 0)
        if keep.sum() == 0:
            out[phase] = np.nan
        else:
            out[phase] = float(np.mean(act[keep] == lab[keep]))
    return out

def get_group(s):
    if s % 2 == 0 and s <= 60:
        group = 'blocked1'
    elif s <= 60:
        group = 'interleaved'
    elif s <=90:
        group = 'blocked2'
    else:
        group = 'blocked3'
    return group