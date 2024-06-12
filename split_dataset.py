import numpy as np 
import os 
import copy 
import time
import threading
import torch 
import random
import argparse
from deepgate.utils.data_utils import read_npz_file

input_path = '/home/zyshi21/data/share/all.npz'
output_folder = '/home/zyshi21/data/share/dg3_dataset'

def save_npz(folder_name, selected_circuits, circuits, no_npz):
    set_folder = os.path.join(output_folder, folder_name)
    if not os.path.exists(set_folder):
        os.mkdir(set_folder)
    
    split_index = []
    npz_cir_cnt = len(selected_circuits) // no_npz
    for npz_idx in range(no_npz):
        split_index.append(npz_idx * npz_cir_cnt)
    split_index.append(len(selected_circuits))
    
    for npz_idx in range(no_npz):
        npz_path = os.path.join(set_folder, '{:02d}.npz'.format(npz_idx))
        graphs = {}
        for cir_idx in range(split_index[npz_idx], split_index[npz_idx + 1]):
            cir_name = selected_circuits[cir_idx]
            g = {}
            succ = True
            for key in circuits[cir_name]:
                if 'path' in key or 'ninp' in key:
                    continue
                g[key] = circuits[cir_name][key]
                if isinstance(g[key], list):
                    for shape_val in np.array(g[key]).shape:
                        if shape_val == 0:
                            succ = False
                            break
            if not succ:
                continue
            graphs[cir_name] = copy.deepcopy(g)   
        np.savez(npz_path, circuits=graphs)
        print('Save NPZ: {}, (length: {:})'.format(npz_path, len(graphs)))
    
if __name__ == '__main__':
    circuits = read_npz_file(input_path)['circuits'].item()
    no_circuits = len(circuits)
    cir_name_list = list(circuits.keys())
    random.shuffle(cir_name_list)
    
    # Test set
    test_set_namelist = cir_name_list[:int(no_circuits * 0.05)]
    save_npz('test', test_set_namelist, circuits, 1)
    
    # Train set
    train_set_namelist = cir_name_list[int(no_circuits * 0.05):]

    save_npz('100p', train_set_namelist, circuits, 10)
    save_npz('50p', train_set_namelist[:int(len(train_set_namelist) * 0.5)], circuits, 5)
    save_npz('30p', train_set_namelist[:int(len(train_set_namelist) * 0.3)], circuits, 3)
    save_npz('10p', train_set_namelist[:int(len(train_set_namelist) * 0.1)], circuits, 1)
    save_npz('5p', train_set_namelist[:int(len(train_set_namelist) * 0.05)], circuits, 1)
    save_npz('1p', train_set_namelist[:int(len(train_set_namelist) * 0.01)], circuits, 1)

    