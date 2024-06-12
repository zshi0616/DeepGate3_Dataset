import numpy as np 
import os 
import copy 
import time
import threading
import torch 
import random
import argparse
from deepgate.utils.data_utils import read_npz_file
from utils.utils import run_command
from utils.circuit_utils import get_fanin_fanout, get_level

input_npz = '/home/zyshi21/data/share/QoR_test/graphs_area.npz'
output_npz = '/home/zyshi21/data/share/QoR_test/graphs_area_labeled.npz'

aig_folder = './QoR_Design'

if __name__ == '__main__':
    circuits = read_npz_file(input_npz)['circuits'].item()
    graphs = {}
    
    for cir_idx, cir_name in enumerate(circuits):
        print("Read Circuit: {}".format(cir_name))
        aig_path = os.path.join(aig_folder, '{}.aig'.format(cir_name))
        
        if not os.path.exists(aig_path):
            print('No file: {}'.format(aig_path))
            raise
        abc_cmd = 'abc -c \"&read {}; &ps; &syn2; &ps; \"'.format(aig_path)
        stdout, _ = run_command(abc_cmd)
        
        # Original 
        arr = stdout[2].split('\\x')
        ori_nds = int(arr[3].split(' ')[-1])
        ori_lvs = int(arr[5].split(' ')[-1])
        # Transformed 
        arr = stdout[3].split('\\x')
        trans_nds = int(arr[3].split(' ')[-1])
        trans_lvs = int(arr[5].split(' ')[-1])
        
        g = {}
        for key in circuits[cir_name]:
            g[key] = circuits[cir_name][key]
        g['ori_nds'] = ori_nds
        g['ori_lvs'] = ori_lvs
        g['trans_nds'] = trans_nds
        g['trans_lvs'] = trans_lvs
        graphs[cir_name] = copy.deepcopy(g)
    
    np.savez(output_npz, circuits=graphs)
    print('Output:', output_npz)
    print('Total:', len(graphs))
            