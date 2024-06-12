import numpy as np 
import os 
import copy 
import time
import threading
import torch 
import random
import argparse
from deepgate.utils.data_utils import read_npz_file
import deepgate as dg

import sys 
sys.path.append('/home/zyshi21/studio/DeepGate3-Transformer/src')
from models.dg2 import DeepGate2

input_path = '/home/zyshi21/data/share/wl_4_hop.npz'
output_path = '/home/zyshi21/data/share/all.npz'

if __name__ == '__main__':
    # Tokenizer 
    dg2 = DeepGate2()
    dg2.load_pretrained('/home/zyshi21/studio/DeepGate3-Transformer/trained/model_last_workload.pth')
    
    # Read Npz
    graphs = {}
    circuits = read_npz_file(input_path)['circuits'].item()
    no_circuits = len(circuits)
    cir_name_list = list(circuits.keys())
    tot_time = 0
    
    for cir_idx, cir_name in enumerate(cir_name_list):
        cir = circuits[cir_name]
        print('Parse: {} ({:} / {:}), Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            cir_name, cir_idx, no_circuits, 
            tot_time, tot_time / ((cir_idx + 1) / no_circuits) - tot_time, 
            len(graphs)
        ))
        start_time = time.time()
        
        # update hop_pi and hop_pi_stats 
        for hop_idx in range(len(cir['hop_pi'])):
            hop_pi = cir['hop_pi'][hop_idx]
            hop_pi_stats = cir['hop_pi_stats'][hop_idx]
            
            for m1_pos in range(len(hop_pi)):
                if hop_pi[m1_pos] != -1:
                    break
            hop_pi = np.concatenate([hop_pi[m1_pos:], hop_pi[:m1_pos]], axis=0)
            hop_pi_stats = np.concatenate([hop_pi_stats[m1_pos:], hop_pi_stats[:m1_pos]], axis=0)
            
            if hop_pi_stats[-1] == -1:
                hop_pi_stats[-1] = -2
            if hop_pi_stats[-2] == -1:
                hop_pi_stats[-2] = -2
            
            cir['hop_pi'][hop_idx] = hop_pi
            cir['hop_pi_stats'][hop_idx] = hop_pi_stats
        
        # dg2
        x_data = []
        for idx in range(len(cir['x'])):
            if cir['x'][idx][0] == 1:
                gate_type = 0
            elif cir['x'][idx][1] == 1:
                gate_type = 1
            elif cir['x'][idx][2] == 1:
                gate_type = 2
            x_data.append([idx, gate_type])
        x_data = np.array(x_data)
        g = dg.parse_pyg_mlpgate(
            x_data, cir['edge_index'].T, [], [], cir['prob'], 
            [], []
        )
        prob = torch.tensor(cir['prob'])
        hs, hf = dg2(g, prob)

        # Save
        g = {}
        for key in cir:
            g[key] = cir[key]
        g['hs'] = hs.detach().numpy()
        g['hf'] = hf.detach().numpy()
        graphs[cir_name] = copy.deepcopy(g)
        print('Read Circuit:', cir_name)
        
        tot_time += time.time() - start_time
            
    np.savez(output_path, circuits=graphs)
    print(output_path)
    print(len(graphs))
    