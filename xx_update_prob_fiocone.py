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
import utils.circuit_utils as circuit_utils

MAX_NO_NODES = 512

def get_parse_args():
    parser = argparse.ArgumentParser(description='WinHop Analyzer')
    
    # Input / Output 
    parser.add_argument('--input_npz', type=str, default='./dataset/4_hop.npz', help='Input NPZ file path')
    parser.add_argument('--output_npz', type=str, default='./dataset/wl_4_hop.npz', help='Output NPZ file path')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    graphs = {}
    args = get_parse_args()
    circuits = read_npz_file(args.input_npz)['circuits'].item()
    no_circuits = len(circuits)
    tot_time = 0
    for cir_idx, cir_name in enumerate(circuits):
        print('Parse: {} ({:} / {:}), Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            cir_name, cir_idx, no_circuits, 
            tot_time, tot_time / ((cir_idx + 1) / no_circuits) - tot_time, 
            len(graphs)
        ))
        
        start_time = time.time()
        g = {}
        for key in circuits[cir_name]:
            g[key] = circuits[cir_name][key]
            
        if len(g['gate']) > MAX_NO_NODES:
            print('Skip Circuit:', cir_name)
            continue
        
        prob = circuit_utils.prepare_workload_prob(g, 15000)
        fanin_fanout_cones = circuit_utils.get_fanin_fanout_cone(g, max_no_nodes=MAX_NO_NODES)
        g['prob'] = prob
        g['fanin_fanout_cones'] = fanin_fanout_cones.numpy()
        
        graphs[cir_name] = copy.deepcopy(g)
        print('Read Circuit:', cir_name)
        
        tot_time += time.time() - start_time
    
    np.savez(args.output_npz, circuits=graphs)
    print(args.output_npz)
    print(len(graphs))
            