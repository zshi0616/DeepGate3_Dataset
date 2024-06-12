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

EPFL_keys = [
    'adder', 'arbiter', 'bar', 'cavlc', 'ctrl', 'dec', 
    'div', 'hyp', 'i2c', 'int2float', 'log2', 
    'max', 'mem_ctrl', 'multiplier', 'priority', 
    'router', 'sin', 'sqrt', 'square', 'voter'
]

class Data_Parser:
    def __init__(self) -> None:
        self.data = []
    
    def insert(self, num):
        self.data.append(num)
        
    def average(self):
        return sum(self.data) / len(self.data)

    def max(self):
        return max(self.data)

    def min(self):
        return min(self.data)

    def std(self):
        d = np.array(self.data)
        return d.std()
    
    def count(self):
        return len(self.data)

if __name__ == '__main__':
    circuits = read_npz_file(input_path)['circuits'].item()
    no_circuits = len(circuits)
    cir_name_list = list(circuits.keys())

    # statistics 
    itc_nodes = Data_Parser()
    itc_levs = Data_Parser()
    iwls_nodes = Data_Parser()
    iwls_levs = Data_Parser()
    epfl_nodes = Data_Parser()
    epfl_levs = Data_Parser()
    rtl_nodes = Data_Parser()
    rtl_levs = Data_Parser()
    oc_nodes = Data_Parser()
    oc_levs = Data_Parser()
    all_nodes = Data_Parser()
    all_levs = Data_Parser()
    
    for cir_idx, cir_name in enumerate(cir_name_list):
        ckt = circuits[cir_name]
        if len(ckt['x']) > 512:
            continue
        
        no_nodes = len(ckt['x'])
        no_levs = np.max(ckt['forward_level'] + 1)
        all_nodes.insert(no_nodes)
        all_levs.insert(no_levs)
        
        # Datasets
        arr = cir_name.replace('trans_', '').split('_')
        if arr[0] == 'trans':
            arr[0] = arr[1]
        if arr[0].isdigit():                            # RTL
            rtl_nodes.insert(no_nodes)
            rtl_levs.insert(no_levs)
            continue
        if (arr[0][0] == 'c' or arr[0][0] == 's') and arr[0][1:].isdigit():   # IWLS
            iwls_nodes.insert(no_nodes)
            iwls_levs.insert(no_levs)
            continue
        if arr[0][0] == 'b' and arr[0][1:].isdigit():   # ITC99
            itc_nodes.insert(no_nodes)
            itc_levs.insert(no_levs)
            continue
            
        is_epfl = False
        for epfl_key in EPFL_keys:
            if epfl_key in cir_name:
                is_epfl = True
                break
        if is_epfl:                                     # EPFL
            epfl_nodes.insert(no_nodes)
            epfl_levs.insert(no_levs)
            continue
    
        oc_nodes.insert(no_nodes)
        oc_levs.insert(no_levs)
    
    # Display
    print('ITC99: {:}'.format(itc_nodes.count()))
    print('Nodes: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        itc_nodes.average(), itc_nodes.std(), itc_nodes.max(), itc_nodes.min()
    ))
    print('Levs: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        itc_levs.average(), itc_levs.std(), itc_levs.max(), itc_levs.min()
    ))
    print()
    print('IWLS: {:}'.format(iwls_nodes.count()))
    print('Nodes: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        iwls_nodes.average(), iwls_nodes.std(), iwls_nodes.max(), iwls_nodes.min()
    ))
    print('Levs: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        iwls_levs.average(), iwls_levs.std(), iwls_levs.max(), iwls_levs.min()
    ))
    print()
    print('EPFL: {:}'.format(epfl_nodes.count()))
    print('Nodes: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        epfl_nodes.average(), epfl_nodes.std(), epfl_nodes.max(), epfl_nodes.min()
    ))
    print('Levs: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        epfl_levs.average(), epfl_levs.std(), epfl_levs.max(), epfl_levs.min()
    ))
    print()
    print('RTL: {:}'.format(rtl_nodes.count()))
    print('Nodes: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        rtl_nodes.average(), rtl_nodes.std(), rtl_nodes.max(), rtl_nodes.min()
    ))
    print('Levs: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        rtl_levs.average(), rtl_levs.std(), rtl_levs.max(), rtl_levs.min()
    ))
    print()
    print('OC: {:}'.format(oc_nodes.count()))
    print('Nodes: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        oc_nodes.average(), oc_nodes.std(), oc_nodes.max(), oc_nodes.min()
    ))
    print('Levs: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        oc_levs.average(), oc_levs.std(), oc_levs.max(), oc_levs.min()
    ))
    print()
    print('='*20)
    print('All: {:}'.format(all_nodes.count()))
    print('Nodes: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        all_nodes.average(), all_nodes.std(), all_nodes.max(), all_nodes.min()
    ))
    print('Levs: Avg. {:.2f}, Std. {:.2f}, Max {:}, Min {:}'.format(
        all_levs.average(), all_levs.std(), all_levs.max(), all_levs.min()
    ))