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

def get_parse_args():
    parser = argparse.ArgumentParser(description='WinHop Analyzer')
    
    # Input / Output 
    parser.add_argument('--input_npz', type=str, default='./QoR_Design_Test/graphs.npz', help='Input NPZ file path')
    parser.add_argument('--output_npz', type=str, default='./QoR_Design_Test/graphs_hop.npz', help='Output NPZ file path')
    
    # Parameters 
    parser.add_argument('--k_hop', type=int, default=8, help='Number of hops')
    
    args = parser.parse_args()
    return args

def get_winhop(g, k_hop=8, analzyer='cone_analyzer/analyzer', graph_filepath='', res_filepath=''):
    max_hop_size = 0 
    for k in range(k_hop+1):
        max_hop_size += 2**k
    
    if graph_filepath == '':
        graph_filepath = './tmp/tmp_graph_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    if res_filepath == '':
        res_filepath = './tmp/tmp_res_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    # Parse graph
    no_nodes = len(g['forward_index'])
    level_list = [[] for I in range(g['forward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].T:
        fanin_list[edge[1].item()].append(edge[0].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
    # Write graph to file
    f = open(graph_filepath, 'w')
    f.write('{} {} {}\n'.format(no_nodes, len(g['edge_index'][0]), k_hop))
    for idx in range(no_nodes):
        f.write('{} {}\n'.format(g['gate'][idx].item(), g['forward_level'][idx].item()))
    for edge in g['edge_index'].T:
        f.write('{} {}\n'.format(edge[0].item(), edge[1].item()))
    f.close()
    while not os.path.exists(graph_filepath):
        time.sleep(0.1)
        
    # Analyze 
    analyze_cmd = '{} {} {}'.format(analzyer, graph_filepath, res_filepath)
    stdout, _ = run_command(analyze_cmd)
    f = open(res_filepath, 'r')
    lines = f.readlines()
    f.close()
    
    no_hops = int(lines[0].replace('\n', ''))
    all_hop_po = torch.zeros((0, 1), dtype=torch.long)
    all_hop_winlev = torch.zeros((0, 1), dtype=torch.long)
    all_hop_nodes = torch.zeros((0, max_hop_size), dtype=torch.long)
    all_hop_nodes_stats = torch.zeros((0, max_hop_size), dtype=torch.long)
    
    for hop_idx in range(no_hops):
        no_nodes_inhop = int(lines[hop_idx*2+1].replace('\n', '').split(' ')[0])
        sliding_level = int(lines[hop_idx*2+1].replace('\n', '').split(' ')[1])
        arr = lines[hop_idx*2+2].replace('\n', '').split(' ')[:-1]
        hop_nodes = [int(x) for x in arr]
        hop_po = hop_nodes[0]
        assert len(hop_nodes) == no_nodes_inhop
        hop_nodes_stats = [1] * len(hop_nodes) + [0] * (max_hop_size - len(hop_nodes))
        hop_nodes = hop_nodes + [-1] * (max_hop_size - len(hop_nodes))
        
        all_hop_po = torch.cat((all_hop_po, torch.tensor([[hop_po]])), dim=0)
        all_hop_winlev = torch.cat((all_hop_winlev, torch.tensor([[sliding_level]])), dim=0)
        all_hop_nodes = torch.cat((all_hop_nodes, torch.tensor([hop_nodes])), dim=0)
        all_hop_nodes_stats = torch.cat((all_hop_nodes_stats, torch.tensor([hop_nodes_stats])), dim=0)
    
    # Record
    g['winhop_po'] = all_hop_po.numpy()
    g['winhop_lev'] = all_hop_winlev.numpy()
    g['winhop_nodes'] = all_hop_nodes.numpy()
    g['winhop_nodes_stats'] = all_hop_nodes_stats.numpy()
    os.remove(graph_filepath)
    os.remove(res_filepath)
    
    return g
    

if __name__ == '__main__':
    graphs = {}
    args = get_parse_args()
    circuits = read_npz_file(args.input_npz)['circuits'].item()
    for cir_idx, cir_name in enumerate(circuits):
        g = {}
        for key in circuits[cir_name]:
            g[key] = circuits[cir_name][key]
        
        g = get_winhop(g, k_hop=args.k_hop)
        graphs[cir_name] = copy.deepcopy(g)
        print('Read Circuit:', cir_name)
    
    np.savez_compressed(args.output_npz, circuits=graphs)
    print(args.output_npz)
    print(len(graphs))
            