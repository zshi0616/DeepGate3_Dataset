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

input_npz = './QoR_Design_Test/graphs_hop.npz'
output_npz = './QoR_Design_Test/graphs_area.npz'
max_area_size = 512

def get_area_fainout_cone(area_nodes, area_nodes_stats, glo_fanin_list):
    no_nodes = len(area_nodes)
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    glo_to_area = {}
    
    # Get glo_to_area index mapping 
    for k, node_idx in enumerate(area_nodes):
        glo_to_area[node_idx] = k
    
    # Get fanin fanout list
    for k, node_idx in enumerate(area_nodes):
        for fanin in glo_fanin_list[node_idx]:
            if fanin in glo_to_area:
                fanin_list[k].append(glo_to_area[fanin])
                fanout_list[glo_to_area[fanin]].append(k)
    pi_indexs = [] 
    for k in range(no_nodes):
        if len(fanin_list[k]) == 0:
            pi_indexs.append(k)
    po_indexs = []
    for k in range(no_nodes):
        if len(fanout_list[k]) == 0:
            po_indexs.append(k)
            
    # Get level
    forward_level = [0] * no_nodes
    q = copy.deepcopy(pi_indexs)
    while len(q) > 0:
        node_idx = q.pop(0)
        for fanout in fanout_list[node_idx]:
            if forward_level[fanout] < forward_level[node_idx] + 1:
                forward_level[fanout] = forward_level[node_idx] + 1
                q.append(fanout)
    backward_level = [0] * no_nodes
    q = copy.deepcopy(po_indexs)
    while len(q) > 0:
        node_idx = q.pop(0)
        for fanin in fanin_list[node_idx]:
            if backward_level[fanin] < backward_level[node_idx] + 1:
                backward_level[fanin] = backward_level[node_idx] + 1
                q.append(fanin)
    forward_level_list = [[] for _ in range(max(forward_level) + 1)]
    backward_level_list = [[] for _ in range(max(backward_level) + 1)]
    for k in range(no_nodes):
        forward_level_list[forward_level[k]].append(k)
        backward_level_list[backward_level[k]].append(k)
                
    # Get PI PO Cover 
    pi_cover = [[] for _ in range(no_nodes)]
    for level in range(len(forward_level_list)):
        for node_idx in forward_level_list[level]:
            if level == 0:
                pi_cover[node_idx].append(node_idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[node_idx]:
                tmp_pi_cover += pi_cover[pre_k]
            pi_cover[node_idx] += tmp_pi_cover
            pi_cover[node_idx] = list(set(pi_cover[node_idx]))
    po_cover = [[] for _ in range(no_nodes)]
    for level in range(len(backward_level_list)):
        for node_idx in backward_level_list[level]:
            if level == 0:
                po_cover[node_idx].append(node_idx)
            tmp_po_cover = []
            for post_k in fanout_list[node_idx]:
                tmp_po_cover += po_cover[post_k]
            po_cover[node_idx] += tmp_po_cover
            po_cover[node_idx] = list(set(po_cover[node_idx]))
                
    # Get cone
    cone = [[-1] * no_nodes for _ in range(no_nodes)]
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i == j:
                cone[i][j] = 0
                continue
            if len(pi_cover[j]) < len(pi_cover[i]) and forward_level[j] < forward_level[i]:
                j_in_i_fanin = True
                for pi in pi_cover[j]:
                    if pi not in pi_cover[i]:
                        j_in_i_fanin = False
                        break
                if j_in_i_fanin:
                    assert cone[i][j] == -1
                    cone[i][j] = 1
                else:
                    cone[i][j] = 0
            elif len(po_cover[j]) <= len(po_cover[i]) and forward_level[j] > forward_level[i]:
                j_in_i_fanout = True
                for po in po_cover[j]:
                    if po not in po_cover[i]:
                        j_in_i_fanout = False
                        break
                if j_in_i_fanout:
                    assert cone[i][j] == -1
                    cone[i][j] = 2
                else:
                    cone[i][j] = 0
            else:
                cone[i][j] = 0
    
    return cone
    

if __name__ == '__main__':
    circuits = read_npz_file(input_npz)['circuits'].item()
    graphs = {}
    
    for cir_idx, cir_name in enumerate(circuits):
        # if cir_name != 'max':
        #     continue
        
        print('Read circuit: ', cir_name)
        winhop_po = circuits[cir_name]['winhop_po']
        winhop_lev = circuits[cir_name]['winhop_lev']
        winhop_nodes = circuits[cir_name]['winhop_nodes']
        winhop_nodes_stats = circuits[cir_name]['winhop_nodes_stats']
        x_data = circuits[cir_name]['x']
        edge_index = circuits[cir_name]['edge_index'].T
        fanin_list, fanout_list = get_fanin_fanout(x_data, edge_index)
        
        area_nodes = []
        area_lev = []
        g = {}
        
        # Hop merge 
        for level in range(winhop_lev.max() + 1):
            tmp_area = []
            area = []
            hop_idx = 0
            while hop_idx < winhop_po.shape[0]:
                if winhop_lev[hop_idx] == level:
                    tmp_area += winhop_nodes[hop_idx][winhop_nodes_stats[hop_idx] == 1].tolist()
                    tmp_area = list(set(tmp_area))
                    if len(tmp_area) < max_area_size:
                        area = copy.deepcopy(tmp_area)
                    else:
                        area_nodes.append(copy.deepcopy(area))
                        area_lev.append(level)
                        tmp_area = []
                        continue
                hop_idx += 1
            if len(tmp_area) > 0:
                area_nodes.append(copy.deepcopy(tmp_area))
                area_lev.append(level)
        
        # Check coverage
        is_cover = [0] * len(circuits[cir_name]['x'])
        for area in area_nodes:
            for node in area:
                is_cover[node] = 1
        coverage = sum(is_cover) / len(is_cover)
        
        # Area node states 
        area_nodes_stats = []       # 0 - node / 1 - PI / 2 - PO / -1 - padding
        for area_idx in range(len(area_nodes)):
            area = area_nodes[area_idx]
            area_stats = [0] * len(area)
            # Check PI and PO
            for k, node_idx in enumerate(area):
                has_fanin = False
                for fanin in fanin_list[node_idx]:
                    if fanin in area:
                        has_fanin = True
                        break
                if not has_fanin:
                    area_stats[k] = 1
                
                has_fanout = False
                for fanout in fanout_list[node_idx]:
                    if fanout in area:
                        has_fanout = True
                        break
                if not has_fanout:
                    area_stats[k] = 2
            area_nodes_stats.append(area_stats)
        
        # Fanin Fanout cone
        area_fanin_fanout_cone = []
        for area_idx in range(len(area_nodes)):
            area = area_nodes[area_idx]
            cone = get_area_fainout_cone(area, area_nodes_stats[area_idx], fanin_list)
            area_fanin_fanout_cone.append(cone)
            
        # Convert to Matrix 
        for area_idx in range(len(area_nodes)):
            area = area_nodes[area_idx]
            area_nodes[area_idx] = area + [-1] * (max_area_size - len(area))
            area_nodes_stats[area_idx] = area_nodes_stats[area_idx] + [-1] * (max_area_size - len(area))
            cone = area_fanin_fanout_cone[area_idx]
            for i in range(len(cone)):
                cone[i] = cone[i] + [-1] * (max_area_size - len(cone[i]))
            for i in range(len(cone), max_area_size):
                cone.append([-1] * max_area_size)
            area_fanin_fanout_cone[area_idx] = cone
            
        # Save 
        area_nodes = np.array(area_nodes)
        area_nodes_stats = np.array(area_nodes_stats)
        area_lev = np.array(area_lev)
        area_fanin_fanout_cone = np.array(area_fanin_fanout_cone)
        for key in circuits[cir_name]:
            g[key] = circuits[cir_name][key]
        g['area_nodes'] = area_nodes
        g['area_nodes_stats'] = area_nodes_stats
        g['area_lev'] = area_lev
        g['area_fanin_fanout_cone'] = area_fanin_fanout_cone
        
        graphs[cir_name] = copy.deepcopy(g)
            
        print('Parse Circuit: {}, # Nodes: {:}, # Areas: {:}, Coverage: {:.2f}%'.format(
            cir_name, len(x_data), len(area_nodes), coverage * 100
        ))
        print()
        
    np.savez(output_npz, circuits=graphs)
    print('Output:', output_npz)
    print('Total:', len(graphs))