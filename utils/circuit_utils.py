'''
Utility functions for circuit: including random pattern generation, logic simulator, \
    reconvergence identification, 
Author: Sadaf Khan, Zhengyuan Shi and Min Li.
'''
# import torch
from numpy.random import randint
import copy
from collections import Counter
import random 
import torch
import os 
import numpy as np
import networkx as nx
from utils.utils import hash_arr, run_command
import deepgate as dg
import time
import threading

def remove_unconnected(x_data, edge_index):
    new_x_data = []
    new_edge_index = []
    is_connected = [False] * len(x_data)
    for edge in edge_index:
        is_connected[edge[0]] = True
        is_connected[edge[1]] = True
    
    new_map = {}
    for idx, is_c in enumerate(is_connected):
        if is_c:
            new_map[idx] = len(new_x_data)
            new_x_data.append(x_data[idx])
    for edge in edge_index:
        new_edge_index.append([new_map[edge[0]], new_map[edge[1]]])
    
    new_x_data = np.array(new_x_data)
    return new_x_data, new_edge_index

def read_file(file_name):
    f = open(file_name, "r")
    data = f.readlines()
    return data

def random_pattern_generator(no_PIs):
    vector = [0] * no_PIs

    vector = randint(2, size=no_PIs)
    return vector


def logic(gate_type, signals):
    if gate_type == 1:  # AND
        for s in signals:
            if s == 0:
                return 0
        return 1

    elif gate_type == 2:  # NAND
        for s in signals:
            if s == 0:
                return 1
        return 0

    elif gate_type == 3:  # OR
        for s in signals:
            if s == 1:
                return 1
        return 0

    elif gate_type == 4:  # NOR
        for s in signals:
            if s == 1:
                return 0
        return 1

    elif gate_type == 5:  # NOT
        for s in signals:
            if s == 1:
                return 0
            else:
                return 1

    # elif gate_type == 6:  # BUFF
    #  for s in signals:
    #      return s

    elif gate_type == 6:  # XOR
        z_count = 0
        o_count = 0
        for s in signals:
            if s == 0:
                z_count = z_count + 1
            elif s == 1:
                o_count = o_count + 1
        if z_count == len(signals) or o_count == len(signals):
            return 0
        return 1

def prob_logic(gate_type, signals):
    '''
    Function to calculate Controlability values, i.e. C1 and C0 for the given node.
    Modified by Min.
    ...
    Parameters:
        gate_type: int, the integer index for the target node.
        signals : list(float), the values for the fan-in signals
    Return:
        zero: float, C0
        one: flaot, C1
    '''
    one = 0.0
    zero = 0.0

    if gate_type == 1:  # AND
        mul = 1.0
        for s in signals:
            mul = mul * s[1]
        one = mul
        zero = 1.0 - mul

    elif gate_type == 2:  # NAND
        mul = 1.0
        for s in signals:
            mul = mul * s[1]
        zero = mul
        one = 1.0 - mul

    elif gate_type == 3:  # OR
        mul = 1.0
        for s in signals:
            mul = mul * s[0]
        zero = mul
        one = 1.0 - mul

    elif gate_type == 4:  # NOR
        mul = 1.0
        for s in signals:
            mul = mul * s[0]
        one = mul
        zero = 1.0 - mul

    elif gate_type == 5:  # NOT
        for s in signals:
            one = s[0]
            zero = s[1]

    elif gate_type == 6:  # XOR
        mul0 = 1.0
        mul1 = 1.0
        for s in signals:
            mul0 = mul0 * s[0]
        for s in signals:
            mul1 = mul1 * s[1]

        zero = mul0 + mul1
        one = 1.0 - zero

    return zero, one


# TODO: correct observability logic
def obs_prob(x, r, y, input_signals):
    if x[r][1] == 1 or x[r][1] == 2:
        obs = y[r]
        for s in input_signals:
            for s1 in input_signals:
                if s != s1:
                    obs = obs * x[s1][3]
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif x[r][1] == 3 or x[r][1] == 4:
        obs = y[r]
        for s in input_signals:
            for s1 in input_signals:
                if s != s1:
                    obs = obs * x[s1][4]
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif x[r][1] == 5:
        obs = y[r]
        for s in input_signals:
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif x[r][1] == 6:
        if len(input_signals) != 2:
            print('Not support non 2-input XOR Gate')
            raise
        # computing for a node
        obs = y[r]
        s = input_signals[1]
        if x[s][3] > x[s][4]:
            obs = obs * x[s][3]
        else:
            obs = obs * x[s][4]
        y[input_signals[0]] = obs

        # computing for b node
        obs = y[r]
        s = input_signals[0]
        if x[s][3] > x[s][4]:
            obs = obs * x[s][3]
        else:
            obs = obs * x[s][4]
        y[input_signals[1]] = obs

    return y



def simulator(x_data, PI_indexes, level_list, fanin_list, num_patterns):
    '''
       Logic simulator
       Modified by Zhengyuan 27-09-2021
       ...
       Parameters:
           x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1, 4th - C0, 5th - Obs.
           level_list: logic levels
           fanin_list: the fanin node indexes for each node
           fanout_list: the fanout node indexes for each node
       Return:
           y_data : simualtion result
       '''
    y = [0] * len(x_data)
    y1 = [0] * len(x_data)
    pattern_count = 0
    no_of_patterns = min(num_patterns, 10 * pow(2, len(PI_indexes)))
    print('No of Patterns: {:}'.format(no_of_patterns))

    print('[INFO] Begin simulation')
    while pattern_count < no_of_patterns:
        input_vector = random_pattern_generator(len(PI_indexes))

        j = 0
        for i in PI_indexes:
            y[i] = input_vector[j]
            j = j + 1

        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(y[pre_idx])
                if len(source_signals) > 0:
                    gate_type = x_data[node_idx][1]
                    y[node_idx] = logic(gate_type, source_signals)
                    if y[node_idx] == 1:
                        y1[node_idx] = y1[node_idx] + 1

        pattern_count = pattern_count + 1
        if pattern_count % 10000 == 0:
            print("pattern count = {:}k".format(int(pattern_count / 1000)))

    for i, _ in enumerate(y1):
        y1[i] = [y1[i] / pattern_count]

    for i in PI_indexes:
        y1[i] = [0.5]

    return y1



def get_gate_type(line, gate_to_index):
    '''
    Function to get the interger index of the gate type.
    Modified by Min.
    ...
    Parameters:
        line : str, the single line in the bench file.
        gate_to_index: dict, the mapping from the gate name to the integer index
    Return:
        vector_row : int, the integer index for the gate. Currently consider 7 gate types.
    '''
    vector_row = -1
    for (gate_name, index) in gate_to_index.items():
        if gate_name  in line:
            vector_row = index

    if vector_row == -1:
        raise KeyError('[ERROR] Find unsupported gate')

    return vector_row


def add_node_index(data):
    '''
    A pre-processing function to handle with the `.bench` format files.
    Will add the node index before the line, and also calculate the total number of nodes.
    Modified by Min.
    ...
    Parameters:
        data : list(str), the lines read out from a bench file
    Return:
        data : list(str), the updated lines for a circuit
        node_index: int, the number of the circuits, not considering `OUTPUT` lines.
        index_map: dict(int:int), the mapping from the original node name to the updated node index.
    '''
    node_index = 0
    index_map = {}

    for i, val in enumerate(data):
        # node level and index  for PI
        if "INPUT" in val:
            node_name = val.split("(")[1].split(")")[0]
            index_map[node_name] = str(node_index)
            data[i] = str(node_index) + ":" + val[:-1] #+ ";0"
            node_index += 1
            

        # index for gate nodes
        if ("= NAND" in val) or ("= NOR" in val) or ("= AND" in val) or ("= OR" in val) or (
                "= NOT" in val) or ("= XOR" in val):
            node_name = val.split(" = ")[0]
            index_map[node_name] = str(node_index)
            data[i] = str(node_index) + ":" + val[:-1]
            node_index += 1

    return data, node_index, index_map

def new_node(name2idx, x_data, node_name, gate_type):
    x_data.append([node_name, gate_type])
    name2idx[node_name] = len(name2idx)

def feature_generation(data, gate_to_index):
    '''
        A pre-processing function to handle with the modified `.bench` format files.
        Will generate the necessary attributes, adjacency matrix, edge connectivity matrix, etc.
        Modified by Zhengyuan 27-09-2021
        Modified by Zhengyuan 13-10-2021
            fixed bug: the key word of gates should be 'OR(' instead of 'OR',
            because variable name may be 'MEMORY' has 'OR'
        ...
        Parameters:
            data : list(str), the lines read out from a bench file (after modified by `add_node_index`)
            gate_to_index: dict(str:int), the mapping from the gate name to the gate index.
        Return:
            x_data: list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
            edge_index_data: list(list(int)), the connectivity matrix wiht shape of [num_edges, 2]
            level_list: logic level [max_level + 1, xx]
            fanin_list: the fanin node indexes for each node
            fanout_list: the fanout node indexes for each node
    '''
    name2idx = {}
    node_cnt = 0
    x_data = []
    edge_index_data = []

    for line in data:
        if 'INPUT(' in line:
            node_name = line.split("(")[-1].split(')')[0]
            new_node(name2idx, x_data, node_name, get_gate_type('INPUT', gate_to_index))
        elif 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '')
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '')
            new_node(name2idx, x_data, node_name, get_gate_type(gate_type, gate_to_index))

    for line_idx, line in enumerate(data):
        if 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '')
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '')
            src_list = line.split('(')[-1].split(')')[0].replace(' ', '').split(',')
            dst_idx = name2idx[node_name]
            for src_node in src_list:
                src_node_idx = name2idx[src_node]
                edge_index_data.append([src_node_idx, dst_idx])

    fanout_list = []
    fanin_list = []
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
        if x_data_info[1] == 0:
            bfs_q.append(idx)
            x_data_level[idx] = 0
    for edge in edge_index_data:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])
    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
    if -1 in x_data_level:
        print('Wrong')
        raise
    else:
        if max_level == 0:
            level_list = [[]]
        else:
            for idx in range(len(x_data)):
                x_data[idx].append(x_data_level[idx])
                level_list[x_data_level[idx]].append(idx)
    return x_data, edge_index_data, level_list, fanin_list, fanout_list

def rename_node(x_data):
    '''
    Convert the data[0] (node name : str) to the index (node index: int)
    ---
    Parameters:
        x_data: list(list(xx)), the node feature matrix
    Return:
        x_data: list(list(xx)), the node feature matrix
    '''
    for idx, x_data_info in enumerate(x_data):
        x_data[idx][0] = int(idx)
    return x_data

def circuit_extraction(x_data, adj, circuit_depth, num_nodes, sub_circuit_size=25):
    '''
    Function to extract several subcircuits from the original circuit.
    Modified by Min.
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        adj : list(list(int)), the adjacency matrix, adj[i][j] = {e(j, i) is in E} 
        circuit_depth : int, the logic depth of the circuit
        num_nodes : int, the total number of nodes in the circuit
        sub_circuit_size: int, the maximum size of the sub-circuits
    Return:
        sub_circuits_x_data : 
        sub_circuits_edges : 
        matrices : 
        
    '''
    adjs = []
    sub_circuits_x_data = []
    sub_circuits_edges = []
    sub_circuits_PIs = []
    sub_circuits_PIs = []

    iterations = 0
    # the current minmium level for the sub-circuit
    min_circuit_level = 0
    # the current maximum level for the sub-circuit
    max_circuit_level = sub_circuit_size

    # init level list
    level_lst = [[] for _ in range(circuit_depth)]

    # level_lis[i] contains the indices for nodes under this logic level
    for idx, node_data in enumerate(x_data):
        level_lst[node_data[2]].append(idx)

    # init predecessor list
    pre_lst = [[] for _ in range(num_nodes)]

    for col_idx, col in enumerate(adj):
        for row_idx, ele in enumerate(col):
            if ele == 1:
                pre_lst[col_idx].append(row_idx)

    while max_circuit_level <= circuit_depth:

        sub_x_data, sub_edges, sub_PIs = generate_sub_circuit(x_data, min_circuit_level, max_circuit_level - 1, level_lst, pre_lst)

        # adj_sub = [ [0] *  len(sub_x_data) ] * len(sub_x_data)
        adj_sub = [[0 for _ in range(len(sub_x_data))] for _ in range(len(sub_x_data))]
        for edge_data in sub_edges:
            adj_sub[edge_data[1]][edge_data[0]] = 1

        adjs.append(adj_sub)

        sub_circuits_x_data.append(sub_x_data)
        sub_circuits_edges.append(sub_edges)
        sub_circuits_PIs.append(sub_PIs)

        min_circuit_level = max_circuit_level
        max_circuit_level += sub_circuit_size

        if (max_circuit_level > circuit_depth > min_circuit_level) and (min_circuit_level != (circuit_depth - 1)):

            sub_x_data, sub_edges, sub_PIs = generate_sub_circuit(x_data, min_circuit_level, max_circuit_level - 1,
                                                                  level_lst, pre_lst)

            # adj_sub = [[0] * len(sub_x_data)] * len(sub_x_data)
            adj_sub = [[0 for x in range(sub_x_data)] for y in range(sub_x_data)]
            for edge_data in sub_edges:
                adj_sub[edge_data[1]][edge_data[0]] = 1

            adjs.append(adj_sub)

            sub_circuits_x_data.append(sub_x_data)
            sub_circuits_edges.append(sub_edges)
            sub_circuits_PIs.append(sub_PIs)
    return sub_circuits_x_data, sub_circuits_edges, adjs, sub_circuits_PIs


def generate_sub_circuit(x_data, min_circuit_level, max_circuit_level, level_lst, pre_lst):
    '''
    Function to extract a sub-circuit from the original circuit using the logic level information.
    Modified by Min.
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        min_circuit_level : int, the current minmium level for the sub-circuit
        max_circuit_level: int, the maximum size of the sub-circuits
        level_lst : list(list(int)), level_lis[i] contains the indices for nodes under this logic level
        pre_lst : list(list(int)), pre_lst[i] contains the indices for predecessor nodes for the i-th node.
    Return:
        sub_x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        sub_edge : list(list(int)), the connectivity matrix wiht shape of [num_edges, 2]
        sub_pi_indexes : list(int), the index for the primary inputs.
    '''
    sub_x_data = []
    sub_pi_indexes = []
    # the list that contains node indices for the extracted logic range.
    sub_node = []
    sub_edge = []
    x_data_tmp = copy.deepcopy(x_data)

    # Picking all nodes in desired depth
    for level in range(min_circuit_level, max_circuit_level + 1):
        if level < len(level_lst):
            for node in level_lst[level]:
                sub_node.append(node)

    # Update logic level
    for idx in sub_node:
        x_data_tmp[idx][2] = x_data_tmp[idx][2] - (min_circuit_level)

    # Separate PI and Gate
    PIs = []
    Gates = []
    for idx in sub_node:
        if x_data_tmp[idx][2] == 0:
            x_data_tmp[idx][1] = 0
            PIs.append(idx)
        else:
            Gates.append(idx)

    # Search subcircuit edge
    for idx in Gates:
        for pre_idx in pre_lst[idx]:
            sub_edge.append([pre_idx, idx])
            # Insert new PI. mli: consider the corner cases that there are some internal nodes connected to the predecessors that are located in the level less than min_circuit_level
            if x_data[pre_idx][2] < min_circuit_level:
                x_data_tmp[pre_idx][1] = 0
                x_data_tmp[pre_idx][2] = 0
                PIs.append(pre_idx)
                sub_node.append(pre_idx)

    # Ignore the no edge node
    node_mask = [0] * len(x_data)
    for edge in sub_edge:
        node_mask[edge[0]] = 1
        node_mask[edge[1]] = 1

    # Map to subcircuit index
    sub_node = list(set(sub_node))
    sub_node = sorted(sub_node, key=lambda x: x_data[x][2])
    sub_cnt = 0
    ori2sub_map = {}  # Original index map to subcircuit
    for node_idx in sub_node:
        if node_mask[node_idx] == 1:
            sub_x_data.append(x_data_tmp[node_idx].copy())
            ori2sub_map[node_idx] = sub_cnt
            sub_cnt += 1
    for edge_idx, edge in enumerate(sub_edge):
        sub_edge[edge_idx] = [ori2sub_map[edge[0]], ori2sub_map[edge[1]]]
    for pi_idx in PIs:
        if node_mask[pi_idx] == 1:
            sub_pi_indexes.append(ori2sub_map[pi_idx])

    return sub_x_data, sub_edge, sub_pi_indexes


def generate_prob_cont(x_data, PI_indexes, level_list, fanin_list):
    '''
    Function to calculate Controlability values, i.e. C1 and C0 for the nodes.
    Modified by Zhengyuan, 27-09-2021
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
        PI_indexes : list(int), the indices for the primary inputs
        level_list: logic levels
        fanin_list: the fanin node indexes for each node
    Return:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0.
    '''
    y = [0] * len(x_data)

    for i in PI_indexes:
        y[i] = [0.5, 0.5]

    for level in range(1, len(level_list), 1):
        for idx in level_list[level]:
            source_node = fanin_list[idx]
            source_signals = []
            for node in source_node:
                source_signals.append(y[node])
            if len(source_signals) > 0:
                zero, one = prob_logic(x_data[idx][1], source_signals)
                y[idx] = [zero, one]

    for i, prob in enumerate(y):
        x_data[i].append(prob[1])
        x_data[i].append(prob[0])

    return x_data


def generate_prob_obs(x_data, level_list, fanin_list, fanout_list):
    '''
        Function to calculate Observability values, i.e. CO.
        Modified by Zhengyuan, 27-09-2021
        ...
        Parameters:
            x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
            level_list: logic levels
            fanin_list: the fanin node indexes for each node
            fanout_list: the fanout node indexes for each node
        Return:
            x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0; 5th - CO.
        '''
    # Array node into level_list

    y = [-1] * len(x_data)

    POs_indexes = []
    for idx, nxt in enumerate(fanout_list):
        if len(nxt) == 0:
            POs_indexes.append(idx)
            y[idx] = 1

    for level in range(len(level_list) - 1, -1, -1):
        for idx in level_list[level]:
            source_signals = fanin_list[idx]
            if len(source_signals) > 0:
                y = obs_prob(x_data, idx, y, source_signals)

    for i, val in enumerate(y):
        x_data[i].append(val)

    return x_data


def dfs_reconvergent_circuit(node_idx, vis, dst_idx, fanout_list, result, x_data):
    if node_idx == dst_idx:
        result += vis
        return
    for nxt_idx in fanout_list[node_idx]:
        if x_data[nxt_idx][2] <= x_data[dst_idx][2]:
            vis.append(nxt_idx)
            dfs_reconvergent_circuit(nxt_idx, vis, dst_idx, fanout_list, result, x_data)
            vis = vis[:-1]
    return result


def identify_reconvergence(x_data, level_list, fanin_list, fanout_list):
    '''
    Function to identify the reconvergence nodes in the given circuit.
    The algorithm is done under the principle that we only consider the minimum reconvergence structure.
    Modified by Zhengyuan 27-09-2021
    ...
    Parameters:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1, 4th - C0, 5th - Obs.
        level_list: logic levels
        fanin_list: the fanin node indexes for each node
        fanout_list: the fanout node indexes for each node
    Return:
        x_data : list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0; 5th - Obs; 6th - fan-out, 7th - boolean recovengence, 8th - index of the source node (-1 for non recovengence).
        rc_lst: list(int), the index for the reconvergence nodes
    '''
    for idx, node in enumerate(x_data):
        if len(fanout_list[idx]) > 1:
            x_data[idx].append(1)
        else:
            x_data[idx].append(0)

        # fanout list (FOL)
    FOL = []
    fanout_num = []
    is_del = []
    # RC (same as reconvergence_nodes)
    rc_lst = []
    max_level = 0
    for x_data_info in x_data:
        if x_data_info[2] > max_level:
            max_level = x_data_info[2]
        FOL.append([])
    for idx, x_data_info in enumerate(x_data):
        fanout_num.append(len(fanout_list[idx]))
        is_del.append(False)

    for level in range(max_level + 1):
        if level == 0:
            for idx in level_list[0]:
                x_data[idx].append(0)
                x_data[idx].append(-1)
                if x_data[idx][6]:
                    FOL[idx].append(idx)
        else:
            for idx in level_list[level]:
                FOL_tmp = []
                FOL_del_dup = []
                save_mem_list = []
                for pre_idx in fanin_list[idx]:
                    if is_del[pre_idx]:
                        print('[ERROR] This node FOL has been deleted to save memory')
                        raise
                    FOL_tmp += FOL[pre_idx]
                    fanout_num[pre_idx] -= 1
                    if fanout_num[pre_idx] == 0:
                        save_mem_list.append(pre_idx)
                for save_mem_idx in save_mem_list:
                    FOL[save_mem_idx].clear()
                    is_del[save_mem_idx] = True
                FOL_cnt_dist = Counter(FOL_tmp)
                source_node_idx = 0
                source_node_level = -1
                is_rc = False
                for dist_idx in FOL_cnt_dist:
                    FOL_del_dup.append(dist_idx)
                    if FOL_cnt_dist[dist_idx] > 1:
                        is_rc = True
                        if x_data[dist_idx][2] > source_node_level:
                            source_node_level = x_data[dist_idx][2]
                            source_node_idx = dist_idx
                if is_rc:
                    x_data[idx].append(1)
                    x_data[idx].append(source_node_idx)
                    rc_lst.append(idx)
                else:
                    x_data[idx].append(0)
                    x_data[idx].append(-1)

                FOL[idx] = FOL_del_dup
                if x_data[idx][6]:
                    FOL[idx].append(idx)
    del (FOL)

    # for node in range(len(x_data)):
    #     x_data[node].append(0)
    # for rc_idx in rc_lst:
    #     x_data[rc_idx][-1] = 1

    return x_data, rc_lst


def backward_search(node_idx, fanin_list, x_data, min_level):
    if x_data[node_idx][2] <= min_level:
        return []
    result = []
    for pre_node in fanin_list[node_idx]:
        if pre_node not in result:
            l = [pre_node]
            res = backward_search(pre_node, fanin_list, x_data, min_level)
            result = result + l + list(set(res))
        else:
            l = [pre_node]
            result = result + l
    return result


def check_reconvergence(x_data, edge_list):
    pre_lst = []
    for node in x_data:
        pre_lst.append([])
    for edge in edge_list:
        pre_lst[edge[1]].append(edge[0])

    for idx, node in enumerate(x_data):
        if node[-2] == 1:
            source_level = x_data[node[-1]][2]
            vis_list = backward_search(idx, pre_lst, x_data, source_level)
            vis_cnt_dist = Counter(vis_list)
            find_source = False
            for dist_idx in vis_cnt_dist:
                if vis_cnt_dist[dist_idx] > 1:
                    find_source = True
                    if x_data[dist_idx][2] > source_level:
                        print("[ERROR] Not the nearest source")
                        raise
            if not find_source:
                print("[ERROR] No source node find")
                raise




def circuit_statistics(circuit_name, x_data, edge_list):
    print('================== Statistics INFO ==================')
    print('Circuit Name: {}'.format(circuit_name))
    print('Number of Nodes: {}'.format(len(x_data)))
    gate_type_cnt = [0] * 10
    gate_type = []
    for x_data_info in x_data:
        gate_type_cnt[x_data_info[1]] += 1
    for k in range(10):
        if gate_type_cnt[k] > 0:
            gate_type.append(k)
    print('Number of Gate Types: {}'.format(len(gate_type)))
    print('Gate: ', gate_type)

    # gate level difference
    level_diff = []
    for node_idx, node_info in enumerate(x_data):
        if node_info[-2] == 1:
            level_diff.append([node_idx, node_info[-1], x_data[node_idx][2] - x_data[node_info[-1]][2]])
    level_diff = sorted(level_diff, key=lambda x: x[-1])
    if level_diff == []:
        print('No reconvergent node')
    else:
        print('Max level = {:}, from {} to {}'.format(level_diff[-1][2],
                                                      x_data[level_diff[-1][0]][0], x_data[level_diff[-1][1]][0]))
        print('Min level = {:}, from {} to {}'.format(level_diff[0][2],
                                                      x_data[level_diff[0][0]][0], x_data[level_diff[0][1]][0]))

    # reconvergent area
    fanout_list = []
    rc_cnt = 0
    for idx, node_info in enumerate(x_data):
        fanout_list.append([])
        if node_info[-2] == 1:
            rc_cnt += 1
    for edge in edge_list:
        fanout_list[edge[0]].append(edge[1])
    rc_gates = []
    for node_idx, node_info in enumerate(x_data):
        if node_info[-2] == 1:
            src_idx = node_info[-1]
            dst_idx = node_idx
            rc_gates += dfs_reconvergent_circuit(src_idx, [src_idx], dst_idx, fanout_list, [], x_data)
    rc_gates_merged = list(set(rc_gates))
    print('Reconvergent nodes: {:}/{:} = {:}'.format(rc_cnt, len(x_data),
                                                     rc_cnt / len(x_data)))
    print('Reconvergent area: {:}/{:} = {:}'.format(len(rc_gates_merged), len(x_data),
                                                    len(rc_gates_merged) / len(x_data)))


def check_difference(dataset):
    diff = 0
    tot = 0
    for g in dataset:
        diff += torch.sum(torch.abs((g.c1 - g.gt)))
        tot += g.c1.size(0)
    print('Average difference between C1 and GT is: ', (diff/tot).item())
    diff = 0
    tot = 0
    for g in dataset:
        diff += torch.sum(torch.abs((g.c1 - g.gt)) * g.rec)
        tot += torch.sum(g.rec)
    print('Average difference between C1 and GT (reconvergent nodes) is: ', (diff/tot).item())
    diff = 0
    tot = 0
    for g in dataset:
        diff += torch.sum(torch.abs((g.c1 - g.gt)) * (1- g.rec))
        tot += torch.sum(1 - g.rec)
    print('Average difference between C1 and GT (non-reconvergent nodes) is: ', (diff/tot).item())


def aig_simulation(x_data, edge_index_data, num_patterns=15000):
    fanout_list = []
    fanin_list = []
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
    for edge in edge_index_data:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])

    PI_indexes = []
    for idx, ele in enumerate(fanin_list):
        if len(ele) == 0:
            PI_indexes.append(idx)
            x_data_level[idx] = 0
            bfs_q.append(idx)

    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
    for idx, ele in enumerate(x_data_level):
        level_list[ele].append(idx)

    ######################
    # Simulation
    ######################
    y = [0] * len(x_data)
    y1 = [0] * len(x_data)
    pattern_count = 0
    no_of_patterns = min(num_patterns, 10 * pow(2, len(PI_indexes)))
    print('No of Patterns: {:}'.format(no_of_patterns))
    print('[INFO] Begin simulation')
    while pattern_count < no_of_patterns:
        input_vector = random_pattern_generator(len(PI_indexes))
        j = 0
        for i in PI_indexes:
            y[i] = input_vector[j]
            j = j + 1
        for level in range(1, len(level_list), 1):
            for node_idx in level_list[level]:
                source_signals = []
                for pre_idx in fanin_list[node_idx]:
                    source_signals.append(y[pre_idx])
                if len(source_signals) > 0:
                    if int(x_data[node_idx][0][1]) == 1:
                        gate_type = 1
                    elif int(x_data[node_idx][0][2]) == 1:
                        gate_type = 5
                    else:
                        raise("This is PI")
                    y[node_idx] = logic(gate_type, source_signals)
                    if y[node_idx] == 1:
                        y1[node_idx] = y1[node_idx] + 1

        pattern_count = pattern_count + 1
        if pattern_count % 10000 == 0:
            print("pattern count = {:}k".format(int(pattern_count / 1000)))

    for i, _ in enumerate(y1):
        y1[i] = [y1[i] / pattern_count]

    for i in PI_indexes:
        y1[i] = [0.5]

    return y1

def get_level(x_data, fanin_list, fanout_list):
    bfs_q = []
    x_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        if len(fanout_list[idx]) == 0 and len(fanin_list[idx]) != 0:
            bfs_q.append(idx)
            x_level[idx] = 0
    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_level[idx] + 1
        for next_node in fanin_list[idx]:
            if x_level[next_node] < tmp_level:
                x_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_level[next_node] > max_level:
                    max_level = x_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
        
    if max_level == 0:
        level_list = [[]]
    else:
        for idx in range(len(x_data)):
            level_list[x_level[idx]].append(idx)
    return level_list

def get_fanin_fanout(x_data, edge_index):
    fanout_list = []
    fanin_list = []
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
    for edge in edge_index:
        fanout_list[edge[0]].append(edge[1])
        fanin_list[edge[1]].append(edge[0])
    return fanin_list, fanout_list


def feature_gen_connect(data, gate_to_index):
    '''
        A pre-processing function to handle with the modified `.bench` format files.
        Will generate the necessary attributes, adjacency matrix, edge connectivity matrix, etc.
        Modified by Stone 27-09-2021
        Modified by Stone 13-10-2021
            fixed bug: the key word of gates should be 'OR(' instead of 'OR',
            because variable name may be 'MEMORY' has 'OR'
        ...
        Parameters:
            data : list(str), the lines read out from a bench file (after modified by `add_node_index`)
            gate_to_index: dict(str:int), the mapping from the gate name to the gate index.
        Return:
            x_data: list(list((str, int, int))), the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 3, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level.
            edge_index_data: list(list(int)), the connectivity matrix wiht shape of [num_edges, 2]
            level_list: logic level [max_level + 1, xx]
            fanin_list: the fanin node indexes for each node
            fanout_list: the fanout node indexes for each node
    '''
    name2idx = {}
    node_cnt = 0
    x_data = []
    edge_index_data = []

    for line in data:
        if 'INPUT(' in line:
            node_name = line.split("(")[-1].split(')')[0]
            new_node(name2idx, x_data, node_name, get_gate_type('PI', gate_to_index))
        elif 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line or 'BUF(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '')
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '')
            new_node(name2idx, x_data, node_name, get_gate_type(gate_type, gate_to_index))

    for line_idx, line in enumerate(data):
        if 'AND(' in line or 'NAND(' in line or 'OR(' in line or 'NOR(' in line \
                or 'NOT(' in line or 'XOR(' in line or 'BUF(' in line:
            node_name = line.split(':')[-1].split('=')[0].replace(' ', '')
            gate_type = line.split('=')[-1].split('(')[0].replace(' ', '')
            src_list = line.split('(')[-1].split(')')[0].replace(' ', '').split(',')
            dst_idx = name2idx[node_name]
            for src_node in src_list:
                src_node_idx = name2idx[src_node]
                edge_index_data.append([src_node_idx, dst_idx])
    
    return x_data, edge_index_data

def feature_gen_level(x_data, fanout_list, gate_to_index):
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        if x_data_info[1] == gate_to_index['PI'] or x_data_info[1] == gate_to_index['DFF'] or x_data_info[1] == gate_to_index['LATCH']:
            bfs_q.append(idx)
            x_data_level[idx] = 0
    while len(bfs_q) > 0:
        idx = bfs_q.pop(0)
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data[next_node][1] == gate_to_index['DFF'] or x_data[next_node][1] == gate_to_index['LATCH']:
                continue
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.append(next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
    
    for idx, x_data_info in enumerate(x_data):
        if x_data_level[idx] == -1:
            print('[ERROR] Find unconnected node')
            raise

    if max_level == 0:
        level_list = [[]]
    else:
        for idx in range(len(x_data)):
            level_list[x_data_level[idx]].append(idx)
            x_data[idx].append(x_data_level[idx])
    return x_data, level_list

def parse_bench(file, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}, MAX_LENGTH=-1):
    data = read_file(file)
    data, num_nodes, _ = add_node_index(data)
    if MAX_LENGTH > 0 and num_nodes > MAX_LENGTH:
        return [], [], [], [], []
    data, edge_data = feature_gen_connect(data, gate_to_index)
    fanin_list, fanout_list = get_fanin_fanout(data, edge_data)
    data, level_list = feature_gen_level(data, fanout_list)
    data = rename_node(data)
    return data, edge_data, fanin_list, fanout_list, level_list

def get_ff_connection(x_data, fanin_list, fanout_list, level_list): 
    fpi_list = []
    ff_fanin_list = []
    ff_fanout_list = []
    
    for idx, x_data_info in enumerate(x_data):
        fpi_list.append([])
        ff_fanin_list.append([])
        ff_fanout_list.append([])
        if x_data_info[1] == 0 or x_data_info[1] == 3:
            fpi_list[idx].append(idx)
    
    for level in range(len(level_list)):
        for idx in level_list[level]:
            # Update fanout_idx
            for fanin_idx in fanin_list[idx]:
                fpi_list[idx] += fpi_list[fanin_idx]
            fpi_list[idx] = list(set(fpi_list[idx]))
                
    for idx, x_data_info in enumerate(x_data):
        if x_data[idx][1] == 3:
            comb_idx = fanin_list[idx][0]
            for ff_idx in fpi_list[comb_idx]:
                ff_fanin_list[idx].append(ff_idx)
                ff_fanout_list[ff_idx].append(idx)
                
    # Detect PO
    po_ff_list = []
    for idx, x_data_info in enumerate(x_data):
        if len(fanin_list[idx]) > 0 and len(fanout_list[idx]) == 0:
            po_ff_list += fpi_list[idx]
    po_ff_list = list(set(po_ff_list))        
    
    return ff_fanin_list, ff_fanout_list

def has_loop(x_data, ff_fanin_list, ff_fanout_list, idx, fanin_idx):
    vis = [0] * len(x_data)
    q = [idx]
    vis[idx] = True
    while len(q) > 0:
        cur_idx = q.pop(0)
        if cur_idx == fanin_idx:
            return True
        for fanout_idx in ff_fanout_list[cur_idx]:
            if not vis[fanout_idx]:
                vis[fanout_idx] = True
                q.append(fanout_idx)
    return False

def get_ff_levels(x_data, ff_fanin_list, ff_fanout_list, gate_to_index):
    has_loop_map = {}
    ff_level = []
    for idx in range(len(x_data)):
        ff_level.append(-1)
    q = []
    for idx, x_data_info in enumerate(x_data):
        if (x_data_info[1] == gate_to_index['PI'] or x_data_info[1] == gate_to_index['DFF']):
            ff_level[idx] = 0
            q.append(idx)
            
    while len(q) > 0:
        cur_idx = q.pop(0)
        for fanout_idx in ff_fanout_list[cur_idx]:
            if ff_level[cur_idx] + 1 > ff_level[fanout_idx]:
                loop_pair = (cur_idx, fanout_idx)
                if loop_pair not in has_loop_map:
                    has_loop_map[loop_pair] = has_loop(x_data, ff_fanin_list, ff_fanout_list, fanout_idx, cur_idx)
                if has_loop_map[loop_pair]:
                    continue
                ff_level[fanout_idx] = ff_level[cur_idx] + 1
                q.append(fanout_idx)
    return ff_level

def get_ppi_cover_list(x_data, ff_fanin_list, ff_levels):
    ppi_cover_list = []
    for idx in range(len(x_data)):
        ppi_cover_list.append([])
    ff_level_list = []
    for idx in range(max(ff_levels) + 1):
        ff_level_list.append([])
    for idx, level in enumerate(ff_levels):
        if level != -1:
            ff_level_list[level].append(idx)
    
    for level, lst in enumerate(ff_level_list):
        for idx in lst:
            if level == 0:
                ppi_cover_list[idx].append(idx)
            else:
                for fanin_idx in ff_fanin_list[idx]:
                    ppi_cover_list[idx] += ppi_cover_list[fanin_idx]
                ppi_cover_list[idx] = list(set(ppi_cover_list[idx]))
    
    return ppi_cover_list

def save_bench(file, x_data, fanin_list, fanout_list, gate_to_idx={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}):
    PI_list = []
    PO_list = []
    for idx, ele in enumerate(fanin_list):
        if len(fanin_list[idx]) == 0:
            PI_list.append(idx)
    for idx, ele in enumerate(fanout_list):
        if len(fanout_list[idx]) == 0:
            PO_list.append(idx)
    
    f = open(file, 'w')
    f.write('# {:} inputs\n'.format(len(PI_list)))
    f.write('# {:} outputs\n'.format(len(PO_list)))
    f.write('\n')
    # Input
    for idx in PI_list:
        f.write('INPUT({})\n'.format(x_data[idx][0]))
    f.write('\n')
    # Output
    for idx in PO_list:
        f.write('OUTPUT({})\n'.format(x_data[idx][0]))
    f.write('\n')
    # Gates
    for idx, x_data_info in enumerate(x_data):
        if idx not in PI_list:
            gate_type = None
            for ele in gate_to_idx.keys():
                if gate_to_idx[ele] == x_data_info[1]:
                    gate_type = ele
                    break
            line = '{} = {}('.format(x_data_info[0], gate_type)
            for k, fanin_idx in enumerate(fanin_list[idx]):
                if k == len(fanin_list[idx]) - 1:
                    line += '{})\n'.format(x_data[fanin_idx][0])
                else:
                    line += '{}, '.format(x_data[fanin_idx][0])
            f.write(line)
    f.write('\n')
    f.close()
    
    return PI_list, PO_list

def feature_gen_pio(x_data, PI_indexs, PO_indexs):
    for idx in range(len(x_data)):
        x_data[idx].append(0)
        x_data[idx].append(0)
    for idx in PI_indexs:
        x_data[idx][-2] = 1
    for idx in PO_indexs:
        x_data[idx][-1] = 1
    return x_data

def get_ppi_cover_list(x_data, ff_fanin_list, ff_levels):
    ppi_cover_list = []
    for idx in range(len(x_data)):
        ppi_cover_list.append([])
    ff_level_list = []
    for idx in range(max(ff_levels) + 1):
        ff_level_list.append([])
    for idx, level in enumerate(ff_levels):
        if level != -1:
            ff_level_list[level].append(idx)
    
    for level, lst in enumerate(ff_level_list):
        for idx in lst:
            if level == 0:
                ppi_cover_list[idx].append(idx)
            else:
                for fanin_idx in ff_fanin_list[idx]:
                    ppi_cover_list[idx] += ppi_cover_list[fanin_idx]
                ppi_cover_list[idx] = list(set(ppi_cover_list[idx]))
    
    return ppi_cover_list
def get_sample_paths(g, no_path=1000, max_path_len=128):
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    PO_index = g['forward_index'][(g['forward_level'] != 0) & (g['backward_level'] == 0)]
    no_nodes = len(g['forward_index'])
    level_list = [[] for I in range(g['forward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].t():
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
        
    # Sample Paths
    path_list = []
    path_len_list = []
    path_hash = []
    no_and_list = []
    no_not_list = []
    for _ in range(no_path):
        # Sample path
        path = []
        node_idx = random.choice(PI_index).item()
        path.append(node_idx)
        while len(fanout_list[node_idx]) > 0 and len(path) < max_path_len:
            node_idx = random.choice(fanout_list[node_idx])
            path.append(node_idx)
        hash_val = hash_arr(path)
        if hash_val in path_hash:
            continue
        else:
            path_hash.append(hash_val)
        
        # AND / NOT 
        no_and = 0
        no_not = 0
        for node_idx in path:
            if g['gate'][node_idx] == 1:
                no_and += 1
            elif g['gate'][node_idx] == 2:
                no_not += 1 
        
        # Path length
        if len(path) < max_path_len:
            path_len_list.append(len(path))
        else:
            path_len_list.append(max_path_len)
        while len(path) < max_path_len:
            path.append(-1)
        path = path[:max_path_len]
              
        no_and_list.append(no_and) 
        no_not_list.append(no_not)
        path_list.append(path)
    
    return path_list, path_len_list, no_and_list, no_not_list

def get_fanin_fanout_cone(g, max_no_nodes=512): 
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    PO_index = g['forward_index'][(g['forward_level'] != 0) & (g['backward_level'] == 0)]
    no_nodes = len(g['forward_index'])
    forward_level_list = [[] for I in range(g['forward_level'].max()+1)]
    backward_level_list = [[] for I in range(g['backward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].T:
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    for k, idx in enumerate(g['forward_index']):
        forward_level_list[g['forward_level'][k].item()].append(k)
        backward_level_list[g['backward_level'][k].item()].append(k)
    
    # PI Cover 
    pi_cover = [[] for _ in range(no_nodes)]
    for level in range(len(forward_level_list)):
        for idx in forward_level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    
    # PO Cover
    po_cover = [[] for _ in range(no_nodes)]
    for level in range(len(backward_level_list)):
        for idx in backward_level_list[level]:
            if level == 0:
                po_cover[idx].append(idx)
            tmp_po_cover = []
            for post_k in fanout_list[idx]:
                tmp_po_cover += po_cover[post_k]
            tmp_po_cover = list(set(tmp_po_cover))
            po_cover[idx] += tmp_po_cover
    
    # fanin and fanout cone 
    fanin_fanout_cones = [[-1]*max_no_nodes for _ in range(max_no_nodes)]
    fanin_fanout_cones = torch.tensor(fanin_fanout_cones, dtype=torch.long)
    for i in range(no_nodes):
        for j in range(no_nodes):
            if i == j:
                fanin_fanout_cones[i][j] = 0
                continue
            if len(pi_cover[j]) <= len(pi_cover[i]) and g['forward_level'][j] < g['forward_level'][i]:
                j_in_i_fanin = True
                for pi in pi_cover[j]:
                    if pi not in pi_cover[i]:
                        j_in_i_fanin = False
                        break
                if j_in_i_fanin:
                    assert fanin_fanout_cones[i][j] == -1
                    fanin_fanout_cones[i][j] = 1
                else:
                    fanin_fanout_cones[i][j] = 0
            elif len(po_cover[j]) <= len(po_cover[i]) and g['forward_level'][j] > g['forward_level'][i]:
                j_in_i_fanout = True
                for po in po_cover[j]:
                    if po not in po_cover[i]:
                        j_in_i_fanout = False
                        break
                if j_in_i_fanout:
                    assert fanin_fanout_cones[i][j] == -1
                    fanin_fanout_cones[i][j] = 2
                else:
                    fanin_fanout_cones[i][j] = 0
            else:
                fanin_fanout_cones[i][j] = 0
    
    assert -1 not in fanin_fanout_cones[:no_nodes, :no_nodes]
    
    return fanin_fanout_cones

def prepare_dg2_labels_cpp(g, no_patterns=15000, 
                           simulator='./simulator/simulator', 
                           graph_filepath='', 
                           res_filepath=''):
    if graph_filepath == '':
        graph_filepath = './tmp/tmp_graph_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    if res_filepath == '':
        res_filepath = './tmp/tmp_res_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    no_pi = len(PI_index)
    no_nodes = len(g['forward_index'])
    level_list = [[] for I in range(g['forward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].t():
        fanin_list[edge[1].item()].append(edge[0].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
    
    # PI Cover
    pi_cover = [[] for _ in range(no_nodes)]
    for level in range(len(level_list)):
        for idx in level_list[level]:
            if level == 0:
                pi_cover[idx].append(idx)
            tmp_pi_cover = []
            for pre_k in fanin_list[idx]:
                tmp_pi_cover += pi_cover[pre_k]
            tmp_pi_cover = list(set(tmp_pi_cover))
            pi_cover[idx] += tmp_pi_cover
    pi_cover_hash_list = []
    for idx in range(no_nodes):
        pi_cover_hash_list.append(hash_arr(pi_cover[idx]))    
            
    # Write graph to file
    f = open(graph_filepath, 'w')
    f.write('{} {} {}\n'.format(no_nodes, len(g['edge_index'][0]), no_patterns))
    for idx in range(no_nodes):
        f.write('{} {}\n'.format(g['gate'][idx].item(), g['forward_level'][idx].item()))
    for edge in g['edge_index'].t():
        f.write('{} {}\n'.format(edge[0].item(), edge[1].item()))
    for idx in range(no_nodes):
        f.write('{}\n'.format(pi_cover_hash_list[idx]))
    f.close()
    
    # Simulation  
    sim_cmd = '{} {} {}'.format(simulator, graph_filepath, res_filepath)
    stdout, exec_time = run_command(sim_cmd)
    f = open(res_filepath, 'r')
    lines = f.readlines()
    f.close()
    prob = [-1] * no_nodes
    for line in lines[:no_nodes]:
        arr = line.replace('\n', '').split(' ')
        prob[int(arr[0])] = float(arr[1])
    tt_index = []
    tt_sim = []
    # TT pairs 
    no_tt_pairs = int(lines[no_nodes].replace('\n', '').split(' ')[1])
    for line in lines[no_nodes+1:no_nodes+1+no_tt_pairs]:
        arr = line.replace('\n', '').split(' ')
        assert len(arr) == 3
        tt_index.append([int(arr[0]), int(arr[1])])
        tt_sim.append(float(arr[2]))

    tt_index = torch.tensor(tt_index)
    tt_sim = torch.tensor(tt_sim)
    prob = torch.tensor(prob)
    
    # Connection pairs 
    no_connection_pairs = int(lines[no_nodes+1+no_tt_pairs].replace('\n', '').split(' ')[1])
    con_index = []
    con_label = []
    for line in lines[no_nodes+2+no_tt_pairs: no_nodes+2+no_tt_pairs+no_connection_pairs]:
        arr = line.replace('\n', '').split(' ')
        assert len(arr) == 3
        con_index.append([int(arr[0]), int(arr[1])])
        con_label.append(int(arr[2]))
    con_index = torch.tensor(con_index)
    con_label = torch.tensor(con_label)
    
    # Remove 
    os.remove(graph_filepath)
    os.remove(res_filepath)
    
    return prob, tt_index, tt_sim, con_index, con_label

def prepare_workload_prob(g, no_patterns=15000, 
                           simulator='./simulator/wl_simulator', 
                           graph_filepath='', 
                           res_filepath=''):
    if graph_filepath == '':
        graph_filepath = './tmp/tmp_graph_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    if res_filepath == '':
        res_filepath = './tmp/tmp_res_{}_{}_{}.txt'.format(
            time.strftime("%Y%m%d-%H%M%S"), threading.currentThread().ident, random.randint(0, 1000)
        )
    # Parse graph 
    PI_index = g['forward_index'][(g['forward_level'] == 0) & (g['backward_level'] != 0)]
    no_pi = len(PI_index)
    no_nodes = len(g['forward_index'])
    level_list = [[] for I in range(g['forward_level'].max()+1)]
    fanin_list = [[] for _ in range(no_nodes)]
    for edge in g['edge_index'].T:
        fanin_list[edge[1].item()].append(edge[0].item())
    for k, idx in enumerate(g['forward_index']):
        level_list[g['forward_level'][k].item()].append(k)
    
    # Write graph to file
    f = open(graph_filepath, 'w')
    f.write('{} {} {} {}\n'.format(no_nodes, len(g['edge_index'][0]), no_pi, no_patterns))
    for idx in range(no_nodes):
        f.write('{} {}\n'.format(g['gate'][idx].item(), g['forward_level'][idx].item()))
    for edge in g['edge_index'].T:
        f.write('{} {}\n'.format(edge[0].item(), edge[1].item()))
    for pi in PI_index:
        f.write('{} {}\n'.format(pi.item(), random.random()))
    f.close()
    
    # Simulation  
    sim_cmd = '{} {} {}'.format(simulator, graph_filepath, res_filepath)
    stdout, exec_time = run_command(sim_cmd)
    f = open(res_filepath, 'r')
    lines = f.readlines()
    f.close()
    prob = [-1] * no_nodes
    for line in lines[:no_nodes]:
        arr = line.replace('\n', '').split(' ')
        prob[int(arr[0])] = float(arr[1])
    
    # Remove 
    os.remove(graph_filepath)
    os.remove(res_filepath)
    
    return prob

def get_connection_pairs(x_data, edge_index, forward_level, no_src=512, no_dst=512, cone=None):
    no_nodes = len(x_data)
    fanin_list = [[] for _ in range(no_nodes)]
    fanout_list = [[] for _ in range(no_nodes)]
    for edge in edge_index.T:
        fanin_list[edge[1].item()].append(edge[0].item())
        fanout_list[edge[0].item()].append(edge[1].item())
    
    # Sample src 
    connect_pair_index = []
    connect_label = []
    src_list = random.sample(range(no_nodes), no_src)
    for src in src_list:
        # Find all connection 
        fanin_cone = []
        fanout_cone = []
        if cone is not None:
            for idx in range(no_nodes):
                if cone[src][idx] == 1:
                    fanin_cone.append(idx)
                elif cone[src][idx] == 2:
                    fanout_cone.append(idx)
        else:
            # Search Fanin cone 
            q = [src]
            while len(q) > 0:
                node_idx = q.pop(0)
                fanin_cone.append(node_idx)
                for pre_k in fanin_list[node_idx]:
                    if pre_k not in fanin_cone:
                        q.append(pre_k)
            # Search Fanout cone 
            q = [src]
            while len(q) > 0:
                node_idx = q.pop(0)
                fanout_cone.append(node_idx)
                for post_k in fanout_list[node_idx]:
                    if post_k not in fanout_cone:
                        q.append(post_k)
        # Sample dst
        dst_list = random.sample(range(no_nodes), no_dst)
        for dst in dst_list:
            if dst == src:
                continue
            pair = [src, dst]
            if pair in connect_pair_index:
                continue
            connect_pair_index.append(pair)
            if forward_level[dst] < forward_level[src] and dst in fanin_cone:     # in fanin cone
                connect_label.append(1)
            elif forward_level[dst] > forward_level[src] and dst in fanout_cone:    # in fanout cone
                connect_label.append(2)
            else:
                connect_label.append(0)
    
    connect_pair_index = torch.tensor(connect_pair_index, dtype=torch.long)
    connect_label = torch.tensor(connect_label, dtype=torch.long)
    
    return connect_pair_index, connect_label

def get_hop_pair_labels(hop_nodes_list, hop_tt, edge_index, no_pairs):
    no_hops = len(hop_nodes_list)
    hop_pair_index = np.random.randint(0, no_hops, [no_pairs, 2])
    hop_ged = []
    hop_node_pair = []
    hop_tt_sim = []
    for pair in hop_pair_index:
        if pair[0] == pair[1]:
            ged = 0
        else:
            g1 = nx.DiGraph()
            g2 = nx.DiGraph()
            for edge in edge_index.T:
                if edge[0] in hop_nodes_list[pair[0]] and edge[1] in hop_nodes_list[pair[0]]:
                    g1.add_edge(edge[0].item(), edge[1].item())
                if edge[0] in hop_nodes_list[pair[1]] and edge[1] in hop_nodes_list[pair[1]]:
                    g2.add_edge(edge[0].item(), edge[1].item())
            ged = nx.graph_edit_distance(g1, g2, timeout=0.1)
            ged = ged / max(len(hop_nodes_list[pair[0]]), len(hop_nodes_list[pair[1]]))
            ged = min(ged, 1.0)
        
        hop_node_pair.append([pair[0], pair[1]])
        hop_ged.append(ged)
        tt_sim = (hop_tt[pair[0]] == hop_tt[pair[1]]).sum() * 1.0 / len(hop_tt[pair[0]])
        hop_tt_sim.append(tt_sim.item())
        
    hop_node_pair = torch.tensor(hop_node_pair, dtype=torch.long)
    hop_ged = torch.tensor(hop_ged, dtype=torch.float)
    return hop_node_pair, hop_ged, hop_tt_sim

def complete_simulation(g_pis, g_pos, g_forward_level, g_nodes, g_edges, g_gates, pi_stats=[]):
    no_pi = len(g_pis) - sum([1 for x in pi_stats if x != 2])
    level_list = []
    fanin_list = []
    index_m = {}
    for level in range(g_forward_level.max()+1):
        level_list.append([])
    for k, idx in enumerate(g_nodes):
        level_list[g_forward_level[k].item()].append(k)
        fanin_list.append([])
        index_m[idx.item()] = k
    for edge in g_edges.t():
        fanin_list[index_m[edge[1].item()]].append(index_m[edge[0].item()])
    
    states = [-1] * len(g_nodes)
    po_tt = []
    for pattern_idx in range(2**no_pi):
        pattern = [int(x) for x in list(bin(pattern_idx)[2:].zfill(no_pi))]
        k = 0 
        while k < len(pi_stats):
            pi_idx = g_pis[k].item()
            if pi_stats[k] == 2:
                states[index_m[pi_idx]] = pattern[k]
            elif pi_stats[k] == 1:
                states[index_m[pi_idx]] = 1
            elif pi_stats[k] == 0:
                states[index_m[pi_idx]] = 0
            else:
                raise ValueError('Invalid pi_stats')
            k += 1
        for level in range(1, len(level_list), 1):
            for node_k in level_list[level]:
                source_signals = []
                for pre_k in fanin_list[node_k]:
                    source_signals.append(states[pre_k])
                if len(source_signals) == 0:
                    continue
                states[node_k] = logic(g_gates[node_k].item(), source_signals)
        po_tt.append(states[index_m[g_pos.item()]])
    
    return po_tt, no_pi

def get_hops(idx, edge_index, x_data, gate, k_hop=4):
    last_target_idx = [idx]
    curr_target_idx = []
    hop_nodes = []
    hop_edges = torch.zeros((2, 0), dtype=torch.long)
    for k in range(k_hop):
        if len(last_target_idx) == 0:
            break
        for n in last_target_idx:
            ne_mask = edge_index[1] == n
            curr_target_idx += edge_index[0, ne_mask].tolist()
            hop_edges = torch.cat([hop_edges, edge_index[:, ne_mask]], dim=-1)
            hop_nodes += edge_index[0, ne_mask].unique().tolist()
        last_target_idx = list(set(curr_target_idx))
        curr_target_idx = []
    
    if len(hop_nodes) < 2:
        return [], [], [], []
    
    # Parse hop 
    hop_nodes = torch.tensor(hop_nodes).unique().long()
    hop_nodes = torch.cat([hop_nodes, torch.tensor([idx])])
    no_hops = k + 1
    hop_forward_level, hop_forward_index, hop_backward_level, _ = dg.return_order_info(hop_edges, len(x_data))
    hop_forward_level = hop_forward_level[hop_nodes]
    hop_backward_level = hop_backward_level[hop_nodes]
    
    hop_gates = gate[hop_nodes]
    hop_pis = hop_nodes[(hop_forward_level==0) & (hop_backward_level!=0)]
    hop_pos = hop_nodes[(hop_forward_level!=0) & (hop_backward_level==0)]
    
    return hop_nodes, hop_gates, hop_pis, hop_pos
 