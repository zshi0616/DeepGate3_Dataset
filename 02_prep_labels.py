import os 
import glob
import deepgate as dg
from torch_geometric.data import Data, InMemoryDataset
import torch
import numpy as np 
import random
import copy
import time
import argparse
import torch.nn.functional as F

import utils.aiger_utils as aiger_utils
import utils.circuit_utils as circuit_utils

gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}
NODE_CONNECT_SAMPLE_RATIO = 0.1
NO_NODE_PATH = 10
NO_NODE_HOP = 10
K_HOP = 4

NO_NODES = [30, 5000]

def get_parse_args():
    parser = argparse.ArgumentParser()

    # Range
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=100000, type=int)
    
    # Input
    parser.add_argument('--aig_dir', default='./dg_aig', type=str)
    
    # Output
    parser.add_argument('--npz_path', default='./npz/graphs.npz', type=str)
    
    args = parser.parse_args()
    
    return args

class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        # if 'hop_forward_index' in key:
        #     return value.shape[0]
        # elif 'path_forward_index' in key:
        #     return value.shape[0]
        if key == 'ninp_node_index' or key == 'ninh_node_index':
            return self.num_nodes
        elif key == 'ninp_path_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'ninh_hop_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_nodes': 
            return self.num_nodes
        elif key == 'winhop_po' or key == 'winhop_nodes':
            return self.num_nodes
        elif key == 'hop_pair_index' or key == 'hop_forward_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'path_forward_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'paths' or key == 'hop_nodes':
            return self.num_nodes
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        elif key == "connect_pair_index" or key == 'hop_pair_index':
            return 1
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_pi_stats' or key == 'hop_tt' or key == 'no_hops':
            return 0
        elif key == 'winhop_po' or key == 'winhop_nodes' or key == 'winhop_nodes_stats' or key == 'winhop_lev':
            return 0
        elif key == 'hop_nodes' or key == 'hop_nodes_stats':
            return 0
        elif key == 'paths':
            return 0
        else:
            return 0

def get_winhop(g, k_hop=8):
    graph = {}
    max_level = g['forward_level'].max()
    forward_level = g['forward_level'].numpy()
    backward_level = g['backward_level'].numpy()
    level_list = [[] for _ in range(max_level+1)]
    for idx in range(len(x_data)):
        level_list[forward_level[idx]].append(idx)
    po_list = forward_index[backward_level == 0]
    
    all_hop_po = torch.zeros((0, 1), dtype=torch.long)
    all_hop_winlev = torch.zeros((0, 1), dtype=torch.long)
    max_hop_nodes_cnt = 0
    for k in range(k_hop+1):
        max_hop_nodes_cnt += 2**k
    all_hop_nodes = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
    all_hop_nodes_stats = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
    
    edge_index = g['edge_index']
    gate = g['gate']
    # edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
    # gate = torch.tensor(g['gate'], dtype=torch.long)
    has_hop = [0] * len(x_data)
    hop_level = k_hop
    hop_winlev = 0
    while hop_level < max_level:
        for idx in level_list[hop_level]:
            hop_nodes, hop_gates, hop_pis, hop_pos = circuit_utils.get_hops(idx, edge_index, x_data, gate, k_hop=k_hop)
            if len(hop_gates) < 2:
                continue
            has_hop[idx] = 1
            
            # Record hop
            all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
            assert len(hop_nodes) <= max_hop_nodes_cnt
            hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
            hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
            hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
            all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
            all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
            all_hop_winlev = torch.cat([all_hop_winlev, torch.tensor([hop_winlev]).view(1, -1)], dim=0)
            
        hop_level += (k_hop - 2)
        hop_winlev += 1
    
    # Add PO 
    for po_idx in po_list:
        if has_hop[po_idx] == 0:
            hop_nodes, hop_gates, hop_pis, hop_pos = circuit_utils.get_hops(po_idx, edge_index, x_data, gate, k_hop=k_hop)
            if len(hop_gates) < 2:
                continue
            has_hop[po_idx] = 1
            
            # Record hop
            all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
            assert len(hop_nodes) <= max_hop_nodes_cnt
            hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
            hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
            hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
            all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
            all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
            all_hop_winlev = torch.cat([all_hop_winlev, torch.tensor([hop_winlev]).view(1, -1)], dim=0)
    graph = {
        'winhop_po': all_hop_po.numpy(),
        'winhop_nodes': all_hop_nodes.numpy(),
        'winhop_nodes_stats': all_hop_nodes_stats.numpy(), 
        'winhop_lev': all_hop_winlev.numpy()
    }
    g.update(graph)
    return g

if __name__ == '__main__':
    args = get_parse_args()
    
    aig_namelist_path = os.path.join(args.aig_dir, 'aig_namelist.txt')
    if not os.path.exists(aig_namelist_path):
        aig_files = glob.glob('{}/*.aig'.format(args.aig_dir))
        aig_namelist = []
        for aig_file in aig_files:
            aig_name = os.path.basename(aig_file).replace('.aig', '')
            aig_namelist.append(aig_name)
        with open(aig_namelist_path, 'w') as f:
            for aig_name in aig_namelist:
                f.write(aig_name + '\n')
    else:
        with open(aig_namelist_path, 'r') as f:
            aig_namelist = f.readlines()
            aig_namelist = [x.strip() for x in aig_namelist]
    
    aig_namelist = aig_namelist[args.start: args.end]
    no_circuits = len(aig_namelist)
    tot_time = 0
    graphs = {}
    for aig_idx, cir_name in enumerate(aig_namelist):
        aig_file = os.path.join(args.aig_dir, cir_name + '.aig')
        # if cir_name != '9848':
        #     continue
        
        start_time = time.time()
        tmp_aag_filename = os.path.join('./tmp', cir_name + '.aag')
        x_data, edge_index = aiger_utils.seqaig_to_xdata(aig_file, tmp_aag_filename)
        print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(
            cir_name, aig_idx, no_circuits, len(x_data), 
            tot_time, tot_time / ((aig_idx + 1) / no_circuits) - tot_time, 
            len(graphs)
        ))
        fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
        
        # Replace DFF as PPI and PPO
        no_ff = 0
        for idx in range(len(x_data)):
            if x_data[idx][1] == gate_to_index['DFF']:
                no_ff += 1
                x_data[idx][1] = gate_to_index['PI']
                for fanin_idx in fanin_list[idx]:
                    fanout_list[fanin_idx].remove(idx)
                fanin_list[idx] = []
        # circuit_utils.save_bench('./tmp/test.bench', x_data, fanin_list, fanout_list)
        
        # Get x_data and edge_index
        edge_index = []
        for idx in range(len(x_data)):
            for fanin_idx in fanin_list[idx]:
                edge_index.append([fanin_idx, idx])
        x_data, edge_index = circuit_utils.remove_unconnected(x_data, edge_index)
        if len(edge_index) == 0 or len(x_data) < NO_NODES[0] or len(x_data) > NO_NODES[1]:
            continue
        x_one_hot = dg.construct_node_feature(x_data, 3)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, x_one_hot.size(0))
        
        graph = OrderedData()
        graph.x = x_one_hot
        graph.edge_index = edge_index
        graph.name = cir_name
        graph.gate = torch.tensor(x_data[:, 1], dtype=torch.long).unsqueeze(1)
        graph.forward_index = forward_index
        graph.backward_index = backward_index
        graph.forward_level = forward_level
        graph.backward_level = backward_level
                
        ################################################
        # DeepGate2 (node-level) labels
        ################################################
        prob, tt_pair_index, tt_sim, con_index, con_label = circuit_utils.prepare_dg2_labels_cpp(graph, 15000)
        graph.connect_pair_index = con_index.T
        graph.connect_label = con_label
        
        assert max(prob).item() <= 1.0 and min(prob).item() >= 0.0
        if len(tt_pair_index) == 0:
            tt_pair_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            tt_pair_index = tt_pair_index.t().contiguous()
        graph.prob = prob
        graph.tt_pair_index = tt_pair_index
        graph.tt_sim = tt_sim
                
        # connect_pair_index, connect_label = circuit_utils.get_connection_pairs(
        #     x_data, edge_index, forward_level, 
        #     no_src=int(len(x_data)*NODE_CONNECT_SAMPLE_RATIO), no_dst=int(len(x_data)*NODE_CONNECT_SAMPLE_RATIO),
        #     cone=None
        # )
        # graph.connect_pair_index = connect_pair_index.T
        # graph.connect_label = connect_label
        
        ################################################
        # Path-level labels    
        ################################################             
        sample_paths, sample_paths_len, sample_paths_no_and, sample_paths_no_not = circuit_utils.get_sample_paths(graph, no_path=64, max_path_len=256)
        graph.path_forward_index = torch.tensor(range(len(sample_paths)), dtype=torch.long)
        graph.paths = torch.tensor(sample_paths, dtype=torch.long)
        graph.paths_len = torch.tensor(sample_paths_len, dtype=torch.long)
        graph.paths_and_ratio = torch.tensor(sample_paths_no_and, dtype=torch.long) / torch.tensor(sample_paths_len, dtype=torch.float)
        graph.paths_no_and = torch.tensor(sample_paths_no_and, dtype=torch.long)
        graph.paths_no_not = torch.tensor(sample_paths_no_not, dtype=torch.long)
        # Sample node in path 
        node_path_pair_index = []
        node_path_labels = []
        for path_idx, sample_path in enumerate(sample_paths):
            path = sample_path[:sample_paths_len[path_idx]]
            node_in_path = np.random.choice(path, NO_NODE_PATH)
            node_in_path = [[x, path_idx] for x in node_in_path]
            node_out_path = [x for x in range(len(x_data)) if x not in path]
            node_out_path = np.random.choice(node_out_path, NO_NODE_PATH)
            node_out_path = [[x, path_idx] for x in node_out_path]
            node_path_pair_index += node_in_path + node_out_path
            node_path_labels += [1] * NO_NODE_PATH + [0] * NO_NODE_PATH
        node_path_pair_index = torch.tensor(node_path_pair_index, dtype=torch.long)
        ninp_node_index = node_path_pair_index[:, 0]
        ninp_path_index = node_path_pair_index[:, 1]
        graph.ninp_node_index = ninp_node_index
        graph.ninp_path_index = ninp_path_index
        node_path_labels = torch.tensor(node_path_labels, dtype=torch.long)
        graph.ninp_labels = node_path_labels
        
        ################################################
        # Hop-level labels    
        ################################################  
        # Random select hops 
        rand_idx_list = list(range(len(x_data)))
        random.shuffle(rand_idx_list)
        rand_idx_list = rand_idx_list[0: int(len(x_data) * 0.15)]
        all_hop_pi = torch.zeros((0, 2**(K_HOP-1)), dtype=torch.long)
        all_hop_pi_stats = torch.zeros((0, 2**(K_HOP-1)), dtype=torch.long)
        all_hop_po = torch.zeros((0, 1), dtype=torch.long)
        max_hop_nodes_cnt = 0
        for k in range(K_HOP+1):
            max_hop_nodes_cnt += 2**k
        all_hop_nodes = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
        all_hop_nodes_stats = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
        all_tt = []
        all_hop_nodes_cnt = []
        all_hop_level_cnt = []
        for idx in rand_idx_list:
            last_target_idx = copy.deepcopy([idx])
            curr_target_idx = []
            hop_nodes = []
            hop_edges = torch.zeros((2, 0), dtype=torch.long)
            for k_hops in range(K_HOP):
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
                continue
            hop_nodes = torch.tensor(hop_nodes).unique().long()
            hop_nodes = torch.cat([hop_nodes, torch.tensor([idx])])
            no_hops = k_hops + 1
            hop_forward_level, hop_forward_index, hop_backward_level, _ = dg.return_order_info(hop_edges, len(x_data))
            hop_forward_level = hop_forward_level[hop_nodes]
            hop_backward_level = hop_backward_level[hop_nodes]
            
            hop_gates = graph.gate[hop_nodes]
            hop_pis = hop_nodes[(hop_forward_level==0) & (hop_backward_level!=0)]
            hop_pos = hop_nodes[(hop_forward_level!=0) & (hop_backward_level==0)]
            if len(hop_pis) > 2**(K_HOP-1):
                continue
            
            hop_pi_stats = [2] * len(hop_pis)  # -1 Padding, 0 Logic-0, 1 Logic-1, 2 variable
            for assigned_pi_k in range(6, len(hop_pi_stats), 1):
                hop_pi_stats[assigned_pi_k] = random.randint(0, 1)
            hop_tt, _ = circuit_utils.complete_simulation(hop_pis, hop_pos, hop_forward_level, hop_nodes, hop_edges, hop_gates, pi_stats=hop_pi_stats)
            while len(hop_tt) < 2**6:
                hop_tt += hop_tt
                hop_pis = torch.cat([torch.tensor([-1]), hop_pis])
                hop_pi_stats.insert(0, -1)
            while len(hop_pi_stats) < 2**(K_HOP-1):
                hop_pis = torch.cat([torch.tensor([-1]), hop_pis])
                hop_pi_stats.insert(0, -1)
            
            # Record the hop 
            all_hop_pi = torch.cat([all_hop_pi, hop_pis.view(1, -1)], dim=0)
            all_hop_po = torch.cat([all_hop_po, hop_pos.view(1, -1)], dim=0)
            all_hop_pi_stats = torch.cat([all_hop_pi_stats, torch.tensor(hop_pi_stats).view(1, -1)], dim=0)
            assert len(hop_nodes) <= max_hop_nodes_cnt
            all_hop_nodes_cnt.append(len(hop_nodes))
            all_hop_level_cnt.append(no_hops)
            hop_nodes_stats = torch.ones(len(hop_nodes), dtype=torch.long)
            hop_nodes = F.pad(hop_nodes, (0, max_hop_nodes_cnt - len(hop_nodes)), value=-1)
            hop_nodes_stats = F.pad(hop_nodes_stats, (0, max_hop_nodes_cnt - len(hop_nodes_stats)), value=0)
            all_hop_nodes = torch.cat([all_hop_nodes, hop_nodes.view(1, -1)], dim=0)
            all_hop_nodes_stats = torch.cat([all_hop_nodes_stats, hop_nodes_stats.view(1, -1)], dim=0)
            all_tt.append(hop_tt)

        graph.hop_pi = all_hop_pi
        graph.hop_po = all_hop_po
        graph.hop_pi_stats = all_hop_pi_stats
        graph.hop_nodes = all_hop_nodes
        graph.hop_nodes_stats = all_hop_nodes_stats
        graph.hop_tt = torch.tensor(all_tt, dtype=torch.long)
        graph.hop_nds = torch.tensor(all_hop_nodes_cnt, dtype=torch.long)
        graph.hop_levs = torch.tensor(all_hop_level_cnt, dtype=torch.long)
        graph.hop_forward_index = torch.tensor(range(len(all_hop_nodes)), dtype=torch.long)
        
        hop_pair_index, hop_pair_ged, hop_pair_tt_sim = circuit_utils.get_hop_pair_labels(
            all_hop_nodes, graph.hop_tt, edge_index, 
            no_pairs=min(int(len(all_hop_nodes) * len(all_hop_nodes) * 0.1), 100)
        )
        no_pairs = len(hop_pair_index)
        if no_pairs == 0:
            continue
        graph.hop_pair_index = hop_pair_index.T.reshape(2, no_pairs)
        graph.hop_ged = hop_pair_ged
        graph.hop_tt_sim = torch.tensor(hop_pair_tt_sim, dtype=torch.float)
        
        # Sample node in hop 
        node_hop_pair_index = []
        node_hop_labels = []
        for hop_idx, sample_hop in enumerate(all_hop_nodes):
            hop = sample_hop[sample_hop != -1].tolist()
            node_in_hop = np.random.choice(hop, NO_NODE_HOP)
            node_in_hop = [[x, hop_idx] for x in node_in_hop]
            node_out_hop = [x for x in range(len(x_data)) if x not in hop]
            node_out_hop = np.random.choice(node_out_hop, NO_NODE_HOP)
            node_out_hop = [[x, hop_idx] for x in node_out_hop]
            node_hop_pair_index += node_in_hop + node_out_hop
            node_hop_labels += [1] * NO_NODE_HOP + [0] * NO_NODE_HOP
        node_hop_pair_index = torch.tensor(node_hop_pair_index, dtype=torch.long)
        node_hop_labels = torch.tensor(node_hop_labels, dtype=torch.long)
        ninh_node_index = node_hop_pair_index[:, 0]
        ninh_hop_index = node_hop_pair_index[:, 1]
        graph.ninh_node_index = ninh_node_index
        graph.ninh_hop_index = ninh_hop_index
        graph.ninh_labels = node_hop_labels
        
        # Win hop
        graph = get_winhop(graph, k_hop=8)
        
        # Statistics
        graph.no_nodes = len(x_data)
        graph.no_edges = len(edge_index[0])
        graph.no_hops = len(all_hop_nodes)
        graph.no_paths = len(sample_paths)
        end_time = time.time()
        tot_time += end_time - start_time
        
        # Save graph 
        g = {}
        for key in graph.keys():
            if key == 'name' or key == 'batch' or key == 'ptr':
                continue
            if torch.is_tensor(graph[key]):
                g[key] = graph[key].cpu().numpy()
            else:
                g[key] = graph[key]
        graphs[cir_name] = copy.deepcopy(g)

    np.savez_compressed(args.npz_path, circuits=graphs)
    print(args.npz_path)
    print(len(graphs))