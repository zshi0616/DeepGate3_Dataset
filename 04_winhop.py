from deepgate.utils.data_utils import read_npz_file
import torch.nn.functional as F
import numpy as np
import torch 
import deepgate as dg

circuit_path = './graphs.npz'
output_path = './graphs_winhop.npz'

k_hop = 8

def get_hops(idx, edge_index, x_data, gate):
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
    

if __name__ == '__main__':
    circuits = read_npz_file(circuit_path)['circuits'].item()
    graphs = {}
    
    for cir_idx, cir_name in enumerate(circuits):
        x_data = circuits[cir_name]['x']
        edge_index = circuits[cir_name]['edge_index']
        forward_level = circuits[cir_name]['forward_level']
        backward_level = circuits[cir_name]['backward_level']
        forward_index = circuits[cir_name]['forward_index']
        max_level = forward_level.max()
        gate = circuits[cir_name]['gate']
        has_hop = [0] * len(x_data)
        graph = {}
        print(cir_name, max_level)
        
        # Get level list 
        level_list = [[] for _ in range(max_level+1)]
        for idx in range(len(x_data)):
            level_list[forward_level[idx]].append(idx)
        po_list = forward_index[backward_level == 0]
        
        # Get fanin list 
        fanin_list = [[] for _ in range(len(x_data))]
        for edge_idx in range(edge_index.shape[1]):
            fanin_list[edge_index[1, edge_idx]].append(edge_index[0, edge_idx])
        
        # Select hops
        all_hop_pi = torch.zeros((0, 2**(k_hop-1)), dtype=torch.long)
        all_hop_po = torch.zeros((0, 1), dtype=torch.long)
        all_hop_winlev = torch.zeros((0, 1), dtype=torch.long)
        max_hop_nodes_cnt = 0
        for k in range(k_hop+1):
            max_hop_nodes_cnt += 2**k
        all_hop_nodes = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
        all_hop_nodes_stats = torch.zeros((0, max_hop_nodes_cnt), dtype=torch.long)
                
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        gate = torch.tensor(gate, dtype=torch.long)
        hop_level = k_hop
        hop_winlev = 0
        if hop_level > max_level:
            continue
        while hop_level < max_level:
            for idx in level_list[hop_level]:
                hop_nodes, hop_gates, hop_pis, hop_pos = get_hops(idx, edge_index, x_data, gate)
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
                hop_nodes, hop_gates, hop_pis, hop_pos = get_hops(idx, edge_index, x_data, gate)
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
            
        graph = {
            'winhop_po': all_hop_po.numpy(),
            'winhop_nodes': all_hop_nodes.numpy(),
            'winhop_nodes_stats': all_hop_nodes_stats.numpy(), 
            'winhop_lev': all_hop_winlev.numpy()
        }
        graph.update(circuits[cir_name])
        
        graphs[cir_name] = graph
          
    np.savez_compressed(output_path, circuits=graphs)
    print(len(graphs))
        

    