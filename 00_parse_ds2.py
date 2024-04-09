import os 
import numpy as np 
from utils.utils import run_command

graph_path = 'raw_data/ds2/graphs.npz'
aig_dir = './ds_aig'

def read_npz_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data

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

if __name__ == '__main__':
    if not os.path.exists(aig_dir):
        os.makedirs(aig_dir)
        
    circuits = read_npz_file(graph_path)['circuits'].item()
    for cir_idx, cir_name in enumerate(circuits):
        x_data = circuits[cir_name]['x']
        edge_index = circuits[cir_name]['edge_index']
        fanin_list, fanout_list = get_fanin_fanout(x_data, edge_index)
        x_data = x_data.astype(np.int32)
        
        tmp_bench_path = './tmp/{}.bench'.format(cir_name)
        fanin_list, fanout_list = save_bench(tmp_bench_path, x_data, fanin_list, fanout_list)
        
        # Save aig
        aig_path = os.path.join(aig_dir, '{}.aig'.format(cir_name))
        abc_cmd = 'abc -c "read_bench {}; strash; write_aiger {}"'.format(tmp_bench_path, aig_path)
        stdout, _ = run_command(abc_cmd)
        os.remove(tmp_bench_path)
        
        print('[INFO] Parse circuit {}'.format(cir_name))