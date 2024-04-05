import numpy as np 
import os 
import copy 
from deepgate.utils.data_utils import read_npz_file

npz_dir = './output'
npz_list = [
    '01', '02', '03'
]
output_dir = './graphs.npz'

if __name__ == '__main__':
    graphs = {}
    for npz_name in npz_list:
        npz_path = os.path.join(npz_dir, f'{npz_name}.npz')
        if not os.path.exists(npz_path):
            print('Not Found:', npz_path)
            continue
        circuits = read_npz_file(npz_path)['circuits'].item()
        for cir_idx, cir_name in enumerate(circuits):
            g = {}
            for key in circuits[cir_name]:
                g[key] = circuits[cir_name][key]
            graphs[cir_name] = copy.deepcopy(g)
            print('Read Circuit:', cir_name)
    
    np.savez_compressed(output_dir, circuits=graphs)
    print(output_dir)
    print(len(graphs))
            