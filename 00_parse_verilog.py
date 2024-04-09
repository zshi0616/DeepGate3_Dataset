import glob
import os
import sys
import platform
import numpy as np
import subprocess
import shutil
import argparse

from datasets import load_dataset
from utils.utils import run_command

CSV_PATH = './Verilog_bigquery_GitHub.csv'

def get_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=100000, type=int)
    args = parser.parse_args()
    
    return args

def process_design_aig(verilog_path, tmp_blif_path, aig_path):
    yosys_cmd = 'yosys -p "read_verilog {}; prep -auto-top; flatten; techmap; write_blif {}"'.format(
        verilog_path, tmp_blif_path
    )
    stdout, yosys_time = run_command(yosys_cmd, 3)
    if not os.path.exists(tmp_blif_path):
        return
    
    # Map to AIG
    abc_cmd = 'abc -c "read_blif {}; strash; write_aiger {}"'.format(tmp_blif_path, aig_path)
    stdout, abc_time = run_command(abc_cmd, 3)
    print('Time: Yosys {:.2f}s, ABC {:.2f}s'.format(yosys_time, abc_time))
    
    os.remove(tmp_blif_path)

if __name__ == '__main__':
    args = get_parse_args()
    ds = load_dataset('csv', data_files=CSV_PATH)
    print(next(iter(ds)))
    
    for cir_idx in range(len(ds['train']['text'])):
        if cir_idx < args.start or cir_idx >= args.end:
            continue
        cir = ds['train']['text'][cir_idx]
        cir_path = f'./tmp/{cir_idx}.v'
        blif_path = cir_path.replace('.v', '.blif')
        aig_path = './aig/{}.aig'.format(cir_idx)
        print('[INFO] Parse circuit {}'.format(cir_idx))
        
        # Preprocess 
        if 'module' not in cir or 'endmodule' not in cir:
            continue
        # if 'posedge' in cir or 'negedge' in cir:
        #     continue
        
        # Write verilog 
        with open(cir_path, 'w') as f:
            f.write(cir)
    
        process_design_aig(cir_path, blif_path, aig_path)
        os.remove(cir_path)
        
        if os.path.exists(aig_path):
            print(f'[{cir_idx}] {cir_path} -> {aig_path}')
        print()

