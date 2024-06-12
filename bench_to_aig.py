
import glob
import os 
from utils.utils import run_command

bench_dir = './QoR_Design'
aig_dir = './QoR_Design'

if __name__ == '__main__':
    if not os.path.exists(aig_dir):
        os.makedirs(aig_dir)
    
    for bench_path in glob.glob(os.path.join(bench_dir, '*.bench')):
        cir_name = os.path.basename(bench_path).split('.')[0]
        aig_path = os.path.join(aig_dir, '{}.aig'.format(cir_name))
        abc_cmd = 'abc -c "read_bench {}; strash; write_aiger {}"'.format(bench_path, aig_path)
        stdout, _ = run_command(abc_cmd)
        print('Save: {} to {}'.format(cir_name, aig_path))
    
    