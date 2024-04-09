import os 
import glob
import time 
import random
import shutil
from utils.utils import run_command

NO_TRANS = 2
NO_SYN = 5
AIG_DIR = 'v_aig'

def parse_aig_head(aig_path):
    f = open(aig_path, 'r', encoding='ISO-8859-1')
    lines = f.readlines()
    f.close()
    for line in lines:
        header = line.replace('\n', '').split(' ')
        if header[0] == 'aig' or header[0] == 'aag':
            if len(header) == 7:
                # “M”, “I”, “L”, “O”, “A” separated by spaces.
                n_variables = eval(header[1])
                n_inputs = eval(header[2])
                n_latch = eval(header[3])
                unknown = eval(header[4])
                n_and = eval(header[5])
                n_outputs = eval(header[6])
            elif len(header) == 6:
                n_variables = eval(header[1])
                n_inputs = eval(header[2])
                n_outputs = eval(header[4])
                n_and = eval(header[5])
                n_latch = eval(header[3])
            else:
                n_variables, n_inputs, n_latch, n_outputs, n_and = 0, 0, 0, 0, 0
            return n_variables, n_inputs, n_latch, n_outputs, n_and
    return 0, 0, 0, 0, 0
            
def get_syn_command(no_syn):
    res = ''
    for _ in range(no_syn):
        action = random.randint(0, 6)
        if action == 0:
            res += 'balance'
            if random.randint(0, 1) == 0:
                res += ' -l'
            if random.randint(0, 1) == 0:
                res += ' -d'
            if random.randint(0, 1) == 0:
                res += ' -s'
            if random.randint(0, 1) == 0:
                res += ' -x'
            res += ';'
        elif action == 1:
            res += 'rewrite'
            if random.randint(0, 1) == 0:
                res += ' -l'
            if random.randint(0, 1) == 0:
                res += ' -z'
            res += ';'
        elif action == 2:
            res += 'resub'
            if random.randint(0, 1) == 0:
                res += ' -l'
            if random.randint(0, 1) == 0:
                res += ' -z'
            attr_K = random.randint(4, 16)
            res += ' -K {}'.format(attr_K)
            attr_N = random.randint(0, 3)
            res += ' -N {}'.format(attr_N)
            res += ';'
        elif action == 3:
            res += 'refactor'
            if random.randint(0, 1) == 0:
                res += ' -l'
            if random.randint(0, 1) == 0:
                res += ' -z'
            attr_N = random.randint(4, 15)
            res += ' -N {}'.format(attr_N)
            res += ';'
        elif action == 4:
            res += 'retime; strash; '
        elif action == 5:
            res += 'lcorr; '
        elif action == 6:
            res += 'scleanup; '
    
    return res


if __name__ == '__main__':
    aig_files = glob.glob('./{}/*.aig'.format(AIG_DIR))
    aig_namelist = []
    for aig_file in aig_files:
        aig_namelist.append(aig_file)
    no_circuits = len(aig_namelist)
    tot_time = 0
    no_trans = 0
        
    for aig_idx, aig_file in enumerate(aig_namelist):
        start_time = time.time()
        aig_name = os.path.basename(aig_file).replace('.aig', '')
        
        if 'trans' not in aig_name:
            continue
        
        print('Parse: {} ({:} / {:}), Time: {:.2f}s, ETA: {:.2f}s, Tot: {:}'.format(
            aig_name, aig_idx, no_circuits, 
            tot_time, tot_time / ((aig_idx + 1) / no_circuits) - tot_time, 
            no_trans
        ))
        origin_var, origin_in, origin_latch, origin_out, origin_and = parse_aig_head(aig_file)
        if origin_var == 0:
            continue
        
        for trans_times in range(NO_TRANS):
            tmp_aig_filename = os.path.join('./tmp', '{}_{}.aig'.format(
                aig_name, trans_times
            ))
            abc_cmd = 'abc -c "read_aiger {}; strash; {}; write_aiger {}"'.format(
                aig_file, get_syn_command(NO_SYN), tmp_aig_filename
            )
            stdout, _ = run_command(abc_cmd)
            if not os.path.exists(tmp_aig_filename):
                continue
            new_var, new_in, new_latch, new_out, new_and = parse_aig_head(tmp_aig_filename)
            diff = abs(new_var - origin_var) + abs(new_latch - origin_latch) + abs(new_and - origin_and)
            if diff < 10: 
                os.remove(tmp_aig_filename)
                continue
            # Save 
            save_path = './{}/trans_{}_{:}.aig'.format(AIG_DIR, aig_name, trans_times)
            shutil.move(tmp_aig_filename, save_path)
            no_trans += 1
        
        tot_time += time.time() - start_time