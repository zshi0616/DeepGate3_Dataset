import os 
import shlex
import subprocess
import time
import numpy as np

def run_command(command, timeout=-1):
    try: 
        command_list = shlex.split(command)
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        start_time = time.time()
        while process.poll() is None:
            if timeout > 0 and time.time() - start_time > timeout:
                process.terminate()
                process.wait()
                raise TimeoutError(f"Command '{command}' timed out after {timeout} seconds")

            time.sleep(0.1)

        stdout, stderr = process.communicate()
        if len(stderr) > len(stdout):
            return str(stderr).split('\\n'), time.time() - start_time
        else:
            return str(stdout).split('\\n'), time.time() - start_time
    except TimeoutError as e:
        return e, -1
    
def hash_arr(arr):
    p = 1543
    md = 6291469
    hash_res = 1
    tmp_arr = arr.copy()
    tmp_arr = np.sort(tmp_arr)
    for ele in tmp_arr:
        hash_res = (hash_res * p + ele) % md
    return hash_res