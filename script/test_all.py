import time

import numpy as np
import pandas as pd
import os
import json
import subprocess
import signal

def get_log(path):
    """
        retrieve all .log files from training output
    """
    log_list = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            log_list.append(os.path.join(path, file_name))
    return log_list

def log_to_json(log, json_dir):
    """
        convert .log file to configs in .json
    """
    log_name = log.split("/")[-1].split(".")[0]
    with open(log, 'r') as f:
        next(f)

        params = f.readline()
        params = params.replace("ClassBalanceFocal", "ClassBalancedFocal")
        idx = params.find('{')
        params = eval(params[idx:])

        assert isinstance(params, dict)

        exp_id = os.path.basename(log).split('.')[0]
        params['test']['exp_id'] = exp_id

        json_path = os.path.join(json_dir, log_name + '.json')
        with open(json_path, 'w') as f_json:
            json.dump(params, f_json)

    return params, json_path
    
if __name__ == '__main__':
    # for testing trained models all at once
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    # root_path = '/apdcephfs/share_1364275/shenyuan/DrugLT/multi_standard_output/drugbank/logs'
    root_path = '/lanqingli/projects/ImDrug/output/protein_16-4096/logs'
    result_path = '/lanqingli/projects/ImDrug/output3/protein_16-4096/result'

    os.makedirs(result_path, exist_ok=True)
    # json_path = './drugbank_temp.json'
    json_dir = '/lanqingli/projects/ImDrug/output3/protein_16-4096/result'

    log_list = get_log(root_path)

    for log in log_list:
        params, json_path = log_to_json(log, json_dir)

        # print('loss: %s, seed: %d' % (params['loss']['type'], params['seed']))
        start_time = time.time()
        cmd = 'python /lanqingli/code/ImDrug-release-new/script/test.py --config %s > %s 2>&1' % (json_path, os.path.join(result_path, params['test']['exp_id'] + '.log'))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        retrun_code = p.wait()
        # time.sleep(300)
        # os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    