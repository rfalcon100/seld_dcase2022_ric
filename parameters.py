#!/usr/bin/env python

import argparse
import math
import os
import yaml
import shutil
from easydict import EasyDict
from datetime import datetime


def get_result_dir_path(experiment_description: str, root: str = './results'):
    """Returns path where to save training results of a experiment specific result.

    Args:
        root: root path of where to save
        experiment_description: "epoch=50-batch=128-arch=FCN-data=FULL"

    Create the directory, "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    Return directory path(str):
        "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    """
    from datetime import datetime
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d_%H_%M_%S")

    path = f"{experiment_description}__{date_time}"
    path = os.path.join(root, path)
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            print("Path already exists")
        else:
            print(f"Couldn't create {path}.")
            path = root
    else:
        print(f"Save weights to {path}")
    finally:
        return path


def list_of_2d_tuples(s):
    '''
    For the argparser, this reads a string in the format:
    --max_pool 2,2 4,1 3,3
    And returns a list of tuples, as:
    [(2,2), (4,1), (3,3)]

    # Call it like this in terminal
    #--max_pool 2,2 4,1 3,3

    # debug:
    # config = parser.parse_args(["--max_pool", "2,2", "2,2", "2,2", "2,2", "3,3", "1,4"])  # test for the list_of_2d_tuples type
    '''
    try:
        yolo = iter(s.split(' '))
        for tmp in yolo:
            x,y = tmp.split(',')
            return int(x), int(y)
    except:
        raise argparse.ArgumentTypeError("Error reading parameters. Tuples must be x,y")


def get_parameters():
    '''
    Gets the parameters for the experiments, including training settings, dataset, spectrograms, and models..
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', type=str, default='debug', help="Optional experiment name.")
    parser.add_argument('--job_id', type=str, default='', help='Job id to append to the experiment name. Helps getting the job log.')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for dataloader.')

    config = parser.parse_args()

    params = {
        'exp_name': 'debug',  # baseline_dcase2021
        'seed_mode': 'balanced',
        'mode': 'train',
        'num_iters': 10000,  # debug 10000
        'batch_size': 64,  # debug 1
        'num_workers': 0,
        'print_every': 50,
        'logging_interval': 500,  # debug 100 or 50
        'lr': 1e-4,
        'lr_decay_rate': 0.9,
        'lr_patience_times': 3,
        'lr_min': 1e-7,  # should be 1e-7 when using scheduler
        'model': 'crnn10',
        'model_normalization': 'batchnorm',
        'model_loss_fn': 'mse',
        'input_shape': [7, 96, 256],
        'output_shape': [3, 12, 256],
        'logging_dir': './logging',
        'dataset_root': '/m/triton/scratch/work/falconr1/sony/data_dcase2022',
        'dataset_list_train': 'dcase2022_devtrain_all.txt',
        'dataset_list_valid': 'dcase2022_devtrain_all.txt',
        'dataset_trim_wavs': 30,
        'dataset_chunk_size': math.ceil(24000 * 1.27),
        'dataset_chunk_mode': 'random',
        'dataset_multi_track': True,
        'thresh_unify': 15,
    }

    params['exp_name'] = config.exp_name
    params['job_id'] = config.job_id
    params['num_workers'] = config.num_workers

    if '2020' in params['dataset_root']:
        params['unique_classes'] = 14
        params['output_shape'][-2] = 14
    elif '2021' in params['dataset_root']:
        params['unique_classes'] = 12
        params['output_shape'][-2] = 12
    elif '2022' in params['dataset_root']:
        params['unique_classes'] = 13
        params['output_shape'][-2] = 13

    if 'debug' in params['exp_name']:
        params['experiment_description'] = f'{params["exp_name"]}'
    else:
        params['experiment_description'] = f'{params["exp_name"]}-{params["job_id"]}_' \
                                           f'n-work:{params["num_workers"]}_' \
                                           f'{params["model"]}_' \
                                           f'{params["model_normalization"]}_' \
                                           f'_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}'
    params['logging_dir'] = f'{params["logging_dir"]}/{params["experiment_description"]}'
    params['directory_output_results'] = f'{params["logging_dir"]}/tmp_results'

    # Results dir saves:
    #   parameters.yaml

    print("")
    print("================ Experiment ================")
    print(params['experiment_description'])
    print("")

    # Print the experiment config
    ctr = 0
    for k, v in params.items():
        ctr += 1
        if ctr % 10 == 0: print(' ')
        print('{} \t {}'.format(k.ljust(15, ' '), v))
    print("")

    # Save config to disk, create directories if needed
    if 'debug' in params['logging_dir'] and os.path.exists(params['logging_dir']):
        shutil.rmtree(params['logging_dir'])
    if not os.path.exists(params['logging_dir']):
        os.mkdir(params['logging_dir'])
    with open(os.path.join(params['logging_dir'], 'params.yaml'), 'w') as f:
        yaml.dump(params, f, default_flow_style=None)
    if not os.path.exists(params['directory_output_results']):
        os.mkdir(params['directory_output_results'])

    return EasyDict(params)


# Test
if __name__ == '__main__':
    # Preparation
    config = get_parameters()
    print(config)
