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
    parser.add_argument('--mode', type=str, default='train', help='train or eval', choices=['train', 'valid', 'eval'])
    parser.add_argument('--job_id', type=str, default='', help='Job id to append to the experiment name. Helps getting the job log.')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for dataloader.')
    parser.add_argument('--detection_threshold', type=float, default=0.4, help='Threshold for detecting events during evaluation.')

    # Model arguments
    parser.add_argument('--model', type=str, default='crnn10', help='Model to sue')
    parser.add_argument('--model_features_transform', type=str, default='stft_iv', help='Features transform to use in the model')
    parser.add_argument('--model_augmentation', action="store_true", help='Enable data augmentation in audio domain')

    # Dataset arguments
    parser.add_argument('--dataset_chunk_size_seconds', type=float, default=2.55, help='Chunk size of the input audio, in seconds. For example 1.27, or 2.55.')
    parser.add_argument('--dataset_root_eval', type=str, default='/m/triton/scratch/work/falconr1/sony/data_dcase2022', help='Root path for the evaluation dataset. See helper.md for examples.')
    parser.add_argument('--dataset_list_eval', type=str, default='dcase2022_eval_all.txt', help='File with wav filenames for the evaluation dataset. See helper.md for examples.')
    config = parser.parse_args()

    params = {
        'exp_name': config.exp_name,  # debug1
        'seed_mode': 'balanced',
        'job_id': config.job_id,
        'mode': config.mode,
        'num_iters': 200000,  # debug 10000
        'batch_size': 32,  # debug 2
        'num_workers': config.num_workers,
        'print_every': 50,
        'logging_interval': 10000,  # debug 100 or 50
        'lr': 1e-4,
        'lr_decay_rate': 0.9,
        'lr_patience_times': 3,
        'lr_min': 1e-7,  # should be 1e-7 when using scheduler
        'model': config.model,
        'model_features_transform': config.model_features_transform,
        'model_augmentation': config.model_augmentation,
        'model_normalization': 'batchnorm',
        'model_loss_fn': 'mse',
        #'input_shape': [7, 96, 256],
        'input_shape': [4, 144000],
        'output_shape': [3, 12, 256],
        'logging_dir': './logging',
        #'dataset_root': ['/m/triton/scratch/work/falconr1/sony/data_dcase2022',
        #                 '/m/triton/scratch/work/falconr1/sony/data_dcase2022_sim'],
        #'dataset_list_train': ['dcase2022_devtrain_all.txt',
        #                       'dcase2022_sim_all.txt'],
        'dataset_root': ['/m/triton/scratch/work/falconr1/sony/data_dcase2022'],
        'dataset_list_train': ['dcase2022_devtrain_all.txt'],
        'dataset_root_valid': '/m/triton/scratch/work/falconr1/sony/data_dcase2022',
        'dataset_list_valid': 'dcase2022_devtest_all.txt',
        'dataset_root_eval': config.dataset_root_eval,
        'dataset_list_eval': config.dataset_list_eval,
        'dataset_trim_wavs':30,
        'dataset_chunk_size': math.ceil(24000 * config.dataset_chunk_size_seconds),
        'dataset_chunk_size_seconds': config.dataset_chunk_size_seconds,
        'dataset_chunk_mode': 'random',
        'dataset_multi_track': False,
        'thresh_unify': 15,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'detection_threshold': config.detection_threshold,
    }

    if '2020' in params['dataset_root_valid']:
        params['unique_classes'] = 14
        params['output_shape'][-2] = 14
    elif '2021' in params['dataset_root_valid']:
        params['unique_classes'] = 12
        params['output_shape'][-2] = 12
    elif '2022' in params['dataset_root_valid']:
        params['unique_classes'] = 13
        params['output_shape'][-2] = 13

    if 'debug' in params['exp_name']:
        params['experiment_description'] = f'{params["exp_name"]}'
    else:
        params['experiment_description'] = f'{params["exp_name"]}-{params["job_id"]}_' \
                                           f'n-work:{params["num_workers"]}_' \
                                           f'{params["model"]}_' \
                                           f'{params["model_normalization"]}_' \
                                           f'{params["dataset_chunk_size"]}_' \
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
