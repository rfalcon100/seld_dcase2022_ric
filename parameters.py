#!/usr/bin/env python
# Helper on how to use the parser:
# https://github.com/bw2/ConfigArgParse

import argparse
import math
import os
import yaml
import shutil
from easydict import EasyDict
from datetime import datetime
import configargparse

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
    #p = configargparse.ArgParser(default_config_files=['./configs/*.yaml'], config_file_parser_class=configargparse.YAMLConfigFileParser)
    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    #p = configargparse.ArgParser(default_config_files=['./configs/*.yaml'])
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path', default='./configs/run_debug.yaml')

    p.add_argument('--exp_name', help="Optional experiment name.")
    p.add_argument('--exp_group', help="Optional experiment group, useful to search for runs in the logs. This is added to the exp name.")
    p.add_argument('--seed_mode', help="Mode for random seeds.", choices=['balanced', 'random'])
    p.add_argument('--seed', type=int)
    p.add_argument('--mode', help='train or eval', choices=['train', 'valid', 'eval'])
    p.add_argument('--debug', action='store_true', help='Enables debug mode, with short runs and no logging.')
    p.add_argument('--job_id', type=str, default='', help='Job id to append to the experiment name. Helps getting the job log.')
    p.add_argument('--logging_dir', help='Directory to save logs and results.')
    p.add_argument('--wandb', action='store_true', help='Enable wandb to log runs.')

    # Run arguments
    p.add_argument('--num_iters', type=int, help='Num of trianing iterations.')
    p.add_argument('--batch_size', type=int, help='Batch size.')
    p.add_argument('--num_workers', type=int, help='Num workers for dataloader.')
    p.add_argument('--print_every', type=int, help='Print current status every x training iterations.')
    p.add_argument('--logging_interval', type=int, help='Validation interval')
    p.add_argument('--lr', type=float, help='Learning rate for optimizer.')
    p.add_argument('--lr_scheduler_step', type=int, help='Step for the lr scheduler')
    p.add_argument('--lr_min', type=float, help='Minimum learning rate so that the Scheduler does not go too low.')
    p.add_argument('--lr_decay_rate', type=float, help='Decay rate for the lr scheduler.')
    p.add_argument('--lr_patience_times', type=float, help='Validaiton steps patienece for the lr scheduler.')

    # Model arguments
    p.add_argument('--model', help='Model to use.')
    p.add_argument('--model_features_transform', help='Features transform to use in the model')
    p.add_argument('--model_augmentation', action='store_true', help='Enable data augmentation in audio domain')
    p.add_argument('--model_loss_fn', help='Loss function.', choices=['mse', 'bce'])
    p.add_argument('--model_normalization', help='Threshold for detecting events during evaluation.')
    p.add_argument('--detection_threshold', type=float, help='Threshold for detecting events during evaluation.')
    p.add_argument('--thresh_unify', type=float, help='Threshold for unify detections during evaluation')
    p.add_argument('--use_mixup', action='store_true')
    p.add_argument('--mixup_alpha', type=float)
    p.add_argument('--input_shape', nargs='+', type=int, help='Input shape for the model. ')   #'input_shape': [4, 144000], when using sample cnn
    p.add_argument('--output_shape', nargs='+', type=int, help='Output shape of the model, so the predictions.')

    # Dataset arguments
    p.add_argument('--dataset_chunk_size_seconds', type=float, help='Chunk size of the input audio, in seconds. For example 1.27, or 2.55.')
    p.add_argument('--dataset_chunk_mode', choices=['random', 'fixed', 'full'])
    p.add_argument('--dataset_multi_track', action='store_true')
    p.add_argument('--dataset_trim_wavs', type=int, help='Trim wavs to this value in seconds when loading. Use -1 to load full wavs.')
    p.add_argument('--dataset_root', nargs='+', type=str)
    p.add_argument('--dataset_list_train', nargs='+')
    p.add_argument('--dataset_root_valid')
    p.add_argument('--dataset_list_valid')
    p.add_argument('--dataset_root_eval', help='Root path for the evaluation dataset. See helper.md for examples.')
    p.add_argument('--dataset_list_eval', help='File with wav filenames for the evaluation dataset. See helper.md for examples.')

    params = p.parse_args()

    # Set fixed values that are not defined in the config files
    params.dataset_chunk_size = math.ceil(24000 * params.dataset_chunk_size_seconds)
    params = vars(params)

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
        params['experiment_description'] = f'{params["exp_group"]}-{params["exp_name"]}-{params["job_id"]}_' \
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


    if params['wandb']:
        import wandb

        wandb_config = {
            "model": params['model'],
            "model_augmentation": params['model_augmentation'],
            "model_features_transform": params['model_features_transform'],
            "model_loss_fn": params['model_loss_fn'],
            "job_id": params['job_id'],
            "num_workers": params['num_workers'],
        }
        wandb.init(project='seld_dcase2022_ric',
                   name=params['exp_name'],
                   tags=['debug' if params['debug'] else 'exp',
                         params['model'],
                         'augmented' if params['model_augmentation'] else 'non-augmented'],
                   group=params['exp_group'] if (params['exp_group'] is not None or params['exp_group'] != '') else None,
                   config=wandb_config,
                   dir=params["logging_dir"],
                   sync_tensorboard=True)

    return EasyDict(params)


# Test
if __name__ == '__main__':
    # Preparation
    config = get_parameters()
    print(config)
