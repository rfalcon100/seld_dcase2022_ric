import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os, shutil, math
import yaml
import time
import wandb
from easydict import EasyDict
from datetime import datetime
from itertools import islice
from typing import List

from dataset.dcase_dataset import DCASE_SELD_Dataset, InfiniteDataLoader, _get_padders
from evaluation.dcase2022_metrics import cls_compute_seld_results
from evaluation.evaluation_dcase2022 import write_output_format_file, get_accdoa_labels, get_multi_accdoa_labels, determine_similar_location, all_seld_eval
from solver import Solver
from feature import Feature_StftPlusIV, Feature_MelPlusPhase, Feature_MelPlusIV

import augmentation.spatial_mixup as spm
import torch_audiomentations as t_aug
from parameters import get_parameters
import utils
import plots


def get_dataset(config):
    dataloader_train = None

    datasets_train = []
    for dset_root, dset_list in zip(config.dataset_root, config.dataset_list_train):
        dataset_tmp = DCASE_SELD_Dataset(directory_root=dset_root,
                                         list_dataset=dset_list,
                                         chunk_size=config.dataset_chunk_size,
                                         chunk_mode=config.dataset_chunk_mode,
                                         trim_wavs=config.dataset_trim_wavs,
                                         multi_track=config.dataset_multi_track,
                                         num_classes=config.unique_classes,
                                         labels_backend=config.dataset_backend,
                                         pad_labels=not config.dataset_ignore_pad_labels,
                                         return_fname=False)
        datasets_train.append(dataset_tmp)
    dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    dataloader_train = InfiniteDataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_workers,
                                          shuffle=True, drop_last=True, pin_memory=False)

    dataset_valid = DCASE_SELD_Dataset(directory_root=config.dataset_root_valid,
                                       list_dataset=config.dataset_list_valid,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       labels_backend=config.dataset_backend,
                                       pad_labels=not config.dataset_ignore_pad_labels,
                                       return_fname=True)

    return dataloader_train, dataset_valid

def get_spatial_mixup(device='cpu', p_comp=1.0):
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid'}

    transform = spm.DirectionalLoudness(t_design_degree=params['t_design_degree'],
                                        G_type=params['G_type'],
                                        use_slepian=params['use_slepian'],
                                        order_output=params['order_output'],
                                        order_input=params['order_input'],
                                        backend=params['backend'],
                                        w_pattern=params['w_pattern'],
                                        device=device,
                                        p_comp=p_comp)

    return transform

def get_rotations(device='cpu', p_comp=1.0):
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid'}

    rotation_params = {'rot_phi': 0.0,
                       'rot_theta': 0.0,
                       'rot_psi': 0.0}
    rotation_angles = [rotation_params['rot_phi'], rotation_params['rot_theta'], rotation_params['rot_psi']]

    rotation = spm.SphericalRotation(rotation_angles_rad=rotation_angles,
                                     t_design_degree=params['t_design_degree'],
                                     order_output=params['order_output'],
                                     order_input=params['order_input'],
                                     device=device,
                                     p_comp=p_comp)

    return rotation

def get_rotations_noise(device='cpu', p_comp=1.0):
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid'}

    rotation = spm.SphericalRotation(rotation_angles_rad=[0,0,0],
                                     t_design_degree=params['t_design_degree'],
                                     order_output=params['order_output'],
                                     order_input=params['order_input'],
                                     ignore_labels=True,
                                     device=device, p_comp=p_comp)

    return rotation

def get_audiomentations(p=0.5, fs=24000):
    from augmentation.spliceout import SpliceOut
    from augmentation.MyBandStopFilter import BandStopFilter
    from augmentation.MyBandPassFilter import BandPassFilter
    # Initialize augmentation callable
    apply_augmentation = t_aug.Compose(
        transforms=[
            t_aug.Gain(p=p, min_gain_in_db=-15.0, max_gain_in_db=5.0, mode='per_example', p_mode='per_example'),
            t_aug.PolarityInversion(p=p, mode='per_example', p_mode='per_example'),
            t_aug.PitchShift(p=p, min_transpose_semitones=-1.5, max_transpose_semitones=1.5, sample_rate=fs, mode='per_example', p_mode='per_example'),
            t_aug.AddColoredNoise(p=p, min_snr_in_db=6.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0, sample_rate=fs, mode='per_example', p_mode='per_example'),
            BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000, min_bandwidth_fraction=0.25, max_bandwidth_fraction=1.99, sample_rate=fs, p_mode='per_example'),
            t_aug.LowPassFilter(p=p,  min_cutoff_freq=1000, max_cutoff_freq=7500, sample_rate=fs, p_mode='per_example'),
            t_aug.HighPassFilter(p=p, min_cutoff_freq=100, max_cutoff_freq=2000, sample_rate=fs, p_mode='per_example'),
            BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000, min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.99, sample_rate=fs, p_mode='per_example'),
            #SpliceOut(p=p, num_time_intervals=8, max_width=400, sample_rate=fs, p_mode='per_example')
        ]
    )

    return apply_augmentation

class RandomAugmentations(nn.Sequential):
    def __init__(self, fs=24000, p=1, p_comp=1, n_aug_min=2, n_aug_max=6, threshold_limiter=1):
        super().__init__()
        self.fs = fs
        self.p = p
        self.p_comp = p_comp
        self.n_aug_min = n_aug_min
        self.n_aug_max = n_aug_max
        self.threshold_limiter = threshold_limiter
        mode = 'per_example'  # for speed, we use batch processing
        p_mode = 'per_example'

        self.augmentations = t_aug.SomeOf((n_aug_min, n_aug_max), p=self.p_comp, output_type='dict',
                                      transforms=[
                                          t_aug.Gain(p=p, min_gain_in_db=-15.0, max_gain_in_db=6.0, mode=mode, p_mode=p_mode),
                                          t_aug.PolarityInversion(p=p, mode=mode, p_mode=p_mode),
                                          #t_aug.PitchShift(p=p, min_transpose_semitones=-1.5, max_transpose_semitones=1.5, sample_rate=fs,
                                          #                 mode=mode, p_mode=p_mode),
                                          t_aug.AddColoredNoise(p=p, min_snr_in_db=2.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0,
                                                                sample_rate=fs, mode=mode, p_mode=p_mode),
                                          t_aug.BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
                                                               min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.1, sample_rate=fs,
                                                               p_mode=p_mode),
                                          t_aug.LowPassFilter(p=p, min_cutoff_freq=1000, max_cutoff_freq=5000, sample_rate=fs,
                                                              p_mode=p_mode),
                                          t_aug.HighPassFilter(p=p, min_cutoff_freq=250, max_cutoff_freq=1500, sample_rate=fs,
                                                               p_mode=p_mode),
                                          t_aug.BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
                                                               min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, sample_rate=fs,
                                                               p_mode=p_mode),
                                          #t_aug.SpliceOut(p=p, num_time_intervals=100, max_width=100, sample_rate=fs, p_mode=p_mode)
                                      ]
                                      )

    def forward(self, input):
        do_reshape = False
        if input.shape == 2:
            do_reshape = True
            input = input[None, ...]  #  audiomentations expects batches

        # Augmentations
        output = self.augmentations(input, sample_rate=self.fs)  # Returns ObjectDict
        output = output['samples']

        # Limiter
        torch.clamp(output, min=-self.threshold_limiter, max=self.threshold_limiter)

        if do_reshape:
            output = output.squeeze(0)
        return output


class RandomSpecAugmentations(nn.Sequential):
    def __init__(self, fs=24000, p=1, p_comp=1, n_aug_min=1, n_aug_max=2):
        super().__init__()
        self.fs = fs
        self.p = p
        self.p_comp = p_comp
        self.n_aug_min = n_aug_min
        self.n_aug_max = n_aug_max
        mode = 'per_example'  # for speed, we use batch processing
        p_mode = 'per_example'

        self.augmentations = t_aug.SomeOf((n_aug_min, n_aug_max), p=self.p_comp, output_type='dict',
                                      transforms=[
                                          t_aug.SpecTimeMasking(time_mask_param=24, iid_masks=True, p_proportion=0.3, p=p,
                                                                mode=mode, p_mode=p_mode),
                                          t_aug.SpecFreqMasking(freq_mask_param=24, iid_masks=True, p=p,
                                                                mode=mode, p_mode=p_mode),
                                      ]
                                      )

    def forward(self, input):
        do_reshape = False
        if input.shape == 2:
            do_reshape = True
            input = input[None, ...]  #  audiomentations expects batches

        # Augmentations
        output = self.augmentations(input)  # Returns ObjectDict
        output = output['samples']

        if do_reshape:
            output = output.squeeze(0)
        return output

class CustomFilter(nn.Sequential):
    def __init__(self, fs=24000, p=1, p_comp=1):
        super().__init__()
        self.fs = fs
        self.p = p
        self.p_comp = p_comp
        mode = 'per_batch'  # for speed, we use batch processing
        p_mode = 'per_batch'
        self.augmentations = t_aug.Compose(output_type='tensor',
                                          transforms=[
                                              t_aug.LowPassFilter(p=p, min_cutoff_freq=5000, max_cutoff_freq=5001, sample_rate=fs,
                                                                  p_mode=p_mode),
                                              t_aug.HighPassFilter(p=p, min_cutoff_freq=125, max_cutoff_freq=126, sample_rate=fs,
                                                                   p_mode=p_mode),
                                          ]
                                          )

    def forward(self, input):
        do_reshape = False
        if input.shape == 2:
            do_reshape = True
            input = input[None, ...]  # audiomentations expects batches

        # Augmentations
        output = self.augmentations(input)

        if do_reshape:
            output = output.squeeze(0)
        return output

def main():
    # Get config
    config = get_parameters()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    utils.seed_everything(seed=config.seed, mode=config.seed_mode)

    # Logging configuration
    writer = SummaryWriter(config.logging_dir)

    # Data
    dataloader_train, dataset_valid = get_dataset(config)

    # Solver
    solver = Solver(config=config, tensorboard_writer=writer)

    # Select features and augmentation and rotation
    augmentation_transform_spatial = None
    augmentation_transform_audio = None
    augmentation_transform_spec = None
    rotations_transform = None
    rotations_noise = None

    if config.model_features_transform == 'stft_iv':
        features_transform = Feature_StftPlusIV(nfft=512).to(device)  # mag STFT with intensity vectors
    elif config.model_features_transform == 'mel_iv':
        features_transform = Feature_MelPlusIV().to(device)  # mel spec with intensity vectors
    elif config.model_features_transform == 'mel_phase':
        features_transform = Feature_MelPlusPhase().to(device)  # mel spec with phase difference
    elif config.model_features_transform == 'bandpass':
        features_transform = CustomFilter().to(device)  # Custom Band pass filter to accomodate for the Eigenmike
    else:
        features_transform = None
    print(features_transform)

    if config.model_spatialmixup:
        augmentation_transform_spatial = get_spatial_mixup(device=device, p_comp=0.0).to(device)
    if config.model_augmentation:
        augmentation_transform_audio = RandomAugmentations(p_comp=0.0).to(device)
    if config.model_spec_augmentation:
        augmentation_transform_spec = RandomSpecAugmentations(p_comp=0.0).to(device)
    if config.model_rotations:
        rotations_transform = get_rotations(device=device, p_comp=0.0).to(device)
    if config.model_rotations_noise:
        rotations_noise = get_rotations_noise(device=device, p_comp=0.0).to(device)

    if 'samplecnn' in config.model:
        class t_transform(nn.Sequential):
            def __int__(self):
                super().__init__()
            def forward(self, input):
                out = nn.functional.interpolate(input, scale_factor=(1, 0.1), mode='nearest-exact')
                return out
        target_transform = t_transform()
    else:
        target_transform = None
    print(target_transform)
    print(rotations_transform)
    print(augmentation_transform_spatial)
    print(augmentation_transform_audio)
    print(augmentation_transform_spec)

    # Initial loss:
    x, target = dataloader_train.dataset[0]
    if features_transform is not None:
        x = features_transform(x.unsqueeze(0).to(device))
    else:
        x = x[None, ...].to(device)
    if target_transform is not None:
        target = target_transform(target[None, ...].to(device))
    else:
        target = target[None, ...].to(device)
    solver.predictor.eval()

    # To debug
    #yolo = RandomSpecAugmentations()
    #y = yolo(x)

    out = solver.predictor(x)
    loss = solver.loss_fns[solver.loss_names[0]](out, target)
    print('Initial loss = {:.6f}'.format(loss.item()))

    # Monitoring variables
    train_loss, val_loss, seld_metrics_macro, seld_metrics_micro = 0, 0, None, None
    best_val_step_macro, best_val_loss, best_metrics_macro = 0, 0, [0,0,0,0,99]
    best_val_step_micro, best_val_loss_micro, best_metrics_micro = 0, 0, [0, 0, 0, 0, 99]
    start_time = time.time()

    if config.mode == 'train':
        print('>>>>>>>> Training START  <<<<<<<<<<<<')

        iter_idx = 0
        start_step_time = time.time()
        for data in islice(dataloader_train, config.num_iters + 1):
            #checkpoint_root = '/m/triton/scratch/work/falconr1/dcase2022/seld_dcase2022_ric/logging'
            #checkpoints_path = 'dcase2022_plus_dcase22-sim_FIXED_w-aug_mixup_b32_sample-5573019_n-work:0_samplecnn_batchnorm_144000__2022-06-22-220123'
            #checkpoints_name = 'model_step_170000.pth'
            #checkpoint = os.path.join(checkpoint_root, checkpoints_path, checkpoints_name)
            #solver = Solver(config=config, model_checkpoint=checkpoint)

            train_loss = train_iteration(config, data, iter_idx=iter_idx, start_time=start_time, start_time_step=start_step_time,
                                         device=device, features_transform=features_transform, rotation_noise=rotations_noise,
                                         augmentation_transform_spatial=augmentation_transform_spatial, augmentation_transform_spec=augmentation_transform_spec,
                                         rotation_transform=rotations_transform, augmentation_transform_audio=augmentation_transform_audio,
                                         target_transform=target_transform, solver=solver, writer=writer)

            if iter_idx % config.print_every == 0 and iter_idx > 0:
                start_step_time = time.time()


            if iter_idx % config.logging_interval == 0 and iter_idx > 0:
                seld_metrics_macro, seld_metrics_micro, val_loss = validation_iteration(config, dataset=dataset_valid, iter_idx=iter_idx,
                                                                                        device=device, features_transform=features_transform, target_transform=target_transform,
                                                                                        solver=solver, writer=writer,
                                                                                        dcase_output_folder=config['directory_output_results'])
                curr_time = time.time() - start_time

                # Check for best validation step
                if seld_metrics_macro[4] < best_metrics_macro[4]:
                    best_metrics_macro = seld_metrics_macro
                    best_val_step_macro = iter_idx
                    best_val_loss = val_loss
                if seld_metrics_micro[4] < best_metrics_micro[4]:
                    best_metrics_micro = seld_metrics_micro
                    best_val_step_micro = iter_idx
                if config.wandb:
                    wandb.log({'best_val_step_macro': best_val_step_macro})
                    wandb.summary['BestMACRO/SELD'] = best_metrics_macro[4]
                    wandb.summary['BestMACRO/ER'] = best_metrics_macro[0]
                    wandb.summary['BestMACRO/F'] = best_metrics_macro[1]
                    wandb.summary['BestMACRO/LE'] = best_metrics_macro[2]
                    wandb.summary['BestMACRO/LR'] = best_metrics_macro[3]
                    wandb.summary['Losses/valid'] = best_val_loss
                    wandb.summary['best_val_step_macro'] = best_val_step_macro

                    wandb.log({'best_val_step_micro': best_val_step_micro})
                    wandb.summary['BestMicro/SELD'] = best_metrics_micro[4]
                    wandb.summary['BestMicro/ER'] = best_metrics_micro[0]
                    wandb.summary['BestMicro/F'] = best_metrics_micro[1]
                    wandb.summary['BestMicro/LE'] = best_metrics_micro[2]
                    wandb.summary['BestMicro/LR'] = best_metrics_micro[3]
                    wandb.summary['Losses/valid'] = best_val_loss
                    wandb.summary['best_val_step_micro'] = best_val_step_micro

                # Print metrics
                print(f'Evaluating using overlap = 1 / {config["evaluation_overlap_fraction"]}')
                print(
                    'iteration: {}/{}, time: {:0.2f}, '
                    'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                    'p_comp: {:0.3f}, '.format(iter_idx, config.num_iters, curr_time,
                        train_loss, val_loss,
                        solver.get_curriculum_params()))
                print('====== micro ======')
                print(
                    'best_val_step_micro: {},  \t\t'
                    'micro: ER/F/LE/LR/SELD: {}, '.format(best_val_step_micro,
                    '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(*seld_metrics_micro[0:5]),))
                print(
                    'best_val_step_micro: {},  \t'
                    'BEST-micro: ER/F/LE/LR/SELD: {}, '.format(best_val_step_micro,
                    '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(*best_metrics_micro[0:5]),))
                print('====== MACRO ======')
                print(
                    'best_val_step_macro: {},  \t\t'
                    'MACRO: ER/F/LE/LR/SELD: {}, '.format(best_val_step_macro,
                    '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(*seld_metrics_macro[0:5]),))
                print(
                    'best_val_step_micro: {},  \t'
                    'BEST-MACRO: ER/F/LE/LR/SELD: {}, '.format(best_val_step_macro,
                    '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(*best_metrics_macro[0:5]),))

                print('\n MACRO Classwise results on validation data')
                print('Class\tER\t\tF\t\tLE\t\tLR\t\tSELD_score')
                seld_metrics_class_wise = seld_metrics_macro[5]
                for cls_cnt in range(config['unique_classes']):
                    print('{}\t\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt,
                                                                                     seld_metrics_class_wise[0][cls_cnt],
                                                                                     seld_metrics_class_wise[1][cls_cnt],
                                                                                     seld_metrics_class_wise[2][cls_cnt],
                                                                                     seld_metrics_class_wise[3][cls_cnt],
                                                                                     seld_metrics_class_wise[4][cls_cnt]))
                print('================================================ \n')



            # Schedulers
            if iter_idx > 0:
                solver.curriculum_scheduler_step(iter_idx,
                                                 val_loss if val_loss is not None else 0,
                                                 seld_metrics_macro[4] if seld_metrics_macro is not None else 0)
            if iter_idx % config.lr_scheduler_step == 0 and iter_idx > 0:
                solver.lr_step(seld_metrics_macro[4] if seld_metrics_macro is not None else 0, step=iter_idx)  # LRstep scheduler based on validation SELD score
            iter_idx += 1
        print('>>>>>>>> Training Finished  <<<<<<<<<<<<')
        wandb.finish()

    elif config.mode == 'eval':
        checkpoint_root = '/m/triton/scratch/work/falconr1/dcase2022/seld_dcase2022_ric/logging'
        #checkpoints_path = ['dcase22_plus_dcase22-sim_w-aug-5405063_n-work:0_crnn10_batchnorm_30480__2022-06-15-195028']
        #checkpoints_name = 'model_step_90000.pth'
        checkpoints_path = ['dcase2022_plus_dcase22-sim_w-aug_mixup_b32_sample-gru-5441557_n-work:0_samplecnn_batchnorm_144000__2022-06-19-001924']
        checkpoints_name = 'model_step_190000.pth'
        #checkpoints_path = ['dcase2022_plus_scase22-sim_w-aug_mixup_255_b32-5431498_n-work:0_crnn10_batchnorm_61200__2022-06-17-125119']
        #checkpoints_name = 'model_step_190000.pth'
        #checkpoints_path = ['dcase2022_plus_scase22-sim_w-aug_mixup_255_long-5423265_n-work:0_crnn10_batchnorm_61200__2022-06-16-150605']
        #checkpoints_name = 'model_step_80000.pth'

        checkpoint = os.path.join(checkpoint_root, checkpoints_path[0], checkpoints_name)
        solver = Solver(config=config, model_checkpoint=checkpoint)

        dataset_eval = DCASE_SELD_Dataset(directory_root=config.dataset_root_eval,
                                          list_dataset=config.dataset_list_eval,
                                          chunk_size=config.dataset_chunk_size,
                                          chunk_mode='full',
                                          trim_wavs=config.dataset_trim_wavs,
                                          multi_track=config.dataset_multi_track,
                                          num_classes=config.unique_classes,
                                          return_fname=True,
                                          ignore_labels=True)

        evaluation(config, dataset_eval, solver, features_transform,
                   target_transform=target_transform, dcase_output_folder=config['directory_output_results'],
                   device=device, detection_threshold=config.detection_threshold)

    elif config.mode == 'valid':  # Validation for those where I missed the performance
        checkpoint_root = '/m/triton/scratch/work/falconr1/dcase2022/seld_dcase2022_ric/logging'
        #checkpoints_path = ['dcase2022_plus_scase22-sim_w-aug_mixup_255-5411345_n-work:0_crnn10_batchnorm_61200__2022-06-15-222729']
        #checkpoints_name = 'model_step_80000.pth'
        #checkpoints_path = ['dcase2022_plus_dcase22-sim_w-aug_mixup_b32_sample-gru-5441557_n-work:0_samplecnn_batchnorm_144000__2022-06-19-001924']
        #checkpoints_name = 'model_step_190000.pth'
        checkpoints_path = ['dcase2022_plus_scase22-sim_w-aug_mixup_255_b32-5431498_n-work:0_crnn10_batchnorm_61200__2022-06-17-125119']
        checkpoints_name = 'model_step_190000.pth'
        checkpoints_path = ['dcase2022_plus_scase22-sim_w-aug_mixup_255_long-5423265_n-work:0_crnn10_batchnorm_61200__2022-06-16-150605']
        checkpoints_name = 'model_step_80000.pth'

        checkpoints_path = ['dcase2022_plus_dcase22-sim_no-aug_mixup_b32_-5440404_n-work:0_samplecnn_batchnorm_144000__2022-06-18-201151']
        checkpoints_name = 'model_step_90000.pth'

        checkpoints_path = ['third-2021-crnn10-2.55_spm+aug-5761546_n-work:0_crnn10_batchnorm_61200__2022-07-05-212417']
        checkpoints_name = 'model_step_10000.pth'

        #checkpoints_path = ['six-2021-features-no-grad-crnn10-2.55_base_mel_iv+spm+aug+rot-5820412_n-work:0_crnn10_batchnorm_61200__2022-07-09-201438']
        #checkpoints_name = 'model_step_200000.pth'
        checkpoints_path = ['six-2021-features-no-grad-spec-crnn10-2.55_base_stft_iv+spm+aug+rot-5836347_n-work:0_crnn10_batchnorm_61200__2022-07-11-211006']
        checkpoints_name = 'model_step_180000.pth'

        #checkpoint = os.path.join(checkpoint_root, checkpoints_path[0], checkpoints_name)
        #solver = Solver(config=config, model_checkpoint=checkpoint)
        solver = Solver(config=config)  # TODO this is for loss upper bound only

        seld_metrics_macro, seld_metrics_micro, val_loss = validation_iteration(config, dataset=dataset_valid, iter_idx=0,
                                                           device=device, features_transform=features_transform,
                                                           target_transform=target_transform, solver=solver, writer=None,
                                                           dcase_output_folder=config['directory_output_results'],
                                                           detection_threshold=config['detection_threshold'])
        curr_time = time.time() - start_time
        # Print metrics
        print(f'Evaluating using overlap = 1 / {config["evaluation_overlap_fraction"]}')
        print(
            'iteration: {}/{}, time: {:0.2f}, '
            'train_loss: {:0.4f}, val_loss: {:0.4f}, '
            'p_comp: {:0.3f}, '.format(-1, config.num_iters, curr_time,
                                       train_loss, val_loss,
                                       solver.get_curriculum_params()))
        print('====== micro ======')
        print(
            'best_val_step_micro: {},  \t\t'
            'micro: ER/F/LE/LR/SELD: {}, '.format(-1,
                                                  '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(*seld_metrics_micro[0:5]), ))
        print('====== MACRO ======')
        print(
            'best_val_step_macro: {},  \t\t'
            'MACRO: ER/F/LE/LR/SELD: {}, '.format(-1,
                                                  '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(*seld_metrics_macro[0:5]), ))

        print('\n MACRO Classwise results on validation data')
        print('Class\tER\t\tF\t\tLE\t\tLR\t\tSELD_score')
        seld_metrics_class_wise = seld_metrics_macro[5]
        for cls_cnt in range(config['unique_classes']):
            print('{}\t\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt,
                                                                             seld_metrics_class_wise[0][cls_cnt],
                                                                             seld_metrics_class_wise[1][cls_cnt],
                                                                             seld_metrics_class_wise[2][cls_cnt],
                                                                             seld_metrics_class_wise[3][cls_cnt],
                                                                             seld_metrics_class_wise[4][cls_cnt]))
        print('================================================ \n')

def train_iteration(config, data, iter_idx, start_time, start_time_step, device, features_transform: [nn.Sequential, None], rotation_noise: [nn.Sequential, None],
                    augmentation_transform_spatial: [nn.Sequential, None], rotation_transform: [nn.Sequential, None], augmentation_transform_spec: [nn.Sequential, None],
                    augmentation_transform_audio: [nn.Sequential, None], target_transform: [nn.Sequential, None], solver, writer):
    # Training iteration
    x, target = data
    x, target = x.to(device), target.to(device)

    # Rotation, Augmentation and Feature extraction
    with torch.no_grad():
        if rotation_transform is not None:
            rotation_transform.reset_R(mode=config.model_rotations_mode)
            rotation_transform.p_comp = solver.get_curriculum_params()
            x, target = rotation_transform(x, target)
        if rotation_noise is not None:
            rotation_noise.reset_R(mode='noise')
            rotation_noise.p_comp = solver.get_curriculum_params()
            x, _ = rotation_noise(x)
        if augmentation_transform_spatial is not None:
            augmentation_transform_spatial.reset_G(G_type='spherical_cap_soft')
            augmentation_transform_spatial.p_comp = solver.get_curriculum_params()
            if False:  # Debugging
                augmentation_transform_spatial.plot_response(plot_channel=0, plot_matrix=True, do_scaling=True, plot3d=False)
            x = augmentation_transform_spatial(x)
        if augmentation_transform_audio is not None:
            augmentation_transform_audio = RandomAugmentations(p_comp=solver.get_curriculum_params())
            x = augmentation_transform_audio(x)
        if features_transform is not None:
            x = features_transform(x)
            if augmentation_transform_spec is not None:
                augmentation_transform_spec = RandomSpecAugmentations(p_comp=solver.get_curriculum_params())
                x = augmentation_transform_spec(x)
        if target_transform is not None:
            target = target_transform(target)

    # Train step
    solver.set_input(x, target)
    solver.train_step()

    # Useful debugging
    #plots.plot_labels_cross_sections(target[0].detach().cpu(), n_classes=list(range(target[0].shape[-2])), plot_cartesian=True)
    #plots.plot_labels(target[0].detach().cpu(), n_classes=list(range(target[0].shape[-2])), savefig=False, plot_cartesian=True)

    # Output training stats
    train_loss = solver.loss_values['rec']

    # Logging and printing
    if writer is not None:
        step = iter_idx
        if iter_idx % 100 == 0:
            writer.add_scalar('Losses/train', train_loss.item(), step)
            #if config.wandb:
            #    wandb.log({'Losses/train': train_loss.item()}, step=step)

            # Learning rates
            lr = solver.get_lrs()
            writer.add_scalar('Lr/gen', lr, step)

            # Grad norm
            grad_norm_model = solver.get_grad_norm()
            writer.add_scalar('grad_norm/disc', grad_norm_model, step)

            # Scheduler
            if augmentation_transform_audio is not None or rotation_transform is not None or rotation_noise is not None or augmentation_transform_spatial is not None or augmentation_transform_spec is not None:
                p_comp = solver.get_curriculum_params()
                writer.add_scalar('params/p_comp', p_comp, iter_idx)

    if iter_idx % config.print_every == 0:
        curr_time = time.time() - start_time
        step_time = time.time() - start_time_step
        print('[%d/%d] iters \t Loss_rec: %.6f \t\t Step_time: %0.2f  \t\t  Elapsed time: %0.2f'
              % (iter_idx, config.num_iters, train_loss.item(), step_time, curr_time))

    # Log an example of the predicted labels
    if (iter_idx % config.logging_interval == 0) and writer is not None:
        fixed_output = solver.get_fixed_output()
        fixed_label = solver.get_fixed_label()
        if fixed_output is not None:
            fixed_output = fixed_output[0, ...]
            fixed_label = fixed_label[0, ...]
            fixed_error = np.abs(fixed_output - fixed_label)

            # Transform to spherical coordinates
            fixed_output_sph = np.zeros_like(fixed_output)
            fixed_label_sph = np.zeros_like(fixed_label)
            fixed_error_sph = np.zeros_like(fixed_error)
            for cc in range(fixed_output_sph.shape[1]):
                tmp = utils.vecs2dirs(fixed_output[:, cc, :].squeeze().transpose(1, 0), include_r=True, use_elevation=True)
                fixed_output_sph[:, cc, ::] = tmp.transpose(1, 0)
                tmp = utils.vecs2dirs(fixed_label[:, cc, :].squeeze().transpose(1, 0), include_r=True, use_elevation=True)
                fixed_label_sph[:, cc, ::] = tmp.transpose(1, 0)
                tmp = utils.vecs2dirs(fixed_error[:, cc, :].squeeze().transpose(1, 0), include_r=True, use_elevation=True)
                fixed_error_sph[:, cc, ::] = tmp.transpose(1, 0)

            # Plot
            fig = plots.plot_labels(fixed_output_sph, savefig=False, plot_cartesian=False, title='Output')
            writer.add_figure('fixed_output/train', fig, iter_idx)
            fig = plots.plot_labels(fixed_label_sph, savefig=False, plot_cartesian=False, title='Target')
            writer.add_figure('fixed_label/train', fig, None)
            #fig = plots.plot_labels(fixed_error_sph, savefig=False, plot_cartesian=False)
            #writer.add_figure('fixed_error/train', fig, iter_idx)

    if (iter_idx % config.logging_interval == 0) and iter_idx > 0:
        torch.save(solver.predictor.state_dict(), os.path.join(config.logging_dir, f'model_step_{iter_idx}.pth'))

    return train_loss.item()

def validation_iteration(config, dataset, iter_idx, solver, features_transform, target_transform: [nn.Sequential, None], dcase_output_folder, device, writer, detection_threshold=0.5):
    # Adapted from the official baseline
    nb_test_batches, test_loss = 0, 0.
    model = solver.predictor
    model.eval()
    file_cnt = 0
    overlap = 1 / config['evaluation_overlap_fraction']  # defualt should be 1  TODO, onluy works for up to 1/32 , when the labels are 128 frames long.

    print(f'Validation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for ctr, (audio, target, fname) in enumerate(dataset):
            # load one batch of data
            audio, target = audio.to(device), target.to(device)
            duration = dataset.durations[fname]
            print(f'Evaluating file {ctr+1}/{len(dataset)}: {fname}')
            print(f'Audio shape: {audio.shape}')
            print(f'Target shape: {target.shape}')

            warnings.warn('WARNING: Hard coded chunk size for evaluation')
            audio_padding, labels_padding = _get_padders(chunk_size_seconds=config.dataset_chunk_size / dataset._fs[fname],
                                                         duration_seconds=math.floor(duration),
                                                         overlap=overlap,
                                                         audio_fs=dataset._fs[fname],
                                                         labels_fs=100)

            # Split each wav into chunks and process them
            audio = audio_padding['padder'](audio)
            audio_chunks = audio.unfold(dimension=1, size=audio_padding['chunk_size'], step=audio_padding['hop_size']).permute((1, 0, 2))

            if config.dataset_multi_track:
                labels = labels_padding['padder'](target.permute(1,2,3,0))
                labels_chunks = labels.unfold(dimension=-1, size=labels_padding['chunk_size'], step=labels_padding['hop_size'])
                labels_chunks = labels_chunks.permute((3, 4, 0, 1, 2))
            else:
                labels = labels_padding['padder'](target)
                labels_chunks = labels.unfold(dimension=-1, size=labels_padding['chunk_size'], step=labels_padding['hop_size'])
                labels_chunks = labels_chunks.permute((2, 0, 1, 3))

            full_output = []
            full_loss = []
            full_labels = []
            if audio_chunks.shape[0] != labels_chunks.shape[0]:
                a = 1
                warnings.warn('WARNING: Possible error in padding.')
            if audio_chunks.shape[0] > labels_chunks.shape[0]:
                audio_chunks = audio_chunks[0:labels_chunks.shape[0], ...]  # Mmm... lets drop the extra audio chunk if there are no labels for it
            if audio_chunks.shape[0] < labels_chunks.shape[0]:
                audio_chunks = torch.concat([audio_chunks, torch.zeros_like(audio_chunks[0:1])])  # Mmm... lets add an empty audio slice
            tmp = torch.utils.data.TensorDataset(audio_chunks, labels_chunks)
            loader = DataLoader(tmp, batch_size=1, shuffle=False, drop_last=False)  # Loader per wav to get batches
            for ctr, (audio, labels) in enumerate(loader):
                if features_transform is not None:
                    audio = features_transform(audio)
                if target_transform is not None:
                    labels = target_transform(labels)
                output = model(audio)
                if config.oracle_mode:
                    output = torch.zeros_like(labels)  # TODO This is just to get the upper bound of the loss
                    if config.dataset_multi_track:
                        output = torch.zeros(size=(labels.shape[0], labels.shape[1], 3*3*12), device=device)  # TODO This is just to get the upper bound of the loss wih mACCDOA
                loss = solver.loss_fns[solver.loss_names[0]](output, labels)
                full_output.append(output)
                full_loss.append(loss)
                if config.oracle_mode:
                    full_labels.append(labels)  # TODO This is just to get the upper bound of the loss
                if torch.isnan(loss):
                    raise ValueError('ERROR: NaNs in loss')

            # Concatenate chunks across timesteps into final predictions
            if config.dataset_multi_track:
                output = torch.concat(full_output, dim=-2)
            else:
                if overlap == 1:
                    output = torch.concat(full_output, dim=-1)
                    if config.oracle_mode:
                        output = torch.concat(full_labels, dim=-1)   # TODO This is just to get the upper bound of the loss
                else:
                    # TODO: maybe this is ready now? at least until overlap 1/32
                    # TODO: No, it only works when validating the ground truth labels, but not the final predictions
                    # Rebuild when using overlap
                    # This is basically a folding operation, using an average of the predictions of each overlapped chunk
                    aa = len(full_output) - 1
                    if config.oracle_mode:
                        full_output = full_labels # TODO This is just to get the upper bound of the loss
                    resulton = torch.zeros(aa, labels.shape[-3], labels.shape[-2], labels_padding['full_size'] + labels_padding['padder'].padding[-3])
                    resulton = torch.zeros(aa, labels.shape[-3], labels.shape[-2],
                                           labels_padding['full_size'] + labels_padding['padder'].padding[-3] + labels_padding['hop_size'])
                    weights = torch.zeros(1, labels_padding['full_size'] + labels_padding['padder'].padding[-3] + labels_padding['hop_size'])
                    for ii in range(0, aa):
                        #print(ii)
                        start_i = ii * labels_padding['hop_size']
                        end_i = start_i + round(labels_padding['hop_size'] * 1/1)
                        end_i = start_i + round(labels_padding['chunk_size'] * 1 / 1)
                        if end_i > resulton.shape[-1]:  # Drop the last part
                            end_i = resulton.shape[-1]
                        #yolingon = full_output[ii][0]
                        try:
                            resulton[ii, :, :, start_i:end_i] = full_output[ii][0,..., 0:end_i-start_i]
                        except:
                            a = 1
                            warnings.warn('WARNING: Error while evaluating with overlap')
                        weights[:, start_i:end_i] = weights[:, start_i:end_i] + 1

                    output = torch.sum(resulton, dim=0, keepdim=True) / weights
                    if torch.any(torch.isnan(output)):
                        warnings.warn('WARNING: NaNs detected in output')

            # Apply detection threshold based on vector norm
            if config.dataset_multi_track:
                pass
            else:
                norms = torch.linalg.vector_norm(output, ord=2, dim=-3, keepdims=True)
                norms = (norms < detection_threshold).repeat(1, output.shape[-3], 1, 1)
                output[norms] = 0.0
            loss = torch.tensor([x.item() for x in full_loss]).mean()

            # Useful fo debug:
            #output.detach().cpu().numpy()[0, 0]
            #plots.plot_labels(labels.detach().cpu().numpy()[0])
            #target.detach().cpu().numpy()[0]

            # Downsample over frames:
            if config.dataset_multi_track:
                if target_transform is None:
                    output = nn.functional.interpolate(output.permute(0, 2, 1), scale_factor=(0.1), mode='nearest-exact').permute(0, 2, 1)
            else:
                if target_transform is None:
                    output = nn.functional.interpolate(output, scale_factor=(1, 0.1), mode='nearest-exact')

                # I think the baseline code needs this in [batch, frames, classes*coords]
                output = output.permute([0, 3, 1, 2])
                output = output.flatten(2, 3)

            if config['dataset_multi_track'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred0 = cls_compute_seld_results.reshape_3Dto2D(sed_pred0)
                doa_pred0 = cls_compute_seld_results.reshape_3Dto2D(doa_pred0)
                sed_pred1 = cls_compute_seld_results.reshape_3Dto2D(sed_pred1)
                doa_pred1 = cls_compute_seld_results.reshape_3Dto2D(doa_pred1)
                sed_pred2 = cls_compute_seld_results.reshape_3Dto2D(sed_pred2)
                doa_pred2 = cls_compute_seld_results.reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred = cls_compute_seld_results.reshape_3Dto2D(sed_pred)
                doa_pred = cls_compute_seld_results.reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            tmp_name = fname.split('/')[-1]
            output_file = os.path.join(dcase_output_folder, tmp_name.replace('.wav', '.csv'))
            file_cnt += 1
            output_dict = {}
            if config['dataset_multi_track'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt],
                                                                doa_pred0[frame_cnt], doa_pred1[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt],
                                                                doa_pred1[frame_cnt], doa_pred2[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt],
                                                                doa_pred2[frame_cnt], doa_pred0[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                           doa_pred_fc[class_cnt + config['unique_classes']],
                                                           doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
                output_dict_polar = {}
                for k, v in output_dict.items():
                    ss = []
                    for this_item in v:
                        tmp = utils.cart2sph(this_item[-3], this_item[-2], this_item[-1])
                        ss.append([this_item[0], *tmp])
                    output_dict_polar[k] = ss
                write_output_format_file(output_file, output_dict_polar, use_cartesian=False)
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            tmp_azi, tmp_ele = utils.cart2sph(doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + config['unique_classes']],
                                                           doa_pred[frame_cnt][class_cnt + 2 * config['unique_classes']])
                            output_dict[frame_cnt].append([class_cnt, tmp_azi, tmp_ele])
                            #output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + config['unique_classes']],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + 2 * config['unique_classes']]])
                write_output_format_file(output_file, output_dict, use_cartesian=False)

            test_loss += loss.item()
            nb_test_batches += 1

        test_loss /= nb_test_batches

    all_test_metric_macro, all_test_metric_micro = all_seld_eval(config, directory_root=dataset.directory_root, fnames=dataset._fnames, pred_directory=dcase_output_folder)

    if writer is not None:
        writer.add_scalar('Losses/valid', test_loss, iter_idx)
        writer.add_scalar('MMacro/ER', all_test_metric_macro[0], iter_idx)
        writer.add_scalar('MMacro/F', all_test_metric_macro[1], iter_idx)
        writer.add_scalar('MMacro/LE', all_test_metric_macro[2], iter_idx)
        writer.add_scalar('MMacro/LR', all_test_metric_macro[3], iter_idx)
        writer.add_scalar('MMacro/SELD', all_test_metric_macro[4], iter_idx)

        writer.add_scalar('Mmicro/ER', all_test_metric_micro[0], iter_idx)
        writer.add_scalar('Mmicro/F', all_test_metric_micro[1], iter_idx)
        writer.add_scalar('Mmicro/LE', all_test_metric_micro[2], iter_idx)
        writer.add_scalar('Mmicro/LR', all_test_metric_micro[3], iter_idx)
        writer.add_scalar('Mmicro/SELD', all_test_metric_micro[4], iter_idx)

    return all_test_metric_macro, all_test_metric_micro, test_loss

def evaluation(config, dataset, solver, features_transform, target_transform: [nn.Sequential, None], dcase_output_folder, device, detection_threshold=0.4):
    # Adapted from the official baseline
    # This is basically the same as validaiton, but we dont compute losses or metrics because there are no labels
    nb_test_batches, test_loss = 0, 0.
    model = solver.predictor
    model.eval()
    file_cnt = 0

    print(f'Evaluation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for ctr, (audio, _, fname) in enumerate(dataset):
            # load one batch of data
            audio = audio.to(device)
            duration = dataset.durations[fname]
            print(f'Evaluating file {ctr+1}/{len(dataset)}: {fname}')
            print(f'Audio shape: {audio.shape}')

            warnings.warn('WARNING: Hard coded chunk size for evaluation')
            audio_padding, labels_padding = _get_padders(chunk_size_seconds=config.dataset_chunk_size / dataset._fs[fname],
                                                         duration_seconds=math.floor(duration),
                                                         overlap=1,
                                                         audio_fs=dataset._fs[fname],
                                                         labels_fs=100)

            # Split each wav into chunks and process them
            audio = audio_padding['padder'](audio)
            audio_chunks = audio.unfold(dimension=1, size=audio_padding['chunk_size'],
                                        step=audio_padding['hop_size']).permute((1, 0, 2))

            full_output = []
            tmp = torch.utils.data.TensorDataset(audio_chunks)
            loader = DataLoader(tmp, batch_size=1, shuffle=False, drop_last=False)  # Loader per wav to get batches
            for ctr, (audio) in enumerate(loader):
                audio = audio[0]
                if features_transform is not None:
                    audio = features_transform(audio)
                output = model(audio)
                full_output.append(output)

            # Concatenate chunks across timesteps into final predictions
            if config.dataset_multi_track:
                output = torch.concat(full_output, dim=-2)
            else:
                output = torch.concat(full_output, dim=-1)

            # Apply detection threshold based on vector norm
            if config.dataset_multi_track:
                pass
            else:
                norms = torch.linalg.vector_norm(output, ord=2, dim=-3, keepdims=True)
                norms = (norms < detection_threshold).repeat(1, output.shape[-3], 1, 1)
                output[norms] = 0.0

            # Downsample over frames:
            if config.dataset_multi_track:
                if target_transform is None:
                    output = nn.functional.interpolate(output.permute(0, 2, 1), scale_factor=(0.1), mode='nearest-exact').permute(0, 2, 1)
            else:
                if target_transform is None:
                    output = nn.functional.interpolate(output, scale_factor=(1, 0.1), mode='nearest-exact')

                # I think the baseline code needs this in [batch, frames, classes*coords]
                output = output.permute([0, 3, 1, 2])
                output = output.flatten(2, 3)

            if config['dataset_multi_track'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred0 = cls_compute_seld_results.reshape_3Dto2D(sed_pred0)
                doa_pred0 = cls_compute_seld_results.reshape_3Dto2D(doa_pred0)
                sed_pred1 = cls_compute_seld_results.reshape_3Dto2D(sed_pred1)
                doa_pred1 = cls_compute_seld_results.reshape_3Dto2D(doa_pred1)
                sed_pred2 = cls_compute_seld_results.reshape_3Dto2D(sed_pred2)
                doa_pred2 = cls_compute_seld_results.reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred = cls_compute_seld_results.reshape_3Dto2D(sed_pred)
                doa_pred = cls_compute_seld_results.reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            tmp_name = fname.split('/')[-1]
            output_file = os.path.join(dcase_output_folder, tmp_name.replace('.wav', '.csv'))
            file_cnt += 1
            output_dict = {}
            if config['dataset_multi_track'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt],
                                                                doa_pred0[frame_cnt], doa_pred1[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt],
                                                                doa_pred1[frame_cnt], doa_pred2[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt],
                                                                doa_pred2[frame_cnt], doa_pred0[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                           doa_pred_fc[class_cnt + config['unique_classes']],
                                                           doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            tmp_azi, tmp_ele = utils.cart2sph(doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + config['unique_classes']],
                                                           doa_pred[frame_cnt][class_cnt + 2 * config['unique_classes']])
                            output_dict[frame_cnt].append([class_cnt, tmp_azi, tmp_ele])
                            #output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + config['unique_classes']],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + 2 * config['unique_classes']]])
            write_output_format_file(output_file, output_dict, use_cartesian=False, ignore_src_id=True)

    print('Finished evaluation')

def debug_plot_helper(target):
    import plots
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Useful debugging
    plots.plot_labels_cross_sections(target.detach().cpu(), n_classes=list(range(target.shape[-2])), plot_cartesian=True)
    plots.plot_labels(target.detach().cpu(), n_classes=list(range(target.shape[-2])), savefig=False, plot_cartesian=True)

    plt.figure()
    plt.show()


def count_active_classes(all_labels: List, detection_threshold=0.5):
    """ Useful function to get the histogram of active classes per frames.
    Tip: Call it with only one label to get the plot.
        count_active_classes(all_labels[0:1])
    """
    import plots
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if len(all_labels) == 1:
        plots.plot_labels_cross_sections(all_labels[0], n_classes=list(range(all_labels[0].shape[-2])), plot_cartesian=True)
        plots.plot_labels(all_labels[0], n_classes=list(range(all_labels[0].shape[-2])), savefig=False, plot_cartesian=True)

    all_count_detections = {}
    for i in range(len(all_labels)):
        this_label = all_labels[i]
        vec_norms = torch.linalg.vector_norm(this_label, ord=2, dim=-3)

        for cls in range(this_label.shape[-2]):
            mask_detected_events = vec_norms[cls, :] > detection_threshold  # detected events for this class
            # mask_detected_events = mask_detected_events.repeat(1, 3, 1)
            tmp_events = this_label[..., cls, mask_detected_events]
            # detections = tmp_events[mask_detected_events]
            this_count_detections = mask_detected_events.nonzero(as_tuple=False)
            if cls in all_count_detections.keys():
                all_count_detections[cls] += len(this_count_detections)
            else:
                all_count_detections[cls] = len(this_count_detections)

    f, ax = plt.subplots(figsize=(10, 15))
    df = pd.DataFrame(list(all_count_detections.items()))
    df.columns = ['class_id', 'count']
    g = sns.barplot(x="class_id", y="count", data=df,
                    label="class_id", color="b")
    sns.despine(left=False, bottom=False)
    #g.set_yscale("log")
    plt.show()

if __name__ == '__main__':
    main()
