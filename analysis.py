import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from typing import Union

import os, shutil, math
import time
from datetime import datetime
from itertools import islice
from typing import List
from tqdm import tqdm
import IPython
import warnings

from dataset.dcase_dataset import DCASE_SELD_Dataset, InfiniteDataLoader, _get_padders
from evaluation.dcase2022_metrics import cls_compute_seld_results
from evaluation.evaluation_dcase2022 import write_output_format_file, get_accdoa_labels, get_multi_accdoa_labels, determine_similar_location, all_seld_eval
from solver import Solver
from feature import Feature_StftPlusIV

import augmentation.spatial_mixup as spm
import torch_audiomentations as t_aug
from parameters import get_parameters
import utils
import plots
import seaborn as sns
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif',
              font_scale=1, color_codes=True, rc={'pcolor.shading': 'auto'})

# Find active frames (where at least 1 class has norm > threshold)
#norma = torch.linalg.vector_norm(targets, ord=2, dim=-3)  # [batch, n_classes, frames]
#mask_active_per_frames = torch.any(norma > threshold, dim=-2)  # [batch, frames]
#active_targets = targets[..., mask_active_per_frames[-1]]  # [batch, 3, n_classes, frames]


def plot_histograms_bivariate_azi_ele(targets, threshold=0.5):
    """ Here I plot the 2d histograms of azimuths and elevations"""
    assert len(targets.shape) == 4, 'Should be [batch, 3, n_classes, frames]'

    # Find active frames (where at least 1 class has norm > threshold)
    norma = torch.linalg.vector_norm(targets, ord=2, dim=-3)  # [batch, n_classes, frames]
    mask_active_per_frames = torch.any(norma > threshold, dim=-2)  # [batch, frames]
    active_targets = targets[..., mask_active_per_frames[-1]]  # [batch, 3, n_classes, frames]

    print(f'All frames shape: {targets.shape}')
    print(f'Active frames with at least 1 active class shape: {active_targets.shape}')

    # Optional, look at the trayectories:
    plots.plot_labels_cross_sections(active_targets[0, ..., 0:100000])

    # Reshape into points (or DOAs), and find active frames of any class
    active_targets = active_targets.permute((0,2,3,1)).reshape((-1, 3)).contiguous()  # [batch * n_classes * frames, 3]
    norma = torch.linalg.vector_norm(active_targets, ord=2, dim=-1)
    mask_active_per_frames = norma > threshold
    active_targets_all_classes = active_targets[mask_active_per_frames, ...]
    print(f'Active frames of all classes: {active_targets.shape}')

    plots.plot_distribution_azi_ele(active_targets_all_classes, type='hex', log_scale=True, title='Original', gridsize=30, bins=20)
    # plots.plot_distribution_azi_ele(active_targets_all_classes, type='hist', log_scale=False, title='Original', cmin=1, gridsize=100)


def plot_histograms_active_per_class(all_labels: List, all_labels_test: List = None, all_labels_sim: List = None,
                                     splits=['dev-train', 'dev-test', 'synth-set'], detection_threshold=0.5,
                                     format_use_log=True, filename='dcase22_hist_active_per_class',
                                     sound_event_classes_2022=['Female speech',
                                                               'Male speech',
                                                               'Clapping',
                                                               'Telephone',
                                                               'Laughter',
                                                               'Domestic sounds',
                                                               'Walk, footsteps',
                                                               'Door, open or close',
                                                               'Music',
                                                               'Musical instrument',
                                                               'Water tap, faucet',
                                                               'Bell',
                                                               'Knock']
                                     ):
    """ Here we compute the histograms of active frames per class.
    Pass 2 sets of labels to compare the splits.
    """
    all_count_detections, all_count_detections_test, all_count_detections_synth = {}, {}, {}

    for list_of_labels, dict_of_dectections in zip([all_labels, all_labels_test, all_labels_sim], [all_count_detections, all_count_detections_test, all_count_detections_synth]):
        for i in range(len(list_of_labels)):
            this_label = list_of_labels[i]
            vec_norms = torch.linalg.vector_norm(this_label, ord=2, dim=-3)

            for cls in range(this_label.shape[-2]):
                #dict_of_dectections[cls] = 0  # Add zero to have the class in the dictionary
                mask_detected_events = vec_norms[cls, :] > detection_threshold  # detected events for this class
                # mask_detected_events = mask_detected_events.repeat(1, 3, 1)
                tmp_events = this_label[..., cls, mask_detected_events]
                # detections = tmp_events[mask_detected_events]
                this_count_detections = mask_detected_events.nonzero(as_tuple=False)
                if cls in dict_of_dectections.keys():
                    dict_of_dectections[cls] += len(this_count_detections)
                else:
                    dict_of_dectections[cls] = len(this_count_detections)

    # df = pd.DataFrame.from_dict(all_count_detections, orient='index', columns=['count'])
    df = pd.DataFrame(list(all_count_detections.items()))
    df.columns = ['class_id', 'count']
    df['class_name'] = sound_event_classes_2022
    split = [splits[0]] * len(sound_event_classes_2022)
    df['split'] = split

    # Second dataframe
    df2 = pd.DataFrame(list(all_count_detections_test.items()))
    df2.columns = ['class_id', 'count']
    df2['class_name'] = sound_event_classes_2022
    split = [splits[1]] * len(sound_event_classes_2022)
    df2['split'] = split

    # Third dataframe
    df3 = pd.DataFrame(list(all_count_detections_synth.items()))
    df3.columns = ['class_id', 'count']
    df3['class_name'] = sound_event_classes_2022
    split = [splits[2]] * len(sound_event_classes_2022)
    df3['split'] = split

    frames = [df, df2, df3]
    df = pd.concat(frames)

    if False:
        #Vertical plot
        f, ax = plt.subplots(figsize=(12, 12))
        g = sns.barplot(y="class_name", x="count", data=df, hue='split', palette='magma')
        # sns.despine(left=False, bottom=False)
        if format_use_log:
            g.set_xscale("log")
            g.set_xticks([10 ** x for x in range(6)])
            # g.set_xticklabels(['0','a','b','c','d','e'])
        plt.show()

    # Horizontal, looks nice
    f, ax = plt.subplots(figsize=(18, 7))
    g = sns.barplot(x="class_name", y="count", data=df, hue='split', palette='magma')
    # g = sns.catplot(x="class_name", kind='count', data=df, hue='split', palette='magma')
    # sns.despine(left=False, bottom=False)
    if format_use_log:
        g.set_yscale("log")
        g.set_yticks([10 ** x for x in range(6)])
        # g.set_xticklabels(['0','a','b','c','d','e'])
    g.set_xticklabels(g.get_xticklabels(), rotation=35)
    plt.tight_layout()
    plt.savefig(f'./figures/{filename}.pdf')
    plt.savefig(f'./figures/{filename}.png')
    plt.show()

def plot_histograms_polyphony(targets: List, detection_threshold=0.5, format_use_log=False, chunk_size=128):
    # Here I test a small plot to get histograms of active sources per chunk
    # This is the polyphony
    # 17.03 it kinda works now
    y = torch.concat(targets, dim=-1)[None,...]   # concat over frames
    #n_examples = y.shape[0]
    n_examples = len(targets)

    ####chunk_size = 128  # This how big the chunk is, if = 128, then we dont split the example at all
    all_active_sources = []
    n_chunks = int(y.shape[-1] / chunk_size)  # n_chunks for each example
    #y_chunks = torch.chunk(y, chunks=n_chunks, dim=-1)
    pad_size = chunk_size - y.shape[-1] % chunk_size
    padder = nn.ConstantPad2d(padding=(0, pad_size, 0, 0), value=0.0)  # Hack to pad 6001 --> 6032, for full files only
    y_chunks = torch.chunk(padder(y), chunks=n_chunks + 1, dim=-1)

    y = torch.cat(y_chunks, dim=0)
    norma = torch.linalg.vector_norm(y, ord=2, dim=-3)  # [n_classes, frames]
    mask_active_sources = (norma > detection_threshold).any(dim=-1)

    for i in range(y.shape[0]):
        n_active_sources = len(mask_active_sources[i].nonzero(as_tuple=False))
        all_active_sources.append(n_active_sources)
    all_active_sources = torch.tensor(all_active_sources)

    # Plot
    fig = plt.figure()
    plt.hist(all_active_sources.detach().cpu().numpy(), bins=[0, 1, 2, 3, 5, 6, 7, 8], density=True)
    ax = plt.gca()
    ax.set_title(f'n_examples = {n_examples} chunk_size = {chunk_size}')
    plt.show()

    return
    # Ths is not ready

    # TODO This is not ready
    # Plot with sns
    df = pd.DataFrame(list(all_active_sources.items()))
    df.columns = ['polyphony', 'count']
    #df['polyphony'] = sound_event_classes_2022
    #split = [splits[0]] * len(sound_event_classes_2022)
    #df['split'] = split

    # Horizontal, looks nice
    f, ax = plt.subplots(figsize=(18, 7))
    g = sns.barplot(x="polyphony", y="count", data=df, palette='magma')
    # g = sns.catplot(x="class_name", kind='count', data=df, hue='split', palette='magma')
    # sns.despine(left=False, bottom=False)
    if format_use_log:
        g.set_yscale("log")
        g.set_yticks([10 ** x for x in range(6)])
        # g.set_xticklabels(['0','a','b','c','d','e'])
    g.set_xticklabels(g.get_xticklabels(), rotation=35)
    plt.tight_layout()
    #plt.savefig(f'./figures/{filename}.pdf')
    #plt.savefig(f'./figures/{filename}.png')
    plt.show()

def plot_speed_and_acceleration(targets, format_use_log=False, chunk_size=128):
    # Based on test_friday from GANtestbe
    # So this is a test to get the velocity using real data
    # I think it works ok, the plot and the numbers look like they match
    # This is good

    n_class = 11
    plot_range = range(3500, 4000)
    y = torch.concat(targets, dim=-1)[None,...]   # concat over frames
    #n_examples = y.shape[0]
    n_examples = len(targets)

    y_velocity = torch.diff(y, dim=-1)    # Optional , multiply y_velocity * 10 to get meters/sec
    y_acceleration = torch.diff(y_velocity, dim=-1)
    y_speed = torch.linalg.vector_norm(y_velocity, ord=2, dim=-3)  # Magnitude over channels

    plots.plot_labels(y, title='y')
    plots.plot_labels_cross_sections(y, title='y')

    # Spherical coords
    y_spherical = np.zeros_like(y)
    for cc in range(y_spherical.shape[1]):
        tmp = utils.vecs2dirs(y[:, cc, :].squeeze().transpose(1, 0), include_r=True, use_elevation=True)
        y_spherical[:, cc, ::] = tmp.transpose(1, 0)
    y_spherical = torch.from_numpy(y_spherical)
    plots.plot_labels(y_spherical, plot_cartesian=False)

    fig = plt.figure()
    n_class = 2  # Manually, select an active class
    n_coord = 0  # Manually, select x,y or z
    plt.plot(y[n_coord, n_class, :], '-r')
    plt.plot(y_velocity[n_coord, n_class, :], '--b')
    plt.plot(y_speed[n_class, :], '-g')
    plt.show()

    # Zoom in:
    fig = plt.figure()
    plt.plot(y[0, n_class, plot_range], '-r')
    plt.plot(y_velocity[0, n_class, plot_range], '--b')
    # plt.plot(y_acceleration[0, n_class, :], ':k')
    plt.plot(y_speed[n_class, plot_range], '-g')
    plt.show()

    # And histograms
    # Another option is to not do it by class
    n_class = range(0, 13)  # This is we want all classes
    y_speed_truncated = y_speed[n_class, :]
    y_speed_truncated = y_speed_truncated[y_speed_truncated > 0.001]
    y_speed_truncated[y_speed_truncated > 1] = 1

    fig = plt.figure()
    plt.hist(y_speed_truncated.detach().cpu().numpy(), density=True, bins=30)
    # plt.violinplot(y_speed, vert=False, showextrema=False, showmedians=True)
    ax = plt.gca()
    # ax.set_xlim([0.0, 0.1])
    ax.set_title('Speed')
    plt.show()

    # TODO This is not ready
    y_acceleration_mag = torch.linalg.vector_norm(y_acceleration, ord=2, dim=-3)
    n_class = range(0, 13)  # This is we want all classes
    y_acceleration_truncated = y_acceleration_mag[n_class, :]
    y_acceleration_truncated = y_acceleration_truncated[y_acceleration_truncated > 0.001]
    y_acceleration_truncated[y_acceleration_truncated > 1] = 1

    fig = plt.figure()
    plt.hist(y_acceleration_truncated.detach().cpu().numpy(), density=True, bins=30)
    # plt.violinplot(y_speed, vert=False, showextrema=False, showmedians=True)
    ax = plt.gca()
    # ax.set_xlim([0.0, 0.1])
    ax.set_title('Accelreation')
    plt.show()


def get_data(config):
    dataset_train = DCASE_SELD_Dataset(directory_root=config.dataset_root[0],
                                       list_dataset=config.dataset_list_train[0],
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       labels_backend='sony',
                                       return_fname=False)

    dataset_valid = DCASE_SELD_Dataset(directory_root=config.dataset_root_valid,
                                       list_dataset=config.dataset_list_valid,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       labels_backend='sony',
                                       return_fname=False)

    if len(config.dataset_list_train) > 1:
        dataset_synth = DCASE_SELD_Dataset(directory_root=config.dataset_root[1],
                                           list_dataset=config.dataset_list_train[1],
                                           chunk_size=config.dataset_chunk_size,
                                           chunk_mode='full',
                                           trim_wavs=config.dataset_trim_wavs,
                                           multi_track=config.dataset_multi_track,
                                           num_classes=config.unique_classes,
                                           labels_backend='sony',
                                           return_fname=False)

    return dataset_train, dataset_valid, dataset_synth



def main():
    config = get_parameters()
    dataset_train, dataset_valid, dataset_synth = get_data(config)

    targets_train = []
    print('Reading files train..... ')
    for _, tmp in tqdm(dataset_train):
        targets_train.append(tmp)
    targets_train_flat = torch.concat(targets_train, dim=-1)   # concat along frames, in case files have different length
    targets_train_flat = targets_train_flat[None, ...]

    targets_valid = []
    print('Reading files validation..... ')
    for _, tmp in tqdm(dataset_valid):
        targets_valid.append(tmp)
    targets_valid_flat = torch.concat(targets_valid, dim=-1)   # concat along frames, in case files have different length
    targets_valid_flat = targets_valid_flat[None, ...]

    targets_synth = []
    print('Reading files synthetic..... ')
    for _, tmp in tqdm(dataset_synth):
        targets_synth.append(tmp)
#    targets_synth_flat = torch.concat(targets_synth, dim=-1)   # concat along frames, in case files have different length
#    targets_synth_flat = targets_synth_flat[None, ...]

    print('Start analysis')
    dataset_train, dataset_valid, dataset_synth = None, None, None  # Free memory
    #####plot_histograms_bivariate_azi_ele(targets_valid_flat)

    #####plot_histograms_active_per_class(targets_train, targets_valid, targets_synth, detection_threshold=0.5)
    #####plt.savefig('./figures/figure_01_active_per_class.pdf')
    #####plt.savefig('./figures/figure_01_active_per_class.png')
    print('End of analysis')

    plot_histograms_polyphony(targets_train, detection_threshold=0.5, format_use_log=False, chunk_size=128)
    plt.savefig('./figures/figure_01_active_per_class.pdf')
    plt.savefig('./figures/figure_01_active_per_class.png')
    print('End of analysis')

    plot_speed_and_acceleration(targets_train)
    plt.savefig('./figures/figure_01_active_per_class.pdf')
    plt.savefig('./figures/figure_01_active_per_class.png')
    print('End of analysis')

if __name__ == '__main__':
    """ 
    Run it like this  for dcase2022
    -c
    ./configs/run_debug.yaml
    --dataset_trim_wavs
    -1
    --dataset_backend
    baseline
    --dataset_root
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    /m/triton/scratch/work/falconr1/sony/data_dcase2022_sim
    --dataset_list_train
    dcase2022_devtrain_all.txt
    dcase2022_sim_all.txt
    --dataset_root_valid
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    --dataset_list_valid
    dcase2022_devtest_all.txt
    
    Like this for dcase2021
        -c
    ./configs/run_debug.yaml
    --dataset_trim_wavs
    -1
    --dataset_root
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    /m/triton/scratch/work/falconr1/sony/data_dcase2022_sim
    --dataset_list_train
    dcase2022_devtrain_all.txt
    dcase2022_sim_all.txt
    --dataset_root_valid
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    --dataset_list_valid
    dcase2022_devtest_all.txt
    
    -c
    ./configs/run_debug.yaml
    --dataset_trim_wavs
    -1
    --dataset_root
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    --dataset_list_train
    dcase2022_devtrain_debug.txt
    dcase2022_devtrain_debug.txt
    --dataset_root_valid
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    --dataset_list_valid
    dcase2022_devtest_debug.txt
    """
    main()
