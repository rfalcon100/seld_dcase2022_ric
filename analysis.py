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


def plot_histograms_positionsOLD(targets, threshold=0.5):
    """ Here I plot the 2d histograms of azimuths and elevations"""

    raise DeprecationWarning('I was using this to test plots, but my I use my plot from plots plots.')
    assert len(targets.shape) == 4, 'Should be [batch, 3, n_classes, frames]'

    # Find active frames (where at least 1 class has norm > threshold)
    norma = torch.linalg.vector_norm(targets, ord=2, dim=-3)  # [batch, n_classes, frames]
    mask_active_per_frames = torch.any(norma > threshold, dim=-2)  # [batch, frames]
    active_targets = targets[..., mask_active_per_frames[-1]]  # [batch, 3, n_classes, frames]

    print(f'All frames shape: {targets.shape}')
    print(f'Active frames with at least 1 active class shape: {active_targets.shape}')

    # Optional, look at the trayectories:
    plots.plot_labels_cross_sections(active_targets[0, ..., 0:200000])

    # Reshape into points (or DOAs), and find active frames of any class
    active_targets = active_targets.permute((0,2,3,1)).reshape((-1, 3)).contiguous()  # [batch * n_classes * frames, 3]
    norma = torch.linalg.vector_norm(active_targets, ord=2, dim=-1)
    mask_active_per_frames = norma > threshold
    active_targets_all_classes = active_targets[mask_active_per_frames, ...]
    print(f'Active frames of all classes: {active_targets.shape}')

    # To spherical coords
    sns.color_palette("magma", as_cmap=True)
    all_new_points_sph = utils.vecs2dirs(active_targets_all_classes.squeeze(), use_elevation=True, include_r=True)
    all_new_points_sph = np.round(all_new_points_sph, decimals=6)  # This is to fix rounding errors with R

    if all_new_points_sph.shape[0] > 1e6:
        warnings.warn('WARNING: There are more then 1M points. Evaluating histograms might take a while.')

    # TODO: GOOD-plot this is like the final boss, but a bit more lame
    print('Beggingin plot')
    df = pd.DataFrame(all_new_points_sph, columns=['azimuth', 'Elevation', 'R'])
    sns.displot(df, x="azimuth", y="Elevation", cbar=True, cmap='magma', kind='hist')
    plt.show()

    # Plot with marginals   # TODO: Mid-plot this is like the final boss, but a bit more lame
    print('Beggingin plot with marginals')
    g = sns.JointGrid(data=df, x="azimuth", y="Elevation")
    # g.plot_joint(sns.kdeplot, cmap='magma', kde=True, fill=True)  # Change kdeplot to histplot for bars
    g.plot_joint(plt.hexbin, cmap='magma', gridsize=15, bins='log',
                 extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2))  # Change kdeplot to histplot for bars
    g.plot_marginals(sns.histplot, bins=15)
    plt.show()

    # Plot KDE  # TODO: Mid-plot this looks kinda nice, but it is too slow
    print('Beggingin KDE with colorbar')
    warnings.warn('This can be very slow')
    kdeplot = sns.jointplot(data=df[0:50000], x="azimuth", y="Elevation", kind="kde", xlim=[0, 2*np.pi], ylim=[-np.pi/2, np.pi/2], cbar=True,
                            cmap='magma', fill=True, cbar_kws={'label': ''})
    plt.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = kdeplot.ax_joint.get_position()
    pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    # get the current colorbar ticks
    cbar_ticks = kdeplot.fig.axes[-1].get_yticks()
    # get the maximum value of the colorbar
    _, cbar_max = kdeplot.fig.axes[-1].get_ylim()
    # change the labels (not the ticks themselves) to a percentage
    kdeplot.fig.axes[-1].set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])
    plt.tight_layout()
    plt.show()


    #=======================================================================
    # FIANLLY: This is linear count  # TODO: Mid-plot this is like the final boss, but linear count
    print('Beggingin plot with marginals')
    g = sns.JointGrid(data=df, x="azimuth", y="Elevation")
    # g.plot_joint(sns.kdeplot, cmap='magma', kde=True, fill=True)  # Change kdeplot to histplot for bars
    aa = g.plot_joint(sns.histplot, cmap='magma', binrange=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2)),
                      cbar=True, cbar_kws={'label': 'Count'})  # Change kdeplot to histplot for bars
    aa = g.plot_joint(plt.hexbin, cmap='magma', gridsize=15,
                      extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2))  # Change kdeplot to histplot for bars
    g.plot_marginals(sns.histplot, bins=15, element="step", color="#03012d")

    plt.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    # get the current colorbar ticks
    cbar_ticks = g.fig.axes[-1].get_yticks()
    # get the maximum value of the colorbar
    _, cbar_max = g.fig.axes[-1].get_ylim()
    # change the labels (not the ticks themselves) to a percentage
    # g.fig.axes[-1].set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])
    plt.tight_layout()
    plt.show()


    #=======================================================================
    # ALMOST: This is log count (not sure if the bar is ok)  # TODO: Bad-plot this is like the final boss, but wrong, we can delete it
    # The colors of the cbar do not match, because histplot is linear, and hexbin is log
    #
    print('Beggingin plot with marginals')
    g = sns.JointGrid(data=df, x="azimuth", y="Elevation")
    # g.plot_joint(sns.kdeplot, cmap='magma', kde=True, fill=True)  # Change kdeplot to histplot for bars
    aa = g.plot_joint(sns.histplot, cmap='magma', binrange=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2)),
                      cbar=True, cbar_kws={'label': 'Log10(n)'})  # Change kdeplot to histplot for bars
    aa = g.plot_joint(plt.hexbin, cmap='magma', gridsize=15, bins='log',
                      extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2))  # Change kdeplot to histplot for bars
    g.plot_marginals(sns.histplot, bins=15, element="step", color="#03012d")

    plt.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    # get the current colorbar ticks
    cbar_ticks = g.fig.axes[-1].get_yticks()
    # get the maximum value of the colorbar
    _, cbar_max = g.fig.axes[-1].get_ylim()
    # change the labels (not the ticks themselves) to a percentage
    g.fig.axes[-1].set_yticklabels([f'{np.log10(t + 1e-8):.2f} %' for t in cbar_ticks])
    plt.tight_layout()
    plt.show()

    #=======================================================================
    # Fuck this, lets do it manually
    counts = np.histogram2d(x=df['azimuth'], y=df['Elevation'], bins=15, range=[(0, 2 * np.pi), (-np.pi / 2, np.pi / 2)])
    log_counts = np.log10(counts[0] + 1e-8)
    ticks = np.linspace(counts[0].min(), counts[0].max(), 10, endpoint=True)
    ticks = 10 ** np.ceil(np.linspace(log_counts[0].min(), log_counts[0].max(), 10, endpoint=True))

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    ax = axs[0]
    hb = ax.hexbin(x=df['azimuth'], y=df['Elevation'], gridsize=15, cmap='inferno')
    ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    ax.set_title("Hexagon binning")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')

    ax = axs[1]
    hb = ax.hexbin(x=df['azimuth'], y=df['Elevation'], gridsize=15, bins='log', cmap='inferno')
    ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    ax.set_title("With a log color scale")
    cb = fig.colorbar(hb, ax=ax, ticks=ticks)
    cb.set_label('log10(N)')
    plt.show()

    # TODO: Good plot
    #=======================================================================
    # FINAL BOSS  GOOD
    # This looks OK to me, just double check the ticks in the color bar and elevation
    # AAAHA the main plot is deformed, the hexes look bad
    # =======================================================================
    # FINAL BOSS
    # This looks OK to me, just double check the ticks in the color bar and elevation
    print('Beggingin plot with marginals')
    g = sns.JointGrid(data=df, x="azimuth", y="Elevation")
    # g.plot_joint(sns.kdeplot, cmap='magma', kde=True, fill=True)  # Change kdeplot to histplot for bars
    # aa = g.plot_joint(sns.histplot, cmap='magma', binrange=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2)),
    #                  cbar=True, cbar_kws={'label': 'Log10(n)'})  # Change kdeplot to histplot for bars
    #aa = g.plot_joint(plt.hexbin, cmap='magma', gridsize=15, bins='log',
    #                  extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2))  # Change kdeplot to histplot for bars
    g.plot_marginals(sns.histplot, bins=15, element="step", color="#03012d")
    # g.fig.axes[-1].set_yticklabels(ticks)
    ax = g.fig.axes[0]
    hb = ax.hexbin(x=df['azimuth'], y=df['Elevation'], gridsize=15, bins='log', cmap='magma', extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2))
    ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    # ax.set_title("With a log color scale")
    cb = fig.colorbar(hb, ax=ax, ticks=ticks)
    cb.set_label('log10(N)')
    g.fig.axes[-1] = cb  # doe snot work
    plt.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    # get the current colorbar ticks
    cbar_ticks = g.fig.axes[-1].get_yticks()
    # get the maximum value of the colorbar
    _, cbar_max = g.fig.axes[-1].get_ylim()
    # change the labels (not the ticks themselves) to a percentage
    # g.fig.axes[-1].set_yticklabels([f'{np.log10(t + 1e-8):.2f} %' for t in cbar_ticks])
    plt.tight_layout()
    plt.show()


def plot_histograms_positions(targets, threshold=0.5):
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

def get_data(config):
    dataset_valid = DCASE_SELD_Dataset(directory_root=config.dataset_root_valid,
                                       list_dataset=config.dataset_list_valid,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       labels_backend='sony',
                                       return_fname=False)

    return dataset_valid

def plot_distribution_azi_eleOLD(points: Union[torch.Tensor, np.ndarray], type='hex',
                              log_scale=True, bins=15, gridsize=15, kde_fill=True, kde_levels=8):
    """ This plots the bivariate distribution for azimuth and elevation.
    Options:
        type: ['hex', 'kde', 'hist']
        log_scale: [True, False]  --> log scale for the count (not the axis)
        kde_fill: [True, False]   --> only for kde
        kde_levels: int  ---> only for kde
        bins = 15
        gridsize = 15
    """
    assert len(points.shape) == 2 and points.shape[-1] == 3, 'ERROR: Wrong shape for input points, should be [n_points, 3]'
    points_sph = utils.vecs2dirs(points.squeeze(), use_elevation=True, include_r=True)
    points_sph = np.round(points_sph, decimals=6)  # This is to fix rounding errors with R
    if points_sph.shape[0] > 1e6:
        warnings.warn('WARNING: There are more then 1M points. Evaluating histograms might take a while.')
    df = pd.DataFrame(points_sph, columns=['azimuth', 'Elevation', 'R'])
    counts = np.histogram2d(x=df['azimuth'], y=df['Elevation'], bins=bins, range=[(0, 2 * np.pi), (-np.pi / 2, np.pi / 2)])
    if log_scale:
        log_counts = np.log10(counts[0] + 1e-8)
        ticks = 10 ** np.ceil(np.linspace(log_counts[0].min(), log_counts[0].max(), 10, endpoint=True))
        cbar_label = 'log10(n)'
    else:
        ticks = np.linspace(counts[0].min(), counts[0].max(), 10, endpoint=True)
        cbar_label = 'Count'
    fig = plt.figure()
    g = sns.JointGrid(data=df, x="azimuth", y="Elevation")
    g.plot_marginals(sns.histplot, bins=bins, element="step", color="#03012d")
    ax = g.fig.axes[0]
    if type == 'hex':
        bins_scale = 'log' if log_scale else None
        hb = ax.hexbin(x=df['azimuth'], y=df['Elevation'], gridsize=gridsize, bins=bins_scale, cmap='magma', extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2))
        ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    elif type == 'hist':
        from matplotlib.colors import LogNorm
        normalizer = LogNorm(vmin=10e-1, vmax=ticks[-1]) if log_scale else None
        _, _, _, hb = ax.hist2d(x=df['azimuth'], y=df['Elevation'], bins=gridsize, cmap='magma', range=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2)), norm=normalizer)
        #g.plot_joint(sns.histplot, cmap='magma', binrange=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2)), norm=normalizer,
        #             cbar=True, cbar_kws={'label': cbar_label})
        ax.grid()
    elif type == 'kde':
        if log_scale: raise ValueError('Log scale is not supported when using KDE')
        hb = g.plot_joint(sns.kdeplot, cmap='magma', levels=kde_levels, cbar=True, fill=kde_fill, gridsize=gridsize)
        ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    if type == 'hex' or type == 'hist':
        cb = fig.colorbar(hb, ax=ax, ticks=ticks)
        cb.set_label(cbar_label)
        g.fig.axes[-1] = cb  # doe snot work
    plt.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    if type == 'kde':
        cbar_ticks = g.fig.axes[-1].get_yticks()
        _, cbar_max = g.fig.axes[-1].get_ylim()
        g.fig.axes[-1].set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])
    plt.tight_layout()
    plt.title('yolo dude')
    plt.show()

def main():
    config = get_parameters()
    dataset = get_data(config)

    targets = []
    print('Reading files ..... ')
    for _, tmp in tqdm(dataset):
        targets.append(tmp)
    targets = torch.concat(targets, dim=-1)   # concat along frames, in case files have different length
    targets = targets[None, ...]

    print('Start analysis')
    dataset = None  # Free memory
    plot_histograms_positions(targets)
    print('End of analysis')

if __name__ == '__main__':
    """ 
    Run it like this
    -c
    ./configs/run_debug.yaml
    --dataset_trim_wavs
    5
    --dataset_root_valid
    /m/triton/scratch/work/falconr1/sony/data_dcase2022
    --dataset_list_valid
    dcase2022_devtrain_all.txt
    """
    main()
