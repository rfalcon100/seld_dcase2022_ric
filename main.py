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
from easydict import EasyDict
from datetime import datetime
from itertools import islice
from typing import List

from dataset.dcase_dataset import DCASE_SELD_Dataset, InfiniteDataLoader, _get_padders
from evaluation.dcase2022_metrics import cls_compute_seld_results
from evaluation.evaluation_dcase2022 import write_output_format_file, get_accdoa_labels, get_multi_accdoa_labels, determine_similar_location, all_seld_eval
from solver import Solver
from feature import my_feature
#from parameters import get_parameters
import utils
import plots


def get_parameters():
    """ Deprecated. We now use the parameters.py"""
    params = {
        'exp_name': 'debug',  #baseline_dcase2021
        'seed_mode': 'balanced',
        'mode': 'train',
        'num_iters': 10000,  # debug 10000
        'batch_size': 1,     # debug 1
        'num_workers': 5,
        'print_every': 50,
        'logging_interval': 500,  # debug 100 or 50
        'lr': 1e-4,
        'lr_decay_rate': 0.9,
        'lr_patience_times': 3,
        'lr_min': 1e-4,  # should be 1e-7 when using scheduler
        'model': 'crnn10',
        'model_normalization': 'batchnorm',
        'model_loss_fn': 'mse',
        'input_shape': [7,96,128],
        'output_shape': [3,12,128],
        'logging_dir': './logging',
        'dataset_root': '/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3',
        'dataset_list_train': 'dcase2021t3_foa_overfit.txt',
        'dataset_list_valid': 'dcase2021t3_foa_overfit.txt',
        'dataset_trim_wavs': -1,
        'dataset_chunk_size': int(24000 * 1.27),
        'dataset_chunk_mode': 'random',
        'dataset_multi_track': False,
        'thresh_unify': 15,
    }

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
        params['experiment_description'] = f'{params["exp_name"]}_' \
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
    with open(os.path.join(params['logging_dir'] , 'params.yaml'), 'w') as f:
        yaml.dump(params, f, default_flow_style=None)
    if not os.path.exists(params['directory_output_results']):
        os.mkdir(params['directory_output_results'])

    return EasyDict(params)


def get_dataset(config):
    dataloader_train = None

    dataset_train = DCASE_SELD_Dataset(directory_root=config.dataset_root,
                                       list_dataset=config.dataset_list_train,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode=config.dataset_chunk_mode,
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       return_fname=False)
    dataloader_train = InfiniteDataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_workers,
                                          shuffle=True, drop_last=True)

    dataset_valid = DCASE_SELD_Dataset(directory_root=config.dataset_root,
                                       list_dataset=config.dataset_list_valid,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       return_fname=True)

    return dataloader_train, dataset_valid


def main():
    # Get config
    config = get_parameters()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    utils.seed_everything(seed=12345, mode=config.seed_mode)

    # Logging configuration
    writer = SummaryWriter(config.logging_dir)

    # Data
    dataloader_train, dataset_valid = get_dataset(config)

    # Solver
    solver = Solver(config=config, tensorboard_writer=writer)

    spectrogram_transform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=24000,
                                             n_fft=512,
                                             hop_length=240,
                                             n_mels=96),
        torchaudio.transforms.AmplitudeToDB()).to(device)

    spectrogram_transform = nn.Sequential(
        torchaudio.transforms.Spectrogram(n_fft=512,
                                         hop_length=240),
        torchaudio.transforms.AmplitudeToDB()).to(device)

    spectrogram_transform = my_feature().to(device)  # mag STFT with intensity vectors

    # Initial loss:
    x, target = dataloader_train.dataset[0]
    x, target = spectrogram_transform(x.unsqueeze(0).to(device)), target.to(device)
    solver.predictor.eval()
    out = solver.predictor(x)
    loss = solver.loss_fns[solver.loss_names[0]](out, target[None, ...])
    print('Initial loss = {:.6f}'.format(loss.item()))

    # Monitoring variables
    train_loss, val_loss, seld_metrics = 0, 0, None
    start_time = time.time()

    if config.mode == 'train':
        print('>>>>>>>> Training START  <<<<<<<<<<<<')

        iter_idx = 0
        for data in islice(dataloader_train, config.num_iters + 1):
            start_step_time = time.time()
            train_loss = train_iteration(config, data, iter_idx=iter_idx, start_time=start_time, start_time_step=start_step_time,
                                         device=device, features_transform=spectrogram_transform,
                                         solver=solver, writer=writer)

            if iter_idx % config.logging_interval == 0 and iter_idx > 0:
                seld_metrics, val_loss = eval_iteration(config, dataset=dataset_valid, iter_idx=iter_idx,
                                                        device=device, features_transform=spectrogram_transform,
                                                        solver=solver, writer=writer,
                                                        dcase_output_folder=config['directory_output_results'])
                print(
                    'iteration: {}/{}, time: {:0.2f}, '
                    'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                    'ER/F/LE/LR/SELD: {}, '.format(
                        iter_idx, config.num_iters, curr_time,
                        train_loss, val_loss,
                        '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(*seld_metrics[0:5]),
                    ))

            # Schedulers
            solver.lr_step(seld_metrics[4] if seld_metrics is not None else 0, step=iter_idx)  # LRstep scheduler based on validation SELD score

            curr_time = time.time() - start_time
            iter_idx += 1
        print('>>>>>>>> Training Finished  <<<<<<<<<<<<')

        print('Classwise results on validation data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        seld_metrics_class_wise = seld_metrics[5]
        for cls_cnt in range(config['unique_classes']):
            print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt, seld_metrics_class_wise[0][cls_cnt],
                                                                           seld_metrics_class_wise[1][cls_cnt],
                                                                           seld_metrics_class_wise[2][cls_cnt],
                                                                           seld_metrics_class_wise[3][cls_cnt],
                                                                           seld_metrics_class_wise[4][cls_cnt]))

    elif config.mode == 'eval':
        raise NotImplementedError


def train_iteration(config, data, iter_idx, start_time, start_time_step, device, features_transform: nn.Sequential, solver, writer):
    # Training iteration
    x, target = data
    x, target = x.to(device), target.to(device)
    x = features_transform(x)
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
        writer.add_scalar('Losses/train', train_loss.item(), step)

        # Learning rates
        lr = solver.get_lrs()
        writer.add_scalar('Lr/gen', lr, step)

        # Grad norm
        grad_norm_model = solver.get_grad_norm()
        writer.add_scalar('grad_norm/disc', grad_norm_model, step)

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
            fig = plots.plot_labels(fixed_output_sph, savefig=False, plot_cartesian=False)
            writer.add_figure('fixed_output/train', fig, iter_idx)
            fig = plots.plot_labels(fixed_label_sph, savefig=False, plot_cartesian=False)
            writer.add_figure('fixed_label/train', fig, None)
            fig = plots.plot_labels(fixed_error_sph, savefig=False, plot_cartesian=False)
            writer.add_figure('fixed_error/train', fig, iter_idx)

    return train_loss.item()


def eval_iteration(config, dataset, iter_idx, solver, features_transform, dcase_output_folder, device, writer):
    # Adapted from the official baseline
    nb_test_batches, test_loss = 0, 0.
    model = solver.predictor
    model.eval()
    file_cnt = 0

    print(f'Evaluation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for ctr, (audio, target, fname) in enumerate(dataset):
            # load one batch of data
            audio, target = audio.to(device), target.to(device)
            duration = dataset.durations[fname]
            print(f'Evaluating file {ctr}/{len(dataset)}: {fname}')
            print(f'Audio shape: {audio.shape}')
            print(f'Target shape: {target.shape}')

            warnings.warn('WARNING: Hard coded chunk size for evaluation')
            audio_padding, labels_padding = _get_padders(chunk_size_seconds=1.27,
                                                         duration_seconds=math.floor(duration),
                                                         overlap=1,
                                                         audio_fs=dataset._fs[fname],
                                                         labels_fs=100)

            # Split each wav into chunks and process them
            audio = audio_padding['padder'](audio)
            audio_chunks = audio.unfold(dimension=1, size=audio_padding['chunk_size'],
                                        step=audio_padding['hop_size']).permute((1, 0, 2))
            labels = labels_padding['padder'](target)
            labels_chunks = labels.unfold(dimension=-1, size=labels_padding['chunk_size'],
                                          step=labels_padding['hop_size']).permute((2, 0, 1, 3))

            full_output = []
            full_loss = []
            tmp = torch.utils.data.TensorDataset(audio_chunks, labels_chunks)
            loader = DataLoader(tmp, batch_size=1, shuffle=False, drop_last=False)  # Loader per wav to get batches
            for ctr, (audio, labels) in enumerate(loader):
                audio = features_transform(audio)
                output = model(audio)
                loss = solver.loss_fns[solver.loss_names[0]](output, labels)
                full_output.append(output)
                full_loss.append(loss)

            # Concatenate chunks across timesteps into final predictions
            output = torch.concat(full_output, dim=-1)

            # Apply detection threshold based on vector norm
            threshold = 0.5
            norms = torch.linalg.vector_norm(output, ord=2, dim=-3, keepdims=True)
            norms = (norms < threshold).repeat(1, output.shape[-3], 1, 1)
            output[norms] = 0.0
            loss = torch.tensor([x.item() for x in full_loss]).mean()

            # Useful fo debug:
            #output.detach().cpu().numpy()[0, 0]
            #target.detach().cpu().numpy()[0]

            # Downsample over frames:
            output = nn.functional.interpolate(output, scale_factor=(1, 0.1), mode='area')

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
            write_output_format_file(output_file, output_dict, use_cartesian=False)

            test_loss += loss.item()
            nb_test_batches += 1

        test_loss /= nb_test_batches

    all_test_metric = all_seld_eval(config, directory_root=dataset.directory_root, fnames=dataset._fnames, pred_directory=dcase_output_folder)

    if writer is not None:
        writer.add_scalar('Losses/valid', test_loss, iter_idx)
        writer.add_scalar('Metrics/ER', all_test_metric[0], iter_idx)
        writer.add_scalar('Metrics/F', all_test_metric[1], iter_idx)
        writer.add_scalar('Metrics/LE', all_test_metric[2], iter_idx)
        writer.add_scalar('Metrics/LR', all_test_metric[3], iter_idx)
        writer.add_scalar('Metrics/SELD', all_test_metric[4], iter_idx)

    return all_test_metric, test_loss

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
