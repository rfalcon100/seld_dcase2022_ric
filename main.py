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
from easydict import EasyDict
from datetime import datetime
from itertools import islice

from dataset.dcase_dataset import DCASE_SELD_Dataset, InfiniteDataLoader, _get_padders
from evaluation.dcase2022_metrics import cls_compute_seld_results
from evaluation.evaluation_dcase2022 import write_output_format_file, get_accdoa_labels, get_multi_accdoa_labels, determine_similar_location, all_seld_eval
from solver import Solver
#from parameters import get_parameters
import utils
import plots


def get_parameters():
    """ Deprecated. We now use the parameters.py"""
    params = {
        'exp_name': 'debug',
        'seed_mode': 'balanced',
        'mode': 'train',
        'num_iters': 5000,
        'batch_size': 2,
        'num_workers': 4,
        'print_every': 50,
        'logging_interval': 5,
        'lr': 1e-5,
        'lr_decay_rate': 0.9,
        'lr_patience_times': 3,
        'lr_min': 1e-8,
        'model': 'crnn10',
        'model_normalization': 'batchnorm',
        'input_shape': [4,96,128],
        'output_shape': [3,12,128],
        'logging_dir': './logging',
        'dataset_root': '/m/triton/scratch/work/falconr1/sony/data_dcase2022',
        'dataset_list_train': 'dcase2022_devtrain_debug.txt',
        'dataset_list_valid': 'dcase2022_devtest_debug.txt',
        'dataset_trim_wavs': 5,
        'dataset_chunk_size': int(24000 * 1.27),
        'dataset_chunk_mode': 'random',
        'dataset_multi_track': False,
        'unique_classes': 12,
        'thresh_unify': 15,
    }

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

    # Save config to disk
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
                                       return_fname=False)
    dataloader_train = InfiniteDataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_workers,
                                          shuffle=True, drop_last=True)

    dataset_valid = DCASE_SELD_Dataset(directory_root=config.dataset_root,
                                       list_dataset=config.dataset_list_valid,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
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

    # Initial loss:
    x, target = dataloader_train.dataset[0]
    x, target = spectrogram_transform(x.unsqueeze(0).to(device)), target.to(device)
    solver.predictor.eval()
    out = solver.predictor(x)
    loss = solver.criterionRec(out, target)
    print('Initial loss = {:.6f}'.format(loss.item()))
    rec_losses = []

    if config.mode == 'train':
        train_iteration(config, dataloader_train=dataloader_train, dataset_valid=dataset_valid, device=device,
              features_transform=spectrogram_transform, solver=solver, writer=writer, rec_losses=rec_losses)
    elif config.mode == 'eval':
        raise NotImplementedError


def train_iteration(config, dataloader_train, dataset_valid, device, features_transform: nn.Sequential, solver, writer, rec_losses):
    # Training loop
    print('>>>>>>>> Training START  <<<<<<<<<<<<')

    iter_idx = 0
    for (x, target) in islice(dataloader_train, config.num_iters):
        print(f'iteration {iter_idx}')
        iter_idx += 1
        x, target = x.to(device), target.to(device)
        x = features_transform(x)
        solver.set_input(x, target)
        solver.train_step()

        # Output training stats
        rec_loss = solver.losses['rec']

        # Schedulers
        solver.lr_step(rec_loss.item(), step=iter_idx)  # Scheduler

        # Logging and printing
        if writer is not None:
            step = iter_idx
            writer.add_scalar('losses/rec_loss', rec_loss.item(), step)

            # Learning rates
            lr = solver.get_lrs()
            writer.add_scalar('Lr/gen', lr, step)

        if iter_idx % config.print_every == 0:
            print('[%d/%d] iters \t Loss_rec: %.4f' % (iter_idx, config.num_iters, rec_loss.item()))

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
                    tmp = utils.vecs2dirs(fixed_output[:, cc, :].squeeze().transpose(1, 0), include_r=True,
                                    use_elevation=True)
                    fixed_output_sph[:, cc, ::] = tmp.transpose(1, 0)
                    tmp = utils.vecs2dirs(fixed_label[:, cc, :].squeeze().transpose(1, 0), include_r=True,
                                    use_elevation=True)
                    fixed_label_sph[:, cc, ::] = tmp.transpose(1, 0)
                    tmp = utils.vecs2dirs(fixed_error[:, cc, :].squeeze().transpose(1, 0), include_r=True,
                                    use_elevation=True)
                    fixed_error_sph[:, cc, ::] = tmp.transpose(1, 0)

                # Plot
                fig = plots.plot_labels(fixed_output_sph, savefig=False, plot_cartesian=False)
                writer.add_figure('fixed_input/train', fig, iter_idx)
                if not iter_idx > 0:
                    fig = plots.plot_labels(fixed_label_sph, savefig=False, plot_cartesian=False)
                writer.add_figure('fixed_label/train', fig, None)
                fig = plots.plot_labels(fixed_error_sph, savefig=False, plot_cartesian=False)
                writer.add_figure('fixed_error/train', fig, iter_idx)

            # Save Losses for plotting later
            rec_losses.append(rec_loss.item())

        # Validation
        if (iter_idx % config.logging_interval == 0):
            eval_iteration(config, dataset=dataset_valid, model=solver.predictor, device=device, criterion=torch.nn.MSELoss(),
                           features_transform=features_transform, dcase_output_folder=config['directory_output_results'])

#    fig = plots.plot_losses(np.asarray(rec_losses), None)
#    if writer is not None:
#        writer.add_figure('2losses', fig, None)

    print('>>>>>>>> Training Finished  <<<<<<<<<<<<')


def eval_iteration(config, dataset, model, criterion, features_transform, dcase_output_folder, device):
    # Adapted from the official baseline

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0

    print(f'Evaluation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for audio, target, fname in dataset:
            # load one batch of data
            audio, target = torch.tensor(audio).to(device).float(), torch.tensor(target).to(device).float()
            duration = dataset.durations[fname]

            audio_padding, labels_padding = _get_padders(chunk_size_seconds=1.27,
                                                         duration_seconds=math.floor(duration),
                                                         overlap=0.5,
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
                loss = criterion(output, labels)
                full_output.append(output)
                full_loss.append(loss)

            # Concatenate chunks across timesteps into final predictions
            output = torch.concat(full_output, dim=-1)
            loss = torch.tensor([x.item() for x in full_loss]).mean()

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
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][
                                                               class_cnt + config['unique_classes']],
                                                           doa_pred[frame_cnt][
                                                               class_cnt + 2 * config['unique_classes']]])
            write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1

        test_loss /= nb_test_batches

    config.eval = 'ss'
    all_test_metric = all_seld_eval(config, directory_root=dataset.directory_root, fnames=dataset._fnames, pred_directory=dcase_output_folder)

    # Print stats
    print(
        'epoch: {}, time: {:0.2f}/{:0.2f}, '
        # 'train_loss: {:0.2f}, val_loss: {:0.2f}, '
        'train_loss: {:0.4f}, val_loss: {:0.4f}, '
        'ER/F/LE/LR/SELD: {}, '
        'best_val_epoch: {} {}'.format(
            epoch_cnt, train_time, val_time,
            train_loss, val_loss,
            '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
            best_val_epoch,
            '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR, best_seld_scr))
    )
    
    return all_test_metric, test_loss


if __name__ == '__main__':
    main()
