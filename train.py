import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
import os, shutil, math
import yaml
from easydict import EasyDict
from datetime import datetime
from itertools import islice

from dataset.dcase_dataset import DCASE_SELD_Dataset, InfiniteDataLoader
from models.discriminators import DiscriminatorBasic, DiscriminatorBasicThreshold, DiscriminatorBasicSN, DiscriminatorModularThreshold
from models.generator import CRNN10
from solver import Solver
#from parameters import get_parameters
import utils
import plots


def get_parameters():
    """ Deprecated. We now use the parameters.py"""
    params = {
        'seed_mode': 'balanced',
        'mode': 'train',
        'num_iters': 500,
        'data_n': 2,
        'batch_size': 32,
        'num_workers': 4,
        'print_every': 50,
        'logging_interval': 50,
        'lr': 1e-7,
        'lr_decay_rate': 0.9,
        'lr_patience_times': 3,
        'lr_min': 1e-8,
        'model': 'crnn10',
        'model_normalization': 'batchnorm',
        'input_shape': [4,96,128],
        'output_shape': [3,12,128],
        'logging_dir': './logging',
        'dataset_root': '/m/triton/scratch/work/falconr1/sony/data_dcase2022',
        'dataset_list': 'dcase2022_devtrain_all.txt',
        'dataset_trim_wavs': 5,
        'dataset_chunk_size': int(24000 * 1.27),
        'dataset_chunk_mode': 'random',
        'dataset_multi_track': False,
    }

    params['experiment_description'] = f'{params["model"]}_' \
                                       f'{params["model_normalization"]}_' \
                                       f'_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}'
    params['logging_dir'] = f'{params["logging_dir"]}/{params["experiment_description"]}'

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

    return EasyDict(params)


def get_dataset(config):
    dataloader_train = None
    dataloader_valid = None

    dataset = DCASE_SELD_Dataset(directory_root=config.dataset_root,
                                 list_dataset=config.dataset_list,
                                 chunk_size=config.dataset_chunk_size,
                                 chunk_mode=config.dataset_chunk_mode,
                                 trim_wavs=config.dataset_trim_wavs,
                                 multi_track=config.dataset_multitrack,
                                 return_fname=True)


    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [math.floor(len(dataset) * 0.8),
                                                                           math.ceil(len(dataset) * 0.2)])
    dataloader_train = InfiniteDataLoader(dataset_train, batch_size=config.batch_size, num_workers=confignum_workers,
                                          shuffle=True, drop_last=True)
    dataloader_valid = InfiniteDataLoader(dataset_valid, batch_size=config.batch_size, num_workers=confignum_workers,
                                          shuffle=True, drop_last=True)

    warnings.warn('WARNING: The validation set is not correct, as it is using the same random slicing.')
    return dataloader_train, dataloader_valid


def main():
    # Get config
    config = get_parameters()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    utils.seed_everything(seed=12345, mode=config.seed_mode)

    # Logging configuration
    writer = SummaryWriter(config.logging_dir)

    # Data
    dataloader_train, dataloader_valid = get_dataset(config)

    # Solver
    solver = Solver(config=config, dataloader_train=dataloader_train, tensorboard_writer=writer)

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
        train(config, dataloader_train=dataloader_train, device=device,
              features_transform=spectrogram_transform, solver=solver, writer=writer, rec_losses=rec_losses)
    elif config.mode == 'eval':
        raise NotImplementedError


def train(config, dataloader_train, device, features_transform: nn.Sequential, solver, writer, rec_losses):
    # Training loop
    iter_idx = 0
    for (x, target, fnames) in islice(dataloader_train, config.num_iters):
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

    fig = plots.plot_losses(np.asarray(rec_losses), None)
    if writer is not None:
        writer.add_figure('2losses', fig, None)

    print('>>>>>>>> Finished <<<<<<<<<<<<')


def valid(config, dataloader_valid, device, features_transform: nn.Sequential, solver, writer, rec_losses):
    raise NotImplementedError
    # Validation loop
    iter_idx = 0
    for (x, target, fnames) in islice(dataloader_train, config.num_iters):
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

    fig = plots.plot_losses(np.asarray(rec_losses), None)
    if writer is not None:
        writer.add_figure('2losses', fig, None)

    print('>>>>>>>> Finished <<<<<<<<<<<<')


if __name__ == '__main__':
    main()
