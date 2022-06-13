'''
The file containd utility functions.

Author: Ricardo Falcon
2021
'''

import os
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import random
import warnings
import datetime
from typing import Iterable, Tuple, TypeVar, Callable, Any, List, Union
from matplotlib.colors import to_rgb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import spaudiopy as spa


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

    Ref: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            # warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]  # wrong code
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                # self.after_scheduler.step(metrics, None)  # not necessary epoch parameter
                self.after_scheduler.step(metrics)
            else:
                # self.after_scheduler.step(metrics, epoch - self.total_epoch)  # not necessary epoch parameter
                self.after_scheduler.step(metrics)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def validate_audio(audio, message=''):
    """ Checks if the audio is valid by:
        -- Not having any infinite values
        -- Not having NaNs
        -- Not being silence (only 0s)
        -- All values in the range [-1, 1]
    """
    if isinstance(audio, np.ndarray):
        tmp = torch.from_numpy(audio)
    else:
        tmp = audio

    check = torch.all(torch.isfinite(tmp))
    check = check and torch.all(~torch.isnan(tmp))
    check = check and torch.any(torch.logical_or(tmp > 0, tmp < 0))
    check = check and torch.all(torch.logical_and(tmp <= 1, tmp >= -1))

    #import matplotlib
    #matplotlib.use('TkAgg')
    #plot_waveform(tmp.transpose(1, 0), sample_rate=24000)

    if message != '':
        print('>>>>>>>> ' + message + f'{check}')
    return check


def seed_everything(seed=12345, mode='balanced'):
    # ULTIMATE random seeding for either full reproducibility or a balanced reproducibility/performance.
    # In general, some operations in cuda are non deterministic to make them faster, but this can leave to
    # small differences in several runs.
    #
    # So as of 21.10.2021, I think that the best way is to use the balanced approach during exploration
    # and research, and then use the full reproducibility to get the final results (and possibly share code)
    #
    # References:
    # https://pytorch.org/docs/stable/notes/randomness.html
    #
    # Args:
    #   -- seed = Random seed
    #   -- mode {'balanced', 'deterministic'}

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode == 'balanced':
        torch.backends.cudnn.deterministic = False   # if set as true, dilated convs are really slow
        torch.backends.cudnn.benchmark = True  # True -> better performance, # False -> reproducibility
    elif mode == 'deterministic':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Throws error:
        # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)`
        # or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS
        # and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable
        # before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
        # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

        # torch.use_deterministic_algorithms(True)



def get_rotation_matrix(rotation_phi, rotation_theta, rotation_psi) -> torch.Tensor:
    """ Returns a full 3d Rotation matrix, for the given rotation angles phi, thetha, psi
    that are rotations over:
    phi --> rotation over x axis (roll)
    theta --> rotation over y axis (pitch)
    psi --> rotation over z axis (yaw)

    Note: This rotates cartesian coordinates.

    Reference:
    See: [1] M. Kronlachner, 'Spatial transformations for the alteration of ambisonic recordings.'
    Equation 2.15
    """
    roll = torch.tensor([[1, 0, 0],
                         [0, np.cos(rotation_phi), -np.sin(rotation_phi)],
                         [0, np.sin(rotation_phi), np.cos(rotation_phi)]])
    pitch = torch.tensor([[np.cos(rotation_theta), 0, np.sin(rotation_theta)],
                          [0, 1, 0],
                          [-np.sin(rotation_theta), 0, np.cos(rotation_theta)]])
    yaw = torch.tensor([[np.cos(rotation_psi), -np.sin(rotation_psi), 0],
                        [np.sin(rotation_psi), np.cos(rotation_psi), 0],
                        [0, 0, 1]])
    R = torch.matmul(torch.matmul(roll, pitch), yaw)
    return R


def colat2ele(colat: Union[float, torch.Tensor]) -> torch.Tensor:
    """Transforms colatitude to elevation (latitude). In radians.

    The polar angle on a Sphere measured from the North Pole instead of the equator.
    The angle $\phi$ in Spherical Coordinates is the Colatitude.
    It is related to the Latitude $\delta$ by $\phi=90^\circ-\delta$.
    """
    ele = math.pi / 2 - colat
    return ele


def ele2colat(ele: Union[float, torch.Tensor]) -> torch.Tensor:
    """Transforms colatitude to elevation (latitude). In radians.

    The polar angle on a Sphere measured from the North Pole instead of the equator.
    The angle $\phi$ in Spherical Coordinates is the Colatitude.
    It is related to the Latitude $\delta$ by $\phi=90^\circ-\delta$.
    """
    colat = math.pi / 2 - ele
    return colat


def vecs2dirs(vecs, positive_azi=True, include_r=False, use_elevation=False):
    """Helper to convert [x, y, z] to [azi, colat].
    From Spaudiopyy, but with safe case when r=0"""
    azi, colat, r = spa.utils.cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2], steady_colat=True)
    if positive_azi:
        azi = azi % (2 * np.pi)  # [-pi, pi] -> [0, 2pi)
    if use_elevation:
        colat = colat2ele(colat)
    if include_r:
        output = np.c_[azi, colat, r]
    else:
        output = np.c_[azi, colat]
    return output


def sph2unit_vec(azimuth: Union[float, torch.Tensor], elevation: Union[float, torch.Tensor]) -> torch.Tensor:
    """ Transforms spherical coordinates into a unit vector .
    Equaiton 2.1 of
    [1]M. Kronlachner, “Spatial transformations for the alteration of ambisonic recordings”.
    """
    assert torch.all(azimuth >= 0) and torch.all(
        azimuth <= 2 * np.pi), 'Azimuth should be in radians, between 0 and 2*pi'

    x = torch.cos(azimuth) * torch.cos(elevation)
    y = torch.sin(azimuth) * torch.cos(elevation)
    z = torch.sin(elevation)

    return torch.stack([x, y, z], dim=-1)


def unit_vec2sph(angle_x: Union[torch.Tensor, float],
                 angle_y: Union[torch.Tensor, float],
                 angle_z: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Transforms angles over each axis (unit vector notation) to spherical angles.
    Equation 2.2 of
    [1]M. Kronlachner, “Spatial transformations for the alteration of ambisonic recordings”.
    """

    azimuth = torch.arctan(angle_y / angle_x)
    elevation = torch.arctan(angle_z / (torch.sqrt(angle_x ** 2 + angle_y ** 2)))

    return azimuth, elevation

def cart2sph(x, y, z):
    azi = np.arctan2(y, x) / np.pi * 180
    ele = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) / np.pi * 180

    return azi, ele

def rms(x: torch.Tensor):
    """Computes th RMS (root-mean-squared) for each channel in the signal tensor, in dB.

    Parameters
    ----------
    x : Tensor
        Input signal in format [..., channels, timesteps]

    """
    tmp = torch.sqrt(torch.mean(x ** 2, dim=-1))
    t = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80)
    # tmp = torchaudio.functional.amplitude_to_DB(tmp, multiplier=10, amin=80, db_multiplier=1)
    tmp = t(tmp)
    return tmp


def sample_beta(alpha: float = 0.0, beta: float = None, shape=[1]):
    """ Draws a sample from the beta distribution. """
    if beta is None:
        beta = alpha
    if alpha > 0 and beta > 0:
        try:
            dist = torch.distributions.beta.Beta(alpha, beta)
            lambda_var = dist.sample(sample_shape=shape)
        except:
            print(f'{alpha}     {beta}')
            raise
    else:
        if beta == 0:
            prob = alpha if alpha <= 1 else 1
        else:
            prob = (1 - beta) if beta <= 1 else 0
        lambda_var = torch.distributions.bernoulli.Bernoulli(probs=prob).sample(sample_shape=shape)
    return lambda_var


def sample_geometric(low=0, high=1, shape=[1]):
    """ Draw a random sample from a geometric distribution between low and high.
    This returns integers.

    From Bergstra and Bengio:
        "We will use the phrase drawn gemietrically from A to B for 0 < A < B to mean drawing
        uniformly in the log domain between log(A) and log(B), exponentiating to get a number
        between A dn B, and then rounding to the nearest integer."
    """

    tmp_low, tmp_high = torch.log(tmp * low), torch.log(tmp * high)
    tmp = torch.rand(size=shape)
    sample = torch.round(torch.exp(tmp_low + (tmp_high - tmp_low) * tmp))
    return sample


def sample_exponential(low=0, high=1, shape=[1], device='cpu'):
    """ Draw a random sample from an exponential distribution between low and high.
    This returns floats.

    From Bergstra and Bengio:
        This is like geometricSample, but without rounding to the nearest integer.

    """
    eps = 1e-10
    tmp = torch.ones(shape, device=device)
    tmp_low, tmp_high = torch.log(tmp * low + 1e-15), torch.log(tmp * high)
    tmp = torch.rand(size=shape, device=device)
    sample = torch.exp(tmp_low + (tmp_high - tmp_low) * tmp)
    sample[sample <= eps] = 0
    return sample


def test_beta_distributions(sampler='pytorch'):
    if sampler == 'pytorch':
        sampler_f = sample_beta
    else:
        sampler_f = mixup_data

    alphas = [0, 0.25, 0.5, 0.75, 1, 2, 10]
    betas = alphas
    trials = 5000
    results = np.zeros((trials, len(alphas), len(betas)))
    bins = 25
    for counter_alpha, this_alpha in enumerate(alphas):
        for counter_beta, this_beta in enumerate(betas):
            print(f'Drawing samples: alpha = {this_alpha}, beta = {this_beta}')
            # for i in range(trials):
            #    results[i, counter_alpha, counter_beta] = sampler_f(this_alpha, this_beta)
            results[:, counter_alpha, counter_beta] = sampler_f(this_alpha, this_beta, shape=[trials])

    fig, axes = plt.subplots(len(alphas), len(alphas), figsize=(10, 10), sharex=True, sharey=True)
    for ii, this_alpha in enumerate(alphas):
        for jj, this_beta in enumerate(betas):
            dat = results[:, ii, jj]
            ax = axes[ii, jj]
            ax.hist(dat, bins=bins, density=True, log=True)
            ax.set_title(f'a={this_alpha}, b={this_beta}')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    test_beta_distributions()
    print('Finished test')


