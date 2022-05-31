'''
The file containd utility functions.

Author: Ricardo Falcon
2021
'''

import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from typing import Iterable, Tuple, TypeVar, Callable, Any, List, Union
from matplotlib.colors import to_rgb

import spaudiopy as spa


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


