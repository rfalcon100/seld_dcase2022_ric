'''
The file contains plotting functions.

Author: Ricardo Falcon
2021
'''

import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from typing import Iterable, Tuple, TypeVar, Callable, Any, List, Union
from matplotlib.colors import to_rgb
import seaborn as sns
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif',
              font_scale=1, color_codes=True, rc={'pcolor.shading': 'auto'})

import utils
import spaudiopy as spa
from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points


def plot_fft(signal: torch.Tensor, fs: int, title: str = 'Frequency Response'):
    '''Plots the frequency response (magnitude) of a signal. '''
    assert len(signal.shape) == 2, 'The signal should be [channels, timesteps] for plotting.'
    n = signal.shape[-1]
    T = 1 / fs
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(n / 2))
    fig = plt.figure()
    for channel in range(signal.shape[0]):
        yf = 20 * np.log10(np.abs(scipy.fft.fft(signal[channel, :].numpy())))
        plt.plot(xf, yf.reshape(-1)[:n // 2])
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlim([63, fs / 2])
    plt.ylim([-80, 10])
    plt.xticks([125, 250, 500, 1000, 2000, 4000, 8000], [125, 250, 500, 1000, 2000, 4000, 8000])
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    fig.suptitle(title)
    plt.show()


def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform",
                  xlim: List = None, ylim: List = None):
    '''Plots a waveform in time domain.'''
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(dim=-2)  # Add freq_band
    assert len(waveform.shape) == 3, 'The signal should be [channels, freqs, timesteps]'
    waveform = waveform.numpy()
    num_channels, freq_bands, num_frames = waveform.shape[-3::]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(1, num_channels, sharey=True)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        for freq in range(freq_bands):
            axes[c].plot(time_axis, waveform[c, freq, :], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_stft_features(feature, share_colorbar=True, title=None):
    """ Plots time-frequency features. This is useful to visualize for example: stft + intensity_vector
    The shared colorbar is just for visualization, to compare the colors of the different channels.
    But it does not change the values.

    Pass stft features like [channels, freqs, frames]
    """
    #features = features.numpy()
    #num_channels, num_frames = feature.shape
    #time_axis = torch.arange(0, num_frames) / sample_rate
    assert len(feature.shape) == 3, 'ERROR, plotting spectrograms does not support batches.'

    if share_colorbar:
        if feature.shape[0] > 4:
            warnings.warn('WARNING: Plotting with shared colorbar is weird when the input features has more than 4 channels. I was expecting STFT + IV or similar')
        vmin_stft, vmax_stft = torch.min(feature[0:4]), torch.max(feature[0:4])
        vmin_phase, vmax_phase = torch.min(feature[4:-1]), torch.max(feature[4:-1])
    else:
        vmin_stft, vmax_stft = None, None
        vmin_phase, vmax_phase = None, None

    fig, ax = plt.subplots(feature.shape[-3], 1, figsize=(12,12))
    for i in range(feature.shape[-3]):
        if i in range(4):
            vmin, vmax = vmin_stft, vmax_stft
        else:
            vmin, vmax = vmin_phase, vmax_phase
        aa = ax[i].matshow(feature[i, :, :], aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        fig.colorbar(aa, ax=ax[i], location='right')
        ax[i].grid(False)
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)
    plt.show()

def plot_3dpoints(points: torch.Tensor, title: str = 'Grid of points',
                  xlim: List = None, ylim: List = None, zlim: List = None, fig=None):
    """Plots a 3d scatter of poits. USeful to look at sampling of a sphere or similar."""
    assert points.shape[1] == 3, 'The points should be in format [n, 3]'

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for row in range(points.shape[0]):
        ax.scatter(points[row, 0], points[row, 1], points[row, 2])

    # Draw axis lines
    x0 = np.array([1, 0, 0])
    y0 = np.array([0, 1, 0])
    z0 = np.array([0, 0, 1])
    for i in range(3):
        ax.plot([-x0[i], x0[i]], [-y0[i], y0[i]], [-z0[i], z0[i]], '--k',
                alpha=0.3)
    # Formatting
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.view_init(30, 45)

    fig.suptitle(title)
    plt.show()


def plot_matrix2polar(t_matrix, order, plot_channel=0, linewidth=2, INDB=True, rlim=None, title=None, ax=None,
                      show_plot=True, do_scaling=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Plots a polar plot of a cross sections for azimuth and elevation for a spherical function.
    For example, this is useful to visualize the polar pattern of the transformation matrix T that is a
    applied to a signal.

    x_hat =  [Y^T * G * W] * X
    where t_matrix = [Y^T * G * W]

    So this function plots the polar patterns of the transformation t_matrix.
    This is done by defining a Y_grid, which is the spherical harmonics of a dense grid of points
    of a unitary sphere. The cross sections are then the XY plane (for azimuth) and XZ plane (for elevation).

    Returns:
        Tuple of tensors of [x, y] for azimuth and elevation responses.

    """
    assert t_matrix.shape[-1] == (order + 1) ** 2, 'Wrong shape for the transformation matrix.'
    if isinstance(t_matrix, np.ndarray):
        t_matrix = torch.from_numpy(t_matrix)

    if ax is None:
        fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))
    n_points = 720

    if do_scaling:
        scale = 4 * np.pi / (order + 1) ** 2  # Scaling the points to accomdate for order
        scale = np.pi
    else:
        scale = 1

    # Azimuth cross section
    azis = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
    Y_grid = spa.sph.sh_matrix(order, azis, utils.ele2colat(np.zeros_like(azis)), SH_type='real',
                               weights=None)  # Level seems ok for input_order == 1
    Y_grid = torch.from_numpy(Y_grid) * scale
    grid_hat_azi = torch.matmul(t_matrix.float(), Y_grid.type(torch.FloatTensor).transpose(1, 0))
    # this_title = r'Azimuth $\hat{X}$' + title if title is not None else r'Azimuth $\hat{X}$'
    this_title = 'Azimuth' + title if title is not None else 'Azimuth'
    _plot_polar(azis, grid_hat_azi[plot_channel, :], title=this_title, ax=axs[0], INDB=INDB, rlim=rlim,
                linewidth=linewidth)

    # Elevation cross section
    eles = np.linspace(-1 * np.pi / 2, 1 * np.pi / 2, int(n_points / 2), endpoint=True)
    eles = np.concatenate([eles, eles[::-1]])
    eles_azis = np.concatenate([np.zeros_like(eles[0:int(n_points / 2)]),
                                np.pi * np.ones_like(eles[int(n_points / 2):])])
    Y_grid = spa.sph.sh_matrix(order, eles_azis, utils.ele2colat(eles), SH_type='real', weights=None)
    Y_grid = torch.from_numpy(Y_grid) * scale
    grid_hat_ele = torch.matmul(t_matrix.float(), Y_grid.type(torch.FloatTensor).transpose(1, 0))
    this_title = 'Elevation' + title if title is not None else 'Elevation'
    eles_plot = azis - np.pi / 2
    _plot_polar(eles_plot, grid_hat_ele[plot_channel, :], title=this_title, ax=axs[1], plot_elevation=True, INDB=INDB,
                rlim=rlim, linewidth=linewidth, rticks=4)
    plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=1.0)
    if show_plot:
        plt.show()

    return torch.from_numpy(azis), grid_hat_azi[:, :], \
           torch.from_numpy(eles_plot), grid_hat_ele[:, :]


def plot_matrix2polarLinear(t_matrix, order, method='fibonacci', plot_channel=0, INDB=True, rlim=None, title=None,
                            ax=None, show_plot=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Plots a polar plot of a linear projection of the 3d response to
     azimuth and elevation for a spherical function.

    Returns:
        Tuple of tensors of [x, y] for azimuth and elevation responses.

    """
    assert t_matrix.shape[-1] == (order + 1) ** 2, 'Wrong shape for the transformation matrix.'
    if isinstance(t_matrix, np.ndarray):
        t_matrix = torch.from_numpy(t_matrix)

    n_points = 720
    if method == 'fibonacci':
        sphere_points = sphere_fibonacci_grid_points(n_points)
    else:
        sphere_points = spa.grids.load_t_design(degree=21)
        n_points = sphere_points.shape[0]

    if isinstance(t_matrix, np.ndarray):
        t_matrix = torch.from_numpy(t_matrix)

    # Spherical harmonics of the points
    sphere_dirs = utils.vecs2dirs(sphere_points)
    Y_grid = spa.sph.sh_matrix(order, sphere_dirs[:, 0], sphere_dirs[:, 1], SH_type='real', weights=None)
    Y_grid = torch.from_numpy(Y_grid)  # should be 4 x n_points (for first order )
    grid_hat = torch.matmul(t_matrix.float(), Y_grid.type(torch.FloatTensor).transpose(1, 0))

    if ax is None:
        fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))
    n_points = 720

    responses_cart = []
    for channel in range(grid_hat.shape[0]):
        this_response_cart = spa.utils.sph2cart(sphere_dirs[:, 0], sphere_dirs[:, 1], r=grid_hat[channel, :])
        this_response_cart = torch.from_numpy(np.asarray(this_response_cart))
        responses_cart.append(this_response_cart)
    # [ambisonic_channels, coordinates, n_points]
    responses_cart = torch.stack(responses_cart, dim=0)

    projected_cart = []
    projected_sph = []
    for channel in range(grid_hat.shape[0]):
        this_response_cart = responses_cart[channel, :, :]
        # Azimnuth: projection to x,y plane
        U = torch.tensor([[1, 0, 0], [0, 1, 0]]).transpose(1, 0).float()
        U = U / torch.norm(U)
        U = torch.matmul(U, U.transpose(1, 0))

        x_tilde = torch.matmul(U, this_response_cart.float())
        x_tilde_sph = spa.utils.cart2sph(x_tilde[0, :], x_tilde[1, :], x_tilde[2, :])
        projected_cart.append(x_tilde)
        projected_sph.append(torch.from_numpy(np.asarray(x_tilde_sph)))
    tmp = projected_sph[0].transpose(1, 0)
    this_title = 'Azimuth (X_{hat})' + title if title is not None else 'Azimuth (X_{hat})'

    # TODO: This is intersting, but not quite what I want
    # This idea idea of hte lienar projection might not be too good afterall
    # GRoup by azimiuth and get maximum values
    coso, idx = np.unique(np.round(tmp[:, 0].numpy(), decimals=0), return_index=True)
    yolo = np.maximum.reduceat(tmp[:, 2].numpy(), idx)
    axs[0].scatter(coso, 20 * np.log10(yolo))
    axs[0].set_rlim([-40, 10])
    axs[1].scatter(tmp[:, 0].numpy(), 20 * np.log10(tmp[:, 2].numpy()))
    axs[1].set_rlim([-40, 10])
    plt.show()

    axs[0].scatter(tmp[:, 0].numpy(), tmp[:, 2].numpy())
    # _plot_polar(tmp[:, 0], tmp[:, 2], title=this_title, ax=axs[0], INDB=INDB, rlim=rlim)
    plt.show()

    return 0

    # Elevation cross section
    eles = np.linspace(-1 * np.pi / 2, 1 * np.pi / 2, int(n_points / 2), endpoint=True)
    eles = np.concatenate([eles, eles[::-1]])
    eles_azis = np.concatenate([np.zeros_like(eles[0:int(n_points / 2)]),
                                np.pi * np.ones_like(eles[int(n_points / 2):])])
    Y_grid = spa.sph.sh_matrix(order, eles_azis, utils.ele2colat(eles), SH_type='real', weights=None)
    Y_grid = torch.from_numpy(Y_grid)  # should be 4 x n_points (for first order )
    grid_hat_ele = torch.matmul(t_matrix, Y_grid.type(torch.FloatTensor).transpose(1, 0))
    this_title = 'Elevation (X_{hat})' + title if title is not None else 'Elevation (X_{hat})'
    eles_plot = azis - np.pi / 2
    _plot_polar(eles_plot, grid_hat_ele[plot_channel, :], title=this_title, ax=axs[1], plot_elevation=True, INDB=INDB,
                rlim=rlim)
    plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=1.0)
    if show_plot:
        plt.show()

    # return torch.stack([torch.from_numpy(azis), grid_hat_azi[:,:]], dim=0), \
    #       torch.stack([torch.from_numpy(eles_plot), grid_hat_ele[:,:]], dim=0)
    return torch.from_numpy(azis), grid_hat_azi[:, :], \
           torch.from_numpy(eles_plot), grid_hat_ele[:, :]


def plot_sphere_points(t_matrix, order=1, plot_channel=0, method='fibonacci', show_plot=True, fig=None,
                       do_scaling=False, debugging=False):
    # Test the method with the sphere
    # new_points = [Y_tranpose * G * W] * Y_grid
    n_points = 720

    if method == 'fibonacci':
        sphere_points = sphere_fibonacci_grid_points(n_points)
    else:
        sphere_points = spa.grids.load_t_design(degree=21)
        n_points = sphere_points.shape[0]

    if isinstance(t_matrix, np.ndarray):
        t_matrix = torch.from_numpy(t_matrix)

    if do_scaling:
        scale = 4 * np.pi / (order + 1) ** 2  # Scaling the points to accomdate for order
        scale = np.pi
    else:
        scale = 1

    # Spherical harmonics of the points in the grid
    sphere_dirs = utils.vecs2dirs(sphere_points)
    Y_grid = spa.sph.sh_matrix(order, sphere_dirs[:, 0], sphere_dirs[:, 1], SH_type='real', weights=None)
    Y_grid = torch.from_numpy(Y_grid) * scale

    # Transform the grid
    grid_hat = torch.matmul(t_matrix.double(), Y_grid.type(torch.DoubleTensor).transpose(1, 0))

    # Debugging
    if debugging:
        tmp = torch.sum(grid_hat, dim=0)
        print(f't_matrix: {t_matrix.shape}')
        print(f'Y_grid: {Y_grid.shape}')
        print(f'grid_hat: {grid_hat.shape}')
        print(f'grid_hat: {spa.utils.rms(grid_hat.numpy()).shape}')

        tmp = spa.sph.src_to_sh(np.ones((n_points, 1)), sphere_dirs[:, 0], sphere_dirs[:, 1], order)
        print(tmp.shape)
    if show_plot:
        spa.plots.spherical_function(grid_hat[plot_channel, :], sphere_dirs[:, 0], sphere_dirs[:, 1], fig=fig)
        ##        spa.plots.sh_coeffs(spa.utils.rms(grid_hat.numpy()), title="Input", fig=fig)   # this is worng, shows some very weird patterns, kinda like its doing rms over spacE?
        ###         spa.plots.spherical_function(spa.utils.rms(grid_hat.numpy()), sphere_dirs[:, 0], sphere_dirs[:, 1], fig=fig)    ### promising , but wronmg
        plt.show()

    # This might be useful...
    # Debugging
    if debugging:
        print(t_matrix.shape)
        # spa.plots.sh_coeffs_subplot(t_matrix[0:4, 0:4])
        spa.plots.sh_coeffs_subplot(t_matrix[:, :])
        plt.show()

    # Get the cartesian coordinates for each channel in the response
    responses = []
    for channel in range(grid_hat.shape[0]):
        this_response = spa.utils.sph2cart(sphere_dirs[:, 0], sphere_dirs[:, 1], r=grid_hat[channel, :])
        this_response = np.asarray(this_response)
        responses.append(torch.from_numpy(this_response))
    # [ambisonic_channels, coordinates, n_points]
    responses = torch.stack(responses, dim=0)
    return responses


def plot_mollweide(t_matrix, order=1, inDB=True, plot_channel=0, title="Mollweide"):
    # TODO This kinda works, but the it does not make sense to show phase using db,
    n_points = 100 ** 2
    azi = np.linspace(-np.pi, np.pi, 360)
    ele = np.linspace(-np.pi / 2., np.pi / 2., 180)
    grid_azi, grid_ele = np.meshgrid(azi, ele)

    if isinstance(t_matrix, np.ndarray):
        t_matrix = torch.from_numpy(t_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')

    # Spherical harmonics of the points
    sphere_dirs = np.array([grid_azi.ravel(), utils.ele2colat(grid_ele.ravel())]).transpose(1, 0)
    Y_grid = spa.sph.sh_matrix(order, sphere_dirs[:, 0], sphere_dirs[:, 1], SH_type='real', weights=None)
    Y_grid = torch.from_numpy(Y_grid)  # should be 4 x n_points (for first order )
    grid_hat = torch.matmul(t_matrix.float(), Y_grid.type(torch.FloatTensor).transpose(1, 0))
    grid_hat = grid_hat[plot_channel, :].reshape(*grid_azi.shape)
    if inDB:
        rpos = np.copy(grid_hat)
        rpos[grid_hat < 0] = np.nan
        rneg = np.copy(grid_hat)
        rneg[grid_hat >= 0] = np.nan
        im = ax.pcolormesh(grid_azi, grid_ele, spa.utils.db(rpos))
        im2 = ax.pcolormesh(grid_azi, grid_ele, spa.utils.db(abs(rneg)))
    # im = ax.pcolormesh(grid_azi, grid_ele, grid_hat, cmap='magma')
    # im = ax.pcolormesh(grid_azi, grid_ele, grid_hat, cmap='coolwarm_r')
    # fig.colorbar(im, cax=ax)
    # fig.colorbar(im)
    plt.show()


def _plot_polar(theta, r, INDB=True, rlim=None, title=None, ax=None, plot_elevation=False, linewidth=1, rticks=6):
    """
    Adapted from Spaudiopy, with added parameter for linewidth and rticks.

    Polar plot (in dB) that allows negative values for `r`.
    This plot compares a reference with r=1 for all theta.
    Examples
    --------
    See :py:func:`spaudiopy.sph.bandlimited_dirac`
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='polar')
    # Split in pos and neg part and set rest NaN for plots
    rpos = np.copy(r)
    rpos[r < 0] = np.nan
    rneg = np.copy(r)
    rneg[r >= 0] = np.nan
    if INDB:
        r_ref = np.zeros_like(r)
        if not rlim:
            rlim = (-40, 10)
        ax.plot(theta, spa.utils.db(rpos), label='$+$', linewidth=linewidth)
        ax.plot(theta, spa.utils.db(abs(rneg)), label='$-$', linewidth=linewidth)
        ax.text(6.5 / 8 * 2 * np.pi, 20.3, 'dB', horizontalalignment='left')
    else:
        r_ref = np.ones_like(r)
        if not rlim:
            rlim = (0, 1)
        ax.plot(theta, rpos, label='$+$')
        ax.plot(theta, np.abs(rneg), label='$-$')

    if plot_elevation:
        ax.set_theta_zero_location('W')
        ax.set_theta_offset(np.pi)
    else:
        ax.set_theta_offset(np.pi / 2)

    ax.plot(theta, r_ref, 'r--', label='ref', linewidth=1)
    ax.set_rmin(rlim[0])
    ax.set_rmax(rlim[1] + (0.5 if INDB else 0.03))
    ax.set_rticks(np.linspace(rlim[0], rlim[1], 6))
    ax.set_rlabel_position(6.5 / 8 * 360)

    if plot_elevation:
        ax.set_thetagrids(angles=[0, 45, 90, 315, 270], labels=['0°', '45°', '90°', '-45°', '-90°'])
        ax.set_theta_direction(-1)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 0.25))

    if title is not None:
        ax.set_title(title)


def plot_transform_matrix(matrix_t: torch.Tensor, indb=True, title=None, xlabel='Input', ylabel='Output',
                          clamp_db_value=-50):
    """ Simple function to plot a 2d matrix. Useful to visuzalize the directional gain matrix G, or the
    full transformation matrix T. """
    if indb:
        matrix_t = 20 * torch.log10(matrix_t)
        matrix_t = torch.clamp(matrix_t, clamp_db_value, 0)
    # T = bb * scale
    # T = 20 * torch.log10(T)
    # T = torch.clamp(T, -50, 0)

    fig = plt.figure()
    if indb:
        ax = plt.imshow(matrix_t, cmap='magma', vmax=0)
    else:
        ax = plt.imshow(matrix_t, cmap='magma')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if indb:
        cbar = fig.colorbar(ax, ticks=[-40, -30, -20, -10, 0])
        cbar.ax.set_yticklabels(['-40 dB', '-30 dB', '-20 dB', '-10 dB', '0 dB'])  # vertically oriented colorbar
    plt.title(title)
    plt.show()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_labels(labels: Union[torch.Tensor, np.ndarray], n_classes: Union[int, List[int]] = None,
                ylim=None, title=None, tkagg=False, savefig=True, plot_cartesian=True, size_modifier=25):
    """Plots the labels as either x,y,z coordinates, or azimuth, elevation, magnitude."""
    if tkagg:
        import matplotlib
        matplotlib.use('TkAgg')  # Trick so that I can see the plots when debugging TODO Fix this

    assert len(labels.shape) == 3, 'Labels should be [3, n_classes, n_frames]'
    assert labels.shape[0] == 3, 'Labels should be [3, n_classes, n_frames]'
    if n_classes is None:
        n_classes = np.arange(labels.shape[1])
    cmap = get_cmap(len(n_classes))
    if plot_cartesian:
        y_labels = ['x', 'y', 'z']
        ylim = [-1, 1] if ylim is None else ylim
    else:
        y_labels = ['azimuth', 'elevation', 'r']

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 12))
    plt.suptitle(title)
    if plot_cartesian:
        for ii in range(3):
            for jj in n_classes:
                tmp_data = labels[ii, jj, :]
                tmp_x = np.arange(tmp_data.shape[-1])
                axs[ii].plot(tmp_x, tmp_data, color=cmap(jj))
            axs[ii].set_ylabel(f'{y_labels[ii]}')
            axs[ii].grid()
            axs[ii].set_ylim(ylim)
    else:
        for ii in range(3):
            if ii != 2:
                for jj in n_classes:
                    tmp_data = labels[ii, jj, :]
                    tmp_r = np.minimum(labels[2, jj, :], 1)
                    tmp_x = np.arange(tmp_data.shape[-1])
                    r, g, b = to_rgb(cmap(jj))  # Vector magnitude determines the alpha
                    tmp_color = [(r, g, b, alpha) for alpha in tmp_r]
                    axs[ii].scatter(tmp_x, tmp_data, s=10, color=tmp_color)
            else:
                for jj in n_classes:
                    tmp_r = labels[ii, jj, :]
                    tmp_x = np.arange(tmp_data.shape[-1])
                    tmp_y = np.ones_like(tmp_x) * jj
                    axs[ii].scatter(tmp_x, tmp_y, s=tmp_r * size_modifier, color=cmap(jj))
            axs[ii].set_ylabel(f'{y_labels[ii]}')
            axs[ii].grid()
            eps = 1e-1
            if ii == 0:
                axs[ii].set_ylim([0 - eps, 2 * np.pi + eps])
            elif ii == 1:
                axs[ii].set_ylim([-np.pi / 2 - eps, np.pi / 2 + eps])
            elif ii == 2:
                axs[ii].set_ylim([0 - eps, len(n_classes) + eps])
    plt.tight_layout()
    plt.show()
    if savefig:
        plt.savefig(f'labels_{title}.png')
    return fig


def plot_labels_cross_sections(labels: Union[torch.Tensor, np.ndarray], n_classes: Union[int, List[int]] = None,
                               rlim=None, title=None, tkagg=False, savefig=True, plot_cartesian=True, size_modifier=25):
    """ Plots the ACCDOA labels [3, n_classes, frames] as 2d cross section plots for azimuth, elevation, and a plot for
    the vector length. Used to visualize the labels."""
    if tkagg:
        import matplotlib
        matplotlib.use('TkAgg')  # Trick so that I can see the plots when debugging TODO Fix this

    assert len(labels.shape) == 3, 'Labels should be [3, n_classes, n_frames]'
    assert labels.shape[0] == 3, 'Labels should be [3, n_classes, n_frames]'
    if n_classes is None:
        n_classes = np.arange(labels.shape[1])
    cmap = get_cmap(len(n_classes))

    fig = plt.figure(figsize=(9, 12))
    # fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 12))

    ax0 = plt.subplot(2, 2, 1, projection='polar')
    ax1 = plt.subplot(2, 2, 2, projection='polar')
    ax2 = plt.subplot(4, 1, 3)
    ax3 = plt.subplot(4, 1, 4)
    plt.suptitle(title)

    if rlim is None:
        rlim = [0, 2]
    for jj in n_classes:
        labels_polar = utils.vecs2dirs(labels[:, jj, :].permute([1, 0]), positive_azi=True, include_r=True,
                                 use_elevation=True)
        r, g, b = to_rgb(cmap(jj))  # Vector magnitude determines the alpha
        tmp_color = [(r, g, b, alpha) for alpha in labels_polar[:, 2]]
        tmp_color = [(r, g, b, 1) for alpha in labels_polar[:, 2]]

        # Azimuth  = projection over XY plane
        plane_xy = torch.tensor([[0, 0, 1.0]])  # XY plane is defined by the normal vector over z axis
        tmp_proj = labels[:, jj, :].transpose(1, 0) - torch.matmul(plane_xy, labels[:, jj, :]).transpose(1, 0) * plane_xy
        labels_polar_proj = utils.vecs2dirs(tmp_proj, positive_azi=True, include_r=True, use_elevation=True)
        ax0.scatter(labels_polar_proj[:, 0], labels_polar_proj[:, 2], s=10, color=tmp_color)

        # Elevation = Projection over XZ plane
        plane_xz = torch.tensor([[0, 1.0, 0]])  # XZ plane is defined by the normal vector over y axis
        tmp_proj = labels[:, jj, :].transpose(1, 0) - torch.matmul(plane_xz, labels[:, jj, :]).transpose(1, 0) * plane_xz
        labels_polar_proj = utils.vecs2dirs(tmp_proj, positive_azi=True, include_r=True, use_elevation=True)

        ids_behind = np.logical_and(labels_polar_proj[:, 0] > np.pi / 2, labels_polar_proj[:, 0] < 3 * np.pi / 2)
        labels_polar_proj[ids_behind, 1] = labels_polar_proj[ids_behind, 1] + (np.pi - 2 * labels_polar_proj[ids_behind, 1])
        ax1.scatter(labels_polar_proj[:, 1], labels_polar_proj[:,2], s=10, color=tmp_color)

        # Plot of vector lengths
        ax2.plot(torch.arange(0, labels.shape[-1]), labels_polar[:, 2], color=cmap(jj))

    # Plot class activity
    ax3.imshow(labels.permute([1, 2, 0]), interpolation='nearest', aspect='auto', origin='lower')  # imshow needs [height, width, channels]

    # Formatting
    eps = 1e-1
    this_title = 'Azimuth'
    ax0.set_title(f'{this_title}')
    ax0.set_theta_offset(np.pi / 2)
    ax0.set_rmin(rlim[0])
    ax0.set_rmax(rlim[1] + eps)
    ax0.set_rticks(np.linspace(rlim[0], rlim[1], 6))

    rlim = [0, 1]
    this_title = 'Elevation'
    ax1.set_title(f'{this_title}')
    ax1.set_theta_zero_location('W')
    ax1.set_theta_offset(np.pi)
    ax1.set_rmin(rlim[0])
    ax1.set_rmax(rlim[1] + eps)
    ax1.set_thetagrids(angles=[0, 45, 90, 315, 270], labels=['0°', '45°', '90°', '-45°', '-90°'])
    ax1.set_theta_direction(-1)
    ax1.set_rticks(np.linspace(rlim[0], rlim[1], 6))

    this_title = 'Vector Length'
    ax2.set_ylabel(f'{this_title}')
    ax2.set_ylim([0 - eps, 1.0 + eps])
    ax2.grid()
    plt.tight_layout()
    plt.show()
    if savefig:
        plt.savefig(f'labels_{title}.png')
    return fig


def sh_rms_map(F_nm, INDB=False, w_n=None, SH_type=None, azi_steps=5, zen_steps=3,
               title=None, fig=None, vmin=None, vmax=None, return_values=False):
    """Plot spherical harmonic signal RMS as function on the sphere.

    Evaluates the maxDI beamformer, if w_n is None.

    NOTE: Adapted from spaudiopy, but added parameter for the colrobar min
    and max. It also returns the min and max rms values optionally.

    Parameters
    ----------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients, Ambisonic signal.
    INDB : bool
        Plot in dB.
    w_n : array_like
        Modal weighting of beamformers that are evaluated on the grid.
    SH_type :  'complex' or 'real' spherical harmonics.

    Examples
    --------
    See :py:mod:`spaudiopy.sph.src_to_sh`

    """
    F_nm = np.atleast_2d(F_nm)
    assert (F_nm.ndim == 2)
    if SH_type is None:
        SH_type = 'complex' if np.iscomplexobj(F_nm) else 'real'
    N_sph = int(np.sqrt(F_nm.shape[0]) - 1)

    azi_steps = np.deg2rad(azi_steps)
    zen_steps = np.deg2rad(zen_steps)
    azi_plot, zen_plot = np.meshgrid(np.arange(np.pi, -(np.pi + azi_steps),
                                               -azi_steps),
                                     np.arange(10e-3, np.pi + zen_steps,
                                               zen_steps))

    Y_smp = spa.sph.sh_matrix(N_sph, azi_plot.ravel(), zen_plot.ravel(), SH_type)
    if w_n is None:
        w_n = spa.sph.hypercardioid_modal_weights(N_sph)
    f_d = Y_smp @ np.diag(spa.sph.repeat_per_order(w_n)) @ F_nm
    rms_d = np.abs(spa.utils.rms(f_d, axis=1))

    if INDB:
        rms_d = spa.utils.db(rms_d)

    if fig is None:
        fig = plt.figure(constrained_layout=True)
    ax = fig.gca()
    ax.set_aspect('equal')

    if vmin is not None and vmax is not None:
        p = ax.pcolormesh(azi_plot, zen_plot, np.reshape(rms_d, azi_plot.shape), shading='auto', vmin=vmin, vmax=vmax)

    else:
        p = ax.pcolormesh(azi_plot, zen_plot, np.reshape(rms_d, azi_plot.shape), shading='auto')
    ax.grid(True)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$',
                        r'$\pi/2$', r'$\pi$'])
    ax.set_yticks(np.linspace(0, np.pi, 3))
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Zenith')

    plt.axhline(y=np.pi / 2, color='grey', linestyle=':')
    plt.axvline(color='grey', linestyle=':')

    plt.xticks([np.pi, np.pi / 2, 0, -np.pi / 2, -np.pi],
               labels=[r"$\pi$", r"$\pi/2$", r"$0$", r"$-\pi/2$", r"$-\pi$"])
    plt.yticks([0, np.pi / 2, np.pi],
               labels=[r"$0$", r"$\pi/2$", r"$\pi$", ])

    cb = plt.colorbar(p, ax=ax, shrink=0.5)
    cb.set_label("RMS in dB" if INDB else "RMS")
    if title is not None:
        ax.set_title(title)

    if return_values:
        return np.min(rms_d), np.max(rms_d)


def sh_rms_map_mollweide(F_nm, INDB=False, w_n=None, SH_type=None, azi_steps=5, zen_steps=3,
                         title=None, fig=None, vmin=None, vmax=None, return_values=False):
    """Plot spherical harmonic signal RMS as function on the sphere, as a mollweide map

    Evaluates the maxDI beamformer, if w_n is None.

    NOTE: Adapted from spaudiopy, but added parameter for the colrobar min
    and max. It also returns the min and max rms values optionally.

    NOTE: The Mollweide plot has the x (longitude) axis ticks fixed , and they dont follow
    the same convention as we do here. So I do a fake rotation just for that.

    Parameters
    ----------
    F_nm : ((N+1)**2, S) numpy.ndarray
        Matrix of spherical harmonics coefficients, Ambisonic signal.
    INDB : bool
        Plot in dB.
    w_n : array_like
        Modal weighting of beamformers that are evaluated on the grid.
    SH_type :  'complex' or 'real' spherical harmonics.

    Examples
    --------
    See :py:mod:`spaudiopy.sph.src_to_sh`

    """
    F_nm = np.atleast_2d(F_nm)
    assert (F_nm.ndim == 2)
    if SH_type is None:
        SH_type = 'complex' if np.iscomplexobj(F_nm) else 'real'
    N_sph = int(np.sqrt(F_nm.shape[0]) - 1)

    azi_steps = np.deg2rad(azi_steps)
    zen_steps = np.deg2rad(zen_steps)
    azi_plot, zen_plot = np.meshgrid(np.arange(np.pi, -(np.pi + azi_steps),
                                               -azi_steps),
                                     np.arange(10e-3, np.pi + zen_steps,
                                               zen_steps))

    # Fake rotation to plot the correct azimuth in the projection
    azi_plot_plot, zen_plot_plot = np.meshgrid(np.arange(-np.pi, (np.pi + azi_steps),
                                                         azi_steps),
                                               np.arange(10e-3, np.pi + zen_steps,
                                                         zen_steps))
    zen_plot_plot = utils.colat2ele(zen_plot_plot)  # For Mollweide, we use elevation

    Y_smp = spa.sph.sh_matrix(N_sph, azi_plot.ravel(), zen_plot.ravel(), SH_type)
    if w_n is None:
        w_n = spa.sph.hypercardioid_modal_weights(N_sph)
    f_d = Y_smp @ np.diag(spa.sph.repeat_per_order(w_n)) @ F_nm
    rms_d = np.abs(spa.utils.rms(f_d, axis=1))

    if INDB:
        rms_d = spa.utils.db(rms_d)

    if fig is None:
        fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection='mollweide')

    if vmin is not None and vmax is not None:
        p = ax.pcolormesh(azi_plot_plot, zen_plot_plot, np.reshape(rms_d, azi_plot.shape), shading='auto', vmin=vmin,
                          vmax=vmax)

    else:
        p = ax.pcolormesh(azi_plot_plot, zen_plot_plot, np.reshape(rms_d, azi_plot.shape), shading='auto')
    ax.grid(True)

    cb = plt.colorbar(p, ax=ax, shrink=0.5)
    cb.set_label("RMS in dB" if INDB else "RMS")
    if title is not None:
        ax.set_title(title)

    if return_values:
        return np.min(rms_d), np.max(rms_d)

def plot_distribution_azi_ele(points: Union[torch.Tensor, np.ndarray], type='hex', title='', filename=None,
                              log_scale=True, bins=15, gridsize=15, kde_fill=True, kde_levels=8, cmin=0):
    """ This plots the bivariate distribution for azimuth and elevation, with marginal distributions
    for each.
    Options:
        type: ['hex', 'kde', 'hist']
        log_scale: [True, False]  --> log scale for the count (not the axis)
        kde_fill: [True, False]   --> only for kde
        kde_levels: int  ---> only for kde
        bins = 15  --> for the maginals
        gridsize = 15  , when using kde, it might be good to go higher, 100 or so.
        cmin = 0 , ---> min value to display, set to > 0 when using linear scale to get rid of the black background

    Some references that are very useful for formatting of these plots:
    https://stackoverflow.com/questions/60947113/seaborn-kdeplot-colorbar
    https://stackoverflow.com/questions/36898008/seaborn-heatmap-with-logarithmic-scale-colorbar
    https://stackoverflow.com/questions/63895392/seaborn-is-not-plotting-within-defined-subplots
    https://stackoverflow.com/questions/21197774/assign-pandas-dataframe-column-dtypes

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

    # Plot starts here
    #fig = plt.figure()  # Not neede because sns.JointGrid is a figure level method
    g = sns.JointGrid(data=df, x="azimuth", y="Elevation")
    g.plot_marginals(sns.histplot, bins=bins, element="step", color="#03012d")
    ax = g.fig.axes[0]
    if type == 'hex':
        bins_scale = 'log' if log_scale else None
        hb = ax.hexbin(x=df['azimuth'], y=df['Elevation'], gridsize=gridsize, bins=bins_scale, cmap='magma', extent=(0, 2 * np.pi, -np.pi / 2, np.pi / 2), mincnt=cmin)
        ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    elif type == 'hist':
        from matplotlib.colors import LogNorm
        normalizer = LogNorm(vmin=10e-1, vmax=ticks[-1]) if log_scale else None
        _, _, _, hb = ax.hist2d(x=df['azimuth'], y=df['Elevation'], bins=gridsize, cmap='magma', range=((0, 2 * np.pi), (-np.pi / 2, np.pi / 2)), norm=normalizer, cmin=cmin)
        ax.grid()
    elif type == 'kde':
        if log_scale: raise ValueError('Log scale is not supported when using KDE')
        hb = g.plot_joint(sns.kdeplot, cmap='magma', levels=kde_levels, cbar=True, fill=kde_fill, gridsize=gridsize)
        ax.axis([0, 2 * np.pi, -np.pi / 2, np.pi / 2])
    if type == 'hex' or type == 'hist':
        cb = g.fig.colorbar(hb, ax=ax, ticks=ticks)
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
    plt.suptitle(title)
    if filename is not None:
        plt.savefig(f'./figures/{filename}.pdf')
        plt.savefig(f'./figures/{filename}.png')
    plt.show()
