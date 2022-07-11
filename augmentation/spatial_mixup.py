'''
The file contains spatial audio data augmentation methods, including:
    - Directional Loudness modifications

This has been updated with the a new option for the backend, using the spatial filterbank.

Author: Ricardo Falcon
2022
'''

import os
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union, Optional, Tuple
import torchaudio
import fnmatch
import warnings

import utils
import plots
import spaudiopy as spa


def get_grid(degree: int):
    """Returns the cartesian coordinates for a t_design.
    This represents a grid of directions, that uniformly samples the unit sphere."""
    t_design = spa.grids.load_t_design(degree=degree)
    return t_design

def draw_random_rotaion_angels() -> torch.Tensor:
    """ Returns a list of random angles for spherical rotations.
    Here we can set up the constraints, for example, to limit the rotation in elevation.
    Remember that hta angles are:
    phi --> roll --> rotate over x axis
    theta --> pitch  --> rotate over y axis  (e.g. elevation)
    psy  --> yaw  --> rotate over z axis (e.g. azimuth)"""
    limits = [(0, np.pi / 2), (0, np.pi / 2), (-np.pi / 4, np.pi / 4)]
    means = torch.tensor([0.0, 0.0, np.pi / 4])
    limits = [(0, np.pi / 2 ),  (-np.pi / 4, np.pi / 4), (0, np.pi / 2)]
    means = torch.tensor([0.0, np.pi / 4, 0.0])

    # testing with rotations over azimtuh only
    limits = [(0, 0 ),  (0,0), (0, np.pi )]
    means = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([np.sum(np.abs(x)) for x in limits])
    tmp = torch.rand(size=(3,))

    #tmp = torch.tensor([0.5, 0.5, 0.5])
    # tmp = torch.tensor([1., 1., 1.])
    # tmp = torch.tensor([0., 0., 0.])
    angles = tmp * b - means
    return angles

def draw_random_spherical_cap(spherical_cap_type='soft', device='cpu') -> Tuple[
    torch.Tensor, torch.Tensor, float, float]:
    """ Draws the parameters for a random spherical cap using:
    spherical_cap_type = 'hard':
        - cap_center = Uniform between [0, 2*pi] for azimuth, and [-pi/2, pi/2]
        - cap_width = Uniform between [pi/4 and pi]
        - g1 = 0
        ######- g1 = Exponential with high = 0, low = -6
        - g2 = Uniform [-20, -6]

    spherical_cap_type = 'soft':
        - cap_center = Uniform between [0, 2*pi] for azimuth, and [-pi/2, pi/2]
        - cap_width = Uniform between [pi/4 and pi]
        - g1 = 0
        ####- g1 = Exponential with high = 0, low = -6
        - g2 = Exponential with high = -3, low = -6
        """

    cap_center = torch.stack([torch.rand(1, device=device) * 2 * np.pi,
                              torch.rand(1, device=device) * np.pi - np.pi / 2], dim=-1)
    cap_width = torch.rand(1, device=device) * (np.pi - np.pi / 4) + np.pi / 4
    # g1 = - utils.sample_exponential(0, 6, shape=[1], device=device)
    g1 = 0

    if spherical_cap_type == 'hard':
        g2 = torch.rand(1, device=device) * -(20 - 6) - 6
    elif spherical_cap_type == 'soft':
        g2 = - utils.sample_exponential(3, 6, shape=[1], device=device)
    else:
        raise ValueError(f'Unsupported spherical cap type: {spherical_cap_type} ')

    return cap_center, cap_width, g1, g2


def compute_Y_and_W(grid, rotation_matrix: torch.Tensor = None, order_input: int = 1, order_output: int = 1,
                    backend='basic', w_pattern='hypercardioid') -> torch.Tensor:
    """ Computes the reconstruction matrix Y, and beamforming matrix W """

    # Directions for the grid in spherical coordinates. Discrete sampling of the sphere
    tmp_directions = utils.vecs2dirs(grid)
    if rotation_matrix is not None:
        tmp_directions_rotated = utils.vecs2dirs(torch.matmul(torch.from_numpy(grid).float(), rotation_matrix.float()))
    else:
        tmp_directions_rotated = tmp_directions
    if backend == 'basic':
        Y = spa.sph.sh_matrix(order_input, tmp_directions[:, 0], tmp_directions[:, 1], SH_type='real', weights=None)
        W = spa.sph.sh_matrix(order_output, tmp_directions_rotated[:, 0], tmp_directions_rotated[:, 1], SH_type='real',
                              weights=None)
    elif backend == 'spatial_filterbank':
        assert order_input == order_output, 'When using spatial filterbank, the input and output orders should be the same'
        assert rotation_matrix is None or torch.all(torch.isclose(rotation_matrix, torch.eye(
            3).double())), 'Soundfield rotations not supported when using spatial filterbank'

        # Weights for polar patterns
        if w_pattern.lower() == "cardioid":
            c_n = spa.sph.cardioid_modal_weights(order_output)
        elif w_pattern.lower() == "hypercardioid":
            c_n = spa.sph.hypercardioid_modal_weights(order_output)
        elif w_pattern.lower() == "maxre":
            c_n = spa.sph.maxre_modal_weights(order_output, True)  # works with amplitude compensation and without!
        else:
            raise ValueError(f'ERROR: Unknown w_pattern type: {w_pattern} . Check spelling? ')
        [W, Y] = spa.sph.design_spat_filterbank(order_output, tmp_directions[:, 0], tmp_directions[:, 1], c_n, 'real',
                                                'perfect')
    else:
        raise ValueError(f'ERROR: Unknown backend : {backend} . Should be either "basic", or "spatial_filterbank"')

    Y = Y.astype(np.double)
    W = W.astype(np.double)
    W = torch.from_numpy(W)
    Y = torch.from_numpy(Y)

    return Y, W

    
class SphericalRotation(nn.Module):
    """ Class to do 3d rotations to signals in spherical harmonics domain

    mode == 'single' --> Applies a single rotation by the specified rotation angles.

    moded == 'random' --> Precomputes num_random_rotations, so that it can be applied fast in runtime.  """

    def __init__(self, rotation_angles_rad: Tuple[float, float, float] = [0.0, 0.0, 0.0],
                 mode='single', num_random_rotations: int = -1, device: str = 'cpu',
                 t_design_degree: int = 3, order_input: int = 1, order_output: int = 1,
                 backend='basic', w_pattern='hypercardioid'):
        super(SphericalRotation, self).__init__()
        assert t_design_degree > 2 * order_output, 'The t-design degree should be > 2 * N_{tilde} of the output order '

        self.rotation_angles_rad = rotation_angles_rad
        self.R = utils.get_rotation_matrix(*self.rotation_angles_rad)
        self.grid = get_grid(degree=t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        self.mode = mode
        self.num_random_rotations = num_random_rotations
        self.device = device
        self.backend = backend
        self.w_pattern = w_pattern

        self.Y = None
        self.W = None
        self.T_mat = None

        if mode == 'single':
            self.rotation_angles_rad = rotation_angles_rad
            Y, W = compute_Y_and_W(self.grid, self.R, self.order_input, self.order_output,
                                   backend=self.backend, w_pattern=self.w_pattern)
            self.Y = Y.to(self.device)
            self.W = W.to(self.device)
            T_mat = torch.matmul(self.Y.transpose(1, 0), self.W)

            if self.backend == 'basic':
                scale = 4 * np.pi / self.n_directions  # TODO August 05, this works ok , except for input_order > 1
                T_mat = scale * T_mat

            self.T_mat = T_mat.to(self.device)

    def forward(self, X: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        assert X.shape[-2] == self.W.shape[
            -1], 'ERROR: The order of the input signal does not match the rotation matrix'

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()
        if X.shape[-2] > X.shape[-1]:  # Channels first format
            warnings.warn('WARNING: It seems that the input tensor X is NOT in channels-first format')
        if self.W is not None and self.Y is not None:
            assert X.shape[-2] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'
        assert self.T_mat.shape[-1] == X.shape[-2], 'Wrong shape for input signal or matrix T.'

        out_x = torch.matmul(self.T_mat, X.double())
        #print(f't_mat shape: {self.T_mat.shape}')
        #print(f'r shape: {self.R.shape}')
        #print(f'targets shape: {targets.shape}')
        out_targets = torch.matmul(targets.double().permute((0, 2, 3, 1)), self.R.transpose(1,0))
        out_targets = out_targets.permute((0, 3, 1, 2))

        return out_x.float(), out_targets.float()

    def __repr__(self):
        rep = "SphericalRotation with: \n"
        rep += f'Device = {self.device} \n'
        rep += f'order_input = {self.order_input} \n'
        rep += f'order_output = {self.order_output} \n'
        rep += f'backend = {self.backend} \n'
        rep += f'w_pattern = {self.w_pattern} \n'
        rep += f'n_directions = {self.n_directions} \n'
        rep += f'Rotation_angles = {self.rotation_angles_rad} \n'

        return rep

    def reset_R(self, rotation_angles_rad: Tuple[float, float, float] = None):
        if rotation_angles_rad is None:
            rotation_angles_rad = draw_random_rotaion_angels()
        self.rotation_angles_rad = rotation_angles_rad
        tmp = utils.get_rotation_matrix(*self.rotation_angles_rad)
        self.R = tmp.to('cpu')  # TODO, in cpu due to compatiblity with spaudiopy, we need to recompute Y

        Y, W = compute_Y_and_W(self.grid, self.R, self.order_input, self.order_output,
                               backend=self.backend, w_pattern=self.w_pattern)
        self.R = self.R.to(self.device)
        self.Y = Y.to(self.device)
        self.W = W.to(self.device)
        self.T_mat = torch.matmul(self.Y.transpose(1, 0), self.W)


class DirectionalLoudness(nn.Module):
    """ Class to modify the directional loudness of an ambisonics recording.

    TODO: Add formula here

    Optionally, soundfield rotation can be done, using a rotation matrix R, computed with the rotation
    angles phi, theta, psi, over the axis x,y,z respectively.

    Backend:
        The backend defines how the matrices Y and W are computed. In the original SpatialMixup paper
        we use a basic form , where Y is a matrix of spherical harmonics pointing to the directions of
        the grid; and W is Y transposed. This is equivalent to a hypercardiods pointing to the grid.
        We then apply and scaling factor to the resulting matrix.

        We also include a spatial filtebank backend, where Y and W and specially constructed matrices to
        have perfect reconstruction, and can include other patterns for the beamforming. This allows for
        more extreme transformations in G.



    References:
     F. Zotter and M. Frank, Ambisonics: A Practical 3D Audio Theory for Recording, Studio Production,
     Sound Reinforcement, and Virtual Reality  Franz Zotter Matthias Frank. Springer, 2019.

     Hold, C., Politis, A., Mc Cormack, L., & Pulkki, V. (2021). Spatial Filter Bank Design in the
     Spherical Harmonic Domain. Proceedings of European Signal Processing Conference, August, 106–110.

     Hold, C., Schlecht, S. J., Politis, A., & Pulkki, V. (2021). SPATIAL FILTER BANK IN THE SPHERICAL
     HARMONIC DOMAIN : RECONSTRUCTION AND APPLICATION. IEEE Workshop on Applications of Signal Processing
     to Audio and Acoustics (WASPAA), 1(1).

     M. Kronlachner, “Spatial transformations for the alteration of ambisonic recordings,” 2014.


"""

    def __init__(self, order_input: int = 1, t_design_degree: int = 3, order_output: int = 1,
                 G_type: str = 'random_diag', G_values: Union[np.ndarray, torch.Tensor] = None,
                 device: str = 'cpu', T_pseudo_floor=1e-8, backend='basic', w_pattern='hypercardioid',
                 use_slepian=False, rotation_angles_rad: Tuple[float, float, float] = None):
        super(DirectionalLoudness, self).__init__()
        assert t_design_degree > 2 * order_output, 'The t-design degree should be > 2 * N_{tilde} of the output order '

        self._Y = None
        self._G = None
        self._W = None
        self.reset_R(rotation_angles_rad)
        self.T_mat = None

        self.grid = get_grid(degree=t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        self.G_type = G_type
        self.device = device
        self.backend = backend  # {'spat_filterbank', 'basic'}
        self.w_pattern = w_pattern
        self.use_slepian = use_slepian

        self.G_cap_center = None  # For spherical caps
        self.G_cap_width = None
        self.G_g1 = None
        self.G_g2 = None

        # Initialize the matrices
        Y, W = compute_Y_and_W(self.grid, None, self.order_input, self.order_output, backend=self.backend,
                               w_pattern=self.w_pattern)
        self.Y, self.W = Y.to(self.device), W.to(self.device)
        self.G = self.compute_G(G_type=G_type, G_values=G_values)
        self.T_mat = self.compute_T_mat()

    def __repr__(self):
        rep = "DirectionalLoudness with: \n"
        rep += f'Device = {self.device} \n'
        rep += f'order_input = {self.order_input} \n'
        rep += f'order_output = {self.order_output} \n'
        rep += f'backend = {self.backend} \n'
        rep += f'w_pattern = {self.w_pattern} \n'
        rep += f'n_directions = {self.n_directions} \n'
        rep += f'G_type = {self.G_type} \n'
        rep += f'Use_slepian = {self.use_slepian} \n'
        rep += f'Spherical_cap params: \n\t{self.G_cap_center}\n, \t{self.G_cap_width}\n, \t{self.G_g1}\n, \t{self.G_g2}'

        return rep

    def reset_R(self, rotation_angles_rad: Tuple[float, float, float] = None):
        if rotation_angles_rad is not None:
            tmp = utils.get_rotation_matrix(*self.rotation_angles_rad)
            self.R = tmp.to(self.device)
        else:
            self.R = None

    def reset_G(self, G_type: str = 'identity',
                G_values: Union[np.ndarray, torch.Tensor] = None,
                capsule_center: Optional[torch.Tensor] = None,
                capsule_width: Optional[float] = np.pi / 2,
                g1_db: Optional[float] = 0,
                g2_db: Optional[float] = -10):

        tmp = self.compute_G(G_type=G_type,
                             G_values=G_values,
                             capsule_center=capsule_center,
                             capsule_width=capsule_width,
                             g1_db=g1_db,
                             g2_db=g2_db)
        self.G = tmp
        self.G_type = G_type
        self.T_mat = self.compute_T_mat()

    def compute_G(self, G_type: str = 'identity',
                  G_values: Union[np.ndarray, torch.Tensor] = None,
                  capsule_center: Optional[torch.Tensor] = None,
                  capsule_width: Optional[float] = np.pi / 2,
                  g1_db: Optional[float] = 0,
                  g2_db: Optional[float] = -10) -> torch.Tensor:
        """
        Returns a matrix G with the gains for each direction.
        Currently supports only this types:
            -- Identity matrix.
            -- Random diagonal.
            -- Fixed values (set diagonal to a vector of values)
            -- Random matrix (including values outside the diagonal)
            -- spherical_cap - Equation 3.18 of the Ambisonics book.
            """
        G = np.eye(self.n_directions)

        if G_type == 'identity':
            pass

        elif G_type == 'random_diag':
            values = np.random.rand(self.n_directions)
            np.fill_diagonal(G, values)

        elif G_type == 'fixed':
            values = G_values
            np.fill_diagonal(G, values)

        elif G_type == 'random':
            G = np.random.rand(self.n_directions, self.n_directions)

        elif fnmatch.fnmatch(G_type, 'spherical_cap*'):
            G = torch.eye(self.n_directions, device=self.device)

            # Draw a random spherical caps from the predetermined hyperparameter sets
            if capsule_center is None:
                if fnmatch.fnmatch(G_type, '*_soft'):
                    cap_type = 'soft'
                else:
                    cap_type = 'hard'
                capsule_center, capsule_width, g1_db, g2_db = draw_random_spherical_cap(spherical_cap_type=cap_type,
                                                                                        device=self.device)

            assert capsule_center.shape[-1] == 2, 'The capsule center should be [1, 2] vector of azimuth and elevation.'
            assert capsule_width > 0 and capsule_width < 1 * np.pi, 'The capsule width should be within 0 and 1*pi radians'

            tmpA = utils.sph2unit_vec(capsule_center[:, 0], capsule_center[:, 1]).to(self.device).double()
            tmpB = torch.from_numpy(self.grid).to(self.device).double()
            tmpA = tmpA.repeat(tmpB.shape[0], 1)
            assert tmpA.shape == tmpB.shape, 'Wrong shape for the capsule center or the grid coordinates'

            # Dot product batch wise, between the angles of the capsule and the grid points
            tmp = torch.bmm(tmpA.unsqueeze(dim=1), tmpB.unsqueeze(dim=2)).squeeze(dim=-1).to(self.device)
            g1, g2 = spa.utils.from_db(g1_db), spa.utils.from_db(g2_db)

            values = g1 * torch.heaviside(tmp - torch.cos(capsule_width / 2), torch.ones_like(tmp) * g1) + \
                     g2 * torch.heaviside(torch.cos(capsule_width / 2) - tmp, torch.ones_like(tmp) * g2)
            G.diagonal().copy_(values.squeeze())  # replace diagonal inline

            self.G_cap_center = capsule_center
            self.G_cap_width = capsule_width
            self.G_g1 = g1_db
            self.G_g2 = g2_db

        if isinstance(G, np.ndarray):
            G = torch.from_numpy(G)
            G = G.to(self.device)

        return G.double()

    def apply_slepian_function(self, T_mat: torch.Tensor, g1_db: Optional[float] = 0,
                               g2_db: Optional[float] = -10,
                               alpha: Optional[float] = 0.5) -> torch.Tensor:
        """ Applies an spherical slepian function to the transformation matrix.
        This is mostly useful when the directional matrix G is not identity, as it helps when using spherical
        harmonics in incomplete spheres.
        """

        g1, g2 = spa.utils.from_db(g1_db), spa.utils.from_db(g2_db)
        U, eig, Vh = torch.linalg.svd(T_mat)
        largest_eig = torch.max(eig)

        values = g1 * torch.heaviside(eig - alpha * largest_eig, torch.ones_like(eig) * g1) + \
                 g2 * torch.heaviside(alpha * largest_eig - eig, torch.ones_like(eig) * g2)
        new_G = torch.zeros(values.shape[-1], values.shape[-1], device=self.device).double()
        new_G.diagonal().copy_(values.squeeze())  # replace diagonal inline

        new_T_mat = torch.matmul(torch.matmul(U, new_G), Vh)

        return new_T_mat.double()

    def compute_T_mat(self) -> torch.Tensor:
        """ Computes the full transformation matrix T_mat, and applies the scaling if selected."""
        if False:
            print('Debugging')
            print(self.Y.shape)
            print(self.G.shape)
            print(self.W.shape)
            print(self.Y)
            print(self.G)
            print(self.W)

        tmp = torch.matmul(self.Y.transpose(1, 0), self.G)
        T_mat = torch.matmul(tmp, self.W)
        if self.backend == 'basic':
            scale = 4 * np.pi / self.n_directions  # TODO August 05, this works ok , except for input_order > 1
            T_mat = scale * T_mat

        if self.use_slepian:
            if fnmatch.fnmatch(self.G_type, 'spherical_cap*'):
                T_mat = self.apply_slepian_function(T_mat, self.G_g1, self.G_g2)
            else:
                warnings.warn('WARNING: Slepian Functions are only applied when using spherical caps.')
        return T_mat.double()

    def forward(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()
        if X.shape[-2] > X.shape[-1]:  # Channels first format
            warnings.warn('WARNING: It seems that the input tensor X is NOT in channels-first format')
        if self.W is not None and self.Y is not None:
            assert X.shape[-2] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'
        assert self.T_mat.shape[-1] == X.shape[-2], 'Wrong shape for input signal or matrix T.'

        out = torch.matmul(self.T_mat, X.double())

        return out.float()

    def plot_response(self, plot_channel=0, title=None, plot3d=True, plot2d=True, plot_matrix=False, show_plot=True,
                      do_scaling=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Plots the polar response of the transformation matrix.

            Method can be {'cross', 'linear'} for either a crosssection fo azimuth and elevaiton, or a 2d
            linear projection.

        Returns:
            Tuple of tensors of [azi, azi_response, ele, ele_response] for azimuth and elevation responses.
            """
        if plot_matrix:
            plots.plot_transform_matrix(self.T_mat.detach().cpu(), xlabel='Input ACN', ylabel='Output ACN')

        if plot3d:
            responses_3d = plots.plot_sphere_points(self.T_mat.detach().cpu(), self.order_input, plot_channel=plot_channel,
                                                    show_plot=show_plot, do_scaling=do_scaling)

        if plot2d:
            azis, response_azi, eles, response_ele = plots.plot_matrix2polar(self.T_mat.detach().cpu(), self.order_input,
                                                                             plot_channel=plot_channel, title=title,
                                                                             show_plot=show_plot, do_scaling=do_scaling)
            return azis, response_azi, eles, response_ele

    def plot_W(self):
        import matplotlib.pyplot as plt
        spa.plots.sh_coeffs_subplot(self.W.detach().cpu().numpy())
        plt.show()

    def plot_Y(self):
        import matplotlib.pyplot as plt
        spa.plots.sh_coeffs_subplot(self.Y.detach().cpu().numpy())
        plt.show()

    def plot_T_mat(self):
        import matplotlib.pyplot as plt
        spa.plots.sh_coeffs_subplot(self.T_mat.detach().cpu().numpy())
        plt.show()

    def plot_G(self):
        plots.plot_transform_matrix(self.G, xlabel='Input direction', ylabel='Output direction', title='G_matrix')


def test_directional_loudness():
    """
    This is a test for the Directional_Loudness class. It mostly tests:
    - When G = identity, the transformation matrix T is close to identity.
    - When G = identity, a processed signal has low reconstruction error
    - Wehn
    """
    params = {'t_design_degree': 4,
              'G_type': 'identity',
              'order_output': 1,
              'order_input': 1,
              'w_pattern': 'cardioid'}
    transform = DirectionalLoudness(t_design_degree=params['t_design_degree'],
                                    G_type=params['G_type'],
                                    order_output=params['order_output'],
                                    order_input=params['order_input'])
    sig_in = torch.randn(size=[1, 4, 24000])
    sig_out = transform(sig_in)

    assert torch.all(
        torch.isclose(transform.T_mat, torch.eye(transform.T_mat.shape))), 'T_mat should be close to identity'
    assert torch.all(
        torch.isclose(sig_in, sig_out)), 'Ouput signal should be close to input signal when using identity'

def test_rotation():
    """
    This is a test for the SphericalRotation class. It mostly tests:
    """
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid',
              'fs': 24000}

    rotation_params = {'rot_phi': 0.0,
                       'rot_theta': 0.0,
                       'rot_psi': 0.0}
    rotation_angles = [rotation_params['rot_phi'], rotation_params['rot_theta'], rotation_params['rot_psi']]
    rotation = SphericalRotation(rotation_angles_rad=rotation_angles,
                                 t_design_degree=params['t_design_degree'],
                                 order_output=params['order_output'],
                                 order_input=params['order_input'])
    audio = torch.randn(size=[1, 4, 24000])
    targets = torch.randn(size=[1, 3, 12, 128])

    with torch.no_grad():
        audio_rot, targets_rot = rotation(audio, targets)

    assert torch.all(
        torch.isclose(audio_rot, audio_rot)), 'Ouput signal should be close to input signal when using identity'
    assert torch.all(
        torch.isclose(targets_rot, targets_rot)), 'Ouput signal should be close to input signal when using identity'

    plots.plot_labels(targets[0], title='Original')
    plots.plot_labels(targets_rot[0], title='Rotated')

    # Now with some more realistic data, these is a trayectory in the unitary sphere
    point_start = torch.tensor([1, 1, 0])[None, ...]
    point_end = torch.tensor([-1, 0.5, 0])[None, ...]
    targets = torch.nn.functional.interpolate(torch.stack([point_start, point_end], dim=-1), size=128, mode='linear')
    norm = torch.linalg.norm(targets, ord=2, dim=-2)
    targets = targets.permute([1, 0, 2]) / norm

    rotation_params = {'rot_phi': 0.0,
                       'rot_theta': 0.0,
                       'rot_psi': np.pi/4}
    rotation_angles = [rotation_params['rot_phi'], rotation_params['rot_theta'], rotation_params['rot_psi']]
    rotation = SphericalRotation(rotation_angles_rad=rotation_angles,
                                 t_design_degree=params['t_design_degree'],
                                 order_output=params['order_output'],
                                 order_input=params['order_input'])
    _, targets_rot = rotation(torch.zeros(size=(1, 4, 24000)), targets[None, ...])
    plots.plot_labels(targets, title='Original')
    plots.plot_labels(targets_rot[0], title='Rotated')
    plots.plot_labels_cross_sections(targets, title='Original')
    plots.plot_labels_cross_sections(targets_rot[0], title='Rotated')

    # Now with a lot of random points in the unitary sphere to find the best ranges for the rotations
    n_points, n_frames, n_rotations = 10, 128, 10
    points = torch.randn(size=(n_points, 3))
    points[:, 2] = 0.0  # Manually select one coord to set to zero for analysis, or none

    norms = torch.linalg.vector_norm(points, ord=2, dim=-1)
    points = points / norms.unsqueeze(-1).repeat(1, 3)

    rotation = SphericalRotation(rotation_angles_rad=rotation_angles,
                                 t_design_degree=params['t_design_degree'],
                                 order_output=params['order_output'],
                                 order_input=params['order_input'])
    all_new_points = []
    for ii in range(n_rotations):
        rotation.reset_R()
        _, new_points = rotation(torch.zeros(size=(1, 4, 1000)), points[..., None, None])
        all_new_points.append(new_points)
    all_new_points = torch.cat(all_new_points, dim=0)

    plots.plot_distribution_azi_ele(points, title='Original', gridsize=40, log_scale=False, cmin=1)
    plots.plot_distribution_azi_ele(all_new_points.squeeze(), title='Rotated', gridsize=40, log_scale=False, cmin=1)

if __name__ == '__main__':
    #test_directional_loudness()
    test_rotation()

