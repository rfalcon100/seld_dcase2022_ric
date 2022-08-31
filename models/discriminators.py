import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import warnings

from torch import Tensor
from torch.nn.utils import spectral_norm

from negative_data_aug import low_pass
from torchinfo import summary


class PassLayer(nn.Module):
    """ Flexible layer that applies the specified function to the input tensor.

    For example, to pass:
    normalization_layer = PassLayer(lambda x: x)
    """

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CoordConvLayer(nn.Conv2d):
    # Based on the CoordConv paper
    # This adds positional encodings as extra channels
    # x = columns from 0 to w, normalized from -1 to 1
    # y = rows from 0 to h, normalized from -1 to 1
    # r (optional) = sqrt(x^2 + y^2)  (so this like a diagonal matrix)
    def __init__(self, *args, with_r=False, **kwargs):
        self.with_r = with_r
        extra_channels = 3 if self.with_r else 2
        kwargs['in_channels'] = kwargs['in_channels'] + extra_channels  # Concat channels for the coordconv layer
        super(CoordConvLayer, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        # Assuming input is [batch, channels, F, frames]  (like a 2d image)

        tmp_x = torch.arange(start=0, end=input.shape[-1], device=input.device)
        tmp_x = 2 * tmp_x / tmp_x.max() - 1  # range [-1, 1]
        x = torch.tile(tmp_x.unsqueeze(0), dims=(input.shape[-2],1)) # [F, frames]
        x = torch.tile(x.unsqueeze(0).unsqueeze(0), dims=(input.shape[0],1,1,1))  # [batch, 1, F, frames]

        tmp_y = torch.arange(start=0, end=input.shape[-2], device=input.device)
        tmp_y = 2 * tmp_y / tmp_y.max() - 1  # range [-1, 1]
        y = torch.tile(tmp_y.unsqueeze(1), dims=(1, input.shape[-1]))  # [F, frames]
        y = torch.tile(y.unsqueeze(0).unsqueeze(0), dims=(input.shape[0], 1, 1, 1))  # [batch, 1, F, frames]

        input = torch.cat([input, y, x], dim=-3)  # Concatenate over channels

        if self.with_r:
            tmp_r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
            input = torch.cat([input, tmp_r], dim=-3)
        return super(CoordConvLayer, self).forward(input=input)


def get_activation(activation='none'):
    """ Returns an activation function as nn.module"""
    if activation == 'sigmoid':
        activation_module = nn.Sigmoid()
    elif activation == 'tanh':
        activation_module = nn.Tanh()
    elif activation == 'relu':
        activation_module = nn.ReLU()
    elif activation == 'none':
        activation_module = PassLayer(lambda x: x)
    else:
        raise ValueError('Wrong activation layer for the discriminator.')

    return activation_module


def get_norm_layer(out_channels, normalization='batch', n_groups=None):
    """ Returns a normalization layer. Based on:
    https://towardsdatascience.com/different-normalization-layers-in-deep-learning-1a7214ff71d6
    https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8
    """
    if normalization == 'none':
        normalization_layer = PassLayer(lambda x: x)  # PassThrough
    if normalization == 'batch':
        # Normalizes each feature across the minibatch
        normalization_layer = nn.BatchNorm2d(out_channels)
    if normalization == 'layer':
        # Normalizes each feature across channels
        normalization_layer = nn.GroupNorm(1, out_channels)  # n_groups = 1 --> Layer norm
    elif normalization == 'instance':
        # Normalizes each feature across features of each observation (similar to x/ x.abs().max() in 1d audio signals, I think...)
        normalization_layer = nn.GroupNorm(out_channels, out_channels)  # n_groups = n_channels --> Instance norm
    elif normalization == 'group':
        # Normalizes across groups of channels
        if n_groups is None:
            n_groups = int(out_channels / 2)
        normalization_layer = nn.GroupNorm(n_groups, out_channels)

    return normalization_layer


class VectorNormBinarizer(nn.Module):
    """Binarizes the input by setting to 0.0 everything where the vector norm is less than threshold_min,
    and sets to 1 everything that is between threhsold_min and threshold_max """
    def __init__(self, min_threshold=0.0, max_threshold=1000.0):
        super(VectorNormBinarizer, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def forward(self, x):
        assert len(x.shape) == 3 or len(x.shape) == 4, 'Wrong size for tensor, this only works for 3d or 4d tensors for now'
        if x.shape[-3] > 3:
            raise ValueError('WARNING: There are more than 3 channels in the inputs, so the norm is computed across all channels.')
        with torch.no_grad():
            norm = torch.linalg.vector_norm(x, ord=2, dim=-3)
            mask_min = (norm < self.min_threshold)
            mask_max = (norm > self.max_threshold)
            mask_or = torch.logical_or(mask_min, mask_max).unsqueeze(-3).repeat(1, x.shape[-3], 1, 1)
            x[mask_or] = 0.0
            #mask_between = torch.logical_not(mask_or)
            tmp = norm.unsqueeze(-3).repeat(1, x.shape[-3], 1, 1)
            x = x / (tmp + 1e-10)

            #print(f'Values under vector norm threshold = {torch.sum(mask)}')
        return x


class VectorNormThreshold(nn.Module):
    """Sets to 0.0 everything with vector norm under the min_threshold,
    and to unitary vector times the max_output everything with vector norm above max_threshold"""
    def __init__(self, min_threshold=0.0, max_threshold=1000.0, max_output=1.0):
        super(VectorNormThreshold, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_output = max_threshold   # TODO change this

    def forward(self, x):
        assert len(x.shape) == 3 or len(x.shape) == 4, 'Wrong size for tensor, this only works for 3d or 4d tensors for now'
        if x.shape[-3] > 3:
            warnings.warn('WARNING: There are more than 3 channels in the inputs, so the norm is computed across the first 3 channels.')
        with torch.no_grad():
            norm = torch.linalg.vector_norm(x[..., 0:3, :, :], ord=2, dim=-3)
            mask_min = (norm < self.min_threshold).unsqueeze(-3).repeat(1, x.shape[-3], 1, 1)
            x[mask_min] = 0.0
            mask_max = (norm > self.max_threshold).unsqueeze(-3).repeat(1, x.shape[-3], 1, 1)
            tmp = norm.unsqueeze(-3).repeat(1, x.shape[-3], 1, 1)
            tmp = (self.max_output * x) / (tmp + 1e-10)
            x[mask_max] = tmp[mask_max]

            #print(f'Values under vector norm threshold = {torch.sum(mask)}')
        return x


class CustomThreshold(nn.Module):
    """ Sets to 0.0 everything outside the range determined by the threshold. """
    def __init__(self, min_threshold=0.0, max_threshold=1000.0):
        super(CustomThreshold, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def forward(self, x: torch.Tensor):
        mask_min = torch.abs(x) < self.min_threshold
        mask_max = torch.abs(x) > self.max_threshold
        mask = torch.logical_or(mask_min, mask_max)
        x[mask] = 0.0
        return x


class DiscriminatorModularThreshold(nn.Module):
    """ Modular discriminator with threshold operator to mimic the outputs of prediction
    where the input is a [3, n_classes, n_frames] tensor of ACCDOA labels.
    The default values for kernels and strides is good for inputs (3,12,128)
    """
    def __init__(self, input_shape=(3, 12, 128), n_feature_maps=64, last_layer_multiplier=8,
                 kernels=[(4,4), (2,4), (4,4), (4,4)],
                 strides=[(2,2), (2,2), (2,2), (2,2)],
                 padding=[(1), (1), (1), (1)],
                 freq_pooling='none', use_spectral_norm=False, with_r=False, use_low_pass=False,
                 threshold_min=None, threshold_max=None, use_threshold_norm=False, use_threshold_binarize=False,
                 final_activation='sigmoid', normalization='batch', block='basic', conditioning='none'):
        super(DiscriminatorModularThreshold, self).__init__()
        self.input_shape = input_shape
        self.ndf = n_feature_maps
        self.final_activation = final_activation
        self.normalization = normalization
        self.block = block
        self.conditioning = conditioning
        self.multi = last_layer_multiplier  # this is hack to match the number of channels x last layer dimensions
        self.use_spectral_norm = use_spectral_norm
        self.threshold_min = threshold_min  # If positive value, used as min_threshold
        self.threshold_max = threshold_max  # If positive value, used as max_threshold
        self.with_r = with_r  # Enables or disables the r coordinate for CoordConv blocks
        self.use_threshold_norm = use_threshold_norm
        self.use_threshold_binarize = use_threshold_binarize
        self.use_low_pass = use_low_pass
        self.freq_pooling = freq_pooling

        assert not (self.use_threshold_norm and self.use_threshold_binarize), 'Cannot use vector norm threshold and binarization at the same time'
        assert len(kernels) == len(strides), 'The number of kernels and stride should be the same.'
        modules = []
        in_channels = self.input_shape[0]
        out_channels = self.ndf

        if self.threshold_min is not None and self.threshold_max is not None:
            if self.use_threshold_norm:
                modules.append(VectorNormThreshold(min_threshold=self.threshold_min, max_threshold=self.threshold_max))
            elif self.use_threshold_binarize:
                modules.append(VectorNormBinarizer(min_threshold=self.threshold_min, max_threshold=self.threshold_max))
            else:
                modules.append(CustomThreshold(min_threshold=self.threshold_min, max_threshold=self.threshold_max))
        for ii in range(len(kernels)):
            modules.append(DiscBlock(in_channels=in_channels, out_channels=out_channels,
                                     kernel=kernels[ii], stride=strides[ii], padding=padding[ii],
                                     block_type=self.block, normalization=normalization,
                                     use_spectral_norm=self.use_spectral_norm, with_r=self.with_r))
            in_channels = out_channels
            out_channels = out_channels * 2
            #print(f'Out_channels = {out_channels}')
        self.main = nn.Sequential(*modules)

        activation = get_activation(self.final_activation)
        if activation is not None:
            self.last = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * self.multi, 1, bias=True),
                activation,
            )
        else:
            self.last = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * self.multi, in_channels * self.multi * 2, bias=True),
                nn.ReLU(),
                nn.Linear(in_channels * self.multi * 2, 1, bias=True),
            )

        # Initialize weights
        for m in self.main.modules():
            weights_init(m)

    def forward(self, input):
        if self.use_low_pass:
            with torch.no_grad():
                if input.shape[-3] > 3:  # For the conditional case, filter only the x,y,z channels
                    tmp = low_pass(input[..., 0:3, :, :])
                    input[..., 0:3, :, :] = tmp
                else:
                    input = low_pass(input)
        out = self.main(input)

        if self.freq_pooling != 'none':
            if self.freq_pooling == 'mean':
                out = torch.mean(out, dim=2, keepdim=True)
            elif self.freq_pooling == 'max':
                (out, _) = torch.max(out, dim=2, keepdim=True)

        #print(f'Yolo = {out.shape}')  # for debugging
        out = self.last(out)
        return out

    def __repr__(self):
        tmp = super(DiscriminatorModularThreshold, self).__repr__()
        fmt_str = 'Discriminator Model ' + self.__class__.__name__ + '\n'
        fmt_str += f'    input_shape = {self.input_shape}\n'
        fmt_str += f'    feature_maps = {self.ndf}\n'
        fmt_str += f'    final_activation = {self.final_activation}\n'
        fmt_str += f'    normalization = {self.normalization}\n'
        fmt_str += f'    block = {self.block}\n'
        fmt_str += f'    with_r = {self.with_r}\n'
        fmt_str += f'    conditioning = {self.conditioning}\n'
        fmt_str += f'    multiplier = {self.multi}\n'
        fmt_str += f'    spectral_norm = {self.use_spectral_norm}\n'
        fmt_str += f'    min_threshold = {self.threshold_min}\n'
        fmt_str += f'    max_threshold = {self.threshold_max}\n'
        fmt_str += f'    use_threshold_norm = {self.use_threshold_norm}\n'
        fmt_str += f'    use_threshold_binarize = {self.use_threshold_binarize}\n'
        fmt_str += f'    use_low_pass = {self.use_low_pass}\n'
        fmt_str += f'    freq_pooling = {self.freq_pooling}\n'
        fmt_str += '\n\n'
        fmt_str += tmp
        return fmt_str

    def update_threshold(self, new_min_threshold, new_max_threshold):
        """ Updates the threshold values, useful for curriculum learning."""
        self.threshold_min = new_min_threshold
        self.threshold_max = new_max_threshold
        self.main[0].max_threshold = self.threshold_max
        self.main[0].min_threshold = self.threshold_min


class DiscBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=1, with_r=False,
                 block_type='basic', normalization='batch', activation='lrelu', n_groups=None,
                 use_spectral_norm=False):
        super(DiscBlock, self).__init__()

        if activation == 'relu':
            activation_layer = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'celu':
            activation_layer = nn.CELU(inplace=True)

        normalization_layer = get_norm_layer(out_channels=out_channels, normalization=normalization, n_groups=n_groups)

        if block_type == 'basic':
            tmp = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel, stride=stride, padding=padding, dilation=(1,1), bias=False)
            if use_spectral_norm:
                tmp = spectral_norm(tmp)
            self.block = nn.Sequential(
                tmp,
                normalization_layer,
                activation_layer,
            )
        elif block_type == 'coord':
            tmp = CoordConvLayer(in_channels=in_channels, out_channels=out_channels, with_r=with_r,
                                 kernel_size=kernel, stride=stride, padding=padding, dilation=(1, 1), bias=False)
            if use_spectral_norm:
                tmp = spectral_norm(tmp)
            self.block = nn.Sequential(
                tmp,
                normalization_layer,
                activation_layer,
            )
        else:
            raise ValueError('Unsupported block for the discriminator.')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp = self.block(input)
        #print(tmp.shape)
        return tmp



def test_discriminator(model='basic'):
    if model == 'basic':
        net = DiscriminatorBasic(n_feature_maps=32)
    if model == 'basicSN':
        net = DiscriminatorBasicSN(n_feature_maps=32)
    if model == 'basic_no-sigmoid':
        net = DiscriminatorBasic(n_feature_maps=32, final_activation='none')
    if model == 'basicThreshold':
        net = DiscriminatorBasicThreshold(n_feature_maps=32, final_activation='none')

    input = torch.rand((10,3,12,128))
    output = net(input)
    print(output.shape)

    print(f'Input shape: {input.shape}')
    summary(net, input_size=(3, 12, 128))


def quick_test():
    input = torch.rand((10, 5))
    layer = nn.Threshold(0.5, 10)
    output = layer(input)
    input_check = torch.any(input < 0.5)
    outout_check = torch.any(output < 0.5)
    print(input_check)
    print(outout_check)
    print(torch.allclose(input, output))
    print(input)
    print(output)

def test_modular_refactored():
    # First my basic model, from GANtestbed
    print('Using my version from GANtestbed')
    input = torch.rand((10, 7, 96, 128))  # 4+3 channels, 96 mel spec bings
    conditioning = 'concat'
    net = DiscriminatorModularThreshold(input_shape=input.shape[1::], n_feature_maps=32, final_activation='none',
                                        kernels=[(4, 4), (4, 4), (4, 4), (4, 4)],
                                        strides=[(2, 2), (2, 2), (3, 2), (4, 4)],
                                        padding=[(1), (1), (1), (1)],
                                        normalization='instance', block='basic', conditioning=conditioning)
    output = net(input)
    print(f'Input shape: {input.shape}')
    print(f'Output shape: {output.shape}')
    summary(net, input_size=input.shape[1::])

    # Second, basic model from sony code, using STFT
    print('Using my version from sony adversarial')
    input = torch.rand((10, 3, 257, 128))  # 4+3+3 channels, 512 fft size
    config = {
        "network": "modular",
        "disc_block": "coord",
        "input_shape": input.shape[1::],
        "conditioning": "none",
        "disc_feature_maps": 64,
        "disc_normalization": "instance",
        "disc_final": "none",
        "use_spectral_norm": True,
        "threshold_min": 0.3,
        "threshold_max": 1.2,
        "use_threshold_norm": True,
        "with_r": True,
        "disc_kernels": [(4,4), (4,4), (4,4), (4,4), (2,2)],
        "disc_strides": [(2,2), (2,2), (4,2), (4,4), (2,2)],
        "disc_padding": [(1), (1), (1), (1), (0)]}
    net = DiscriminatorModularThreshold(input_shape=config['input_shape'], n_feature_maps=config['disc_feature_maps'],
                                        final_activation=config['disc_final'],
                                        kernels=config['disc_kernels'],
                                        strides=config['disc_strides'],
                                        padding=config['disc_padding'],
                                        last_layer_multiplier=4,
                                        threshold_min=config['threshold_min'],
                                        threshold_max=config['threshold_max'],
                                        use_threshold_norm=config['use_threshold_norm'],
                                        normalization=config['disc_normalization'], block=config['disc_block'],
                                        conditioning=config['conditioning'],
                                        use_spectral_norm=config['use_spectral_norm'])
    output = net(input)
    print(net)
    print(f'Input shape: {input.shape}')
    print(f'Output shape: {output.shape}')
    summary(net, input_size=input.shape[1::])

    print('End of test')

def test_binarize():
    layer = VectorNormBinarizer(max_threshold=1)

    x = torch.rand(1, 3, 3, 3)
    print(x)
    y = layer(x)

    print(y)

    print(torch.linalg.vector_norm(x, ord=2, dim=-3))
    print(torch.linalg.vector_norm(y, ord=2, dim=-3))

    assert torch.all(torch.logical_or(
        torch.allclose(y, torch.zeros_like(y)),
        torch.allclose(y, torch.ones_like(y)))), 'Wrong values for norms.'

if __name__ == '__main__':
    #test_discriminator('basic')
    #test_discriminator('basicSN')
    #test_discriminator('basic_no-sigmoid')
    #test_discriminator('basicThreshold')
    #quick_test()

    test_modular_refactored()

