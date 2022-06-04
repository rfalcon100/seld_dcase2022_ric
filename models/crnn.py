import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class CRNN(nn.Module):
    '''
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    '''
    def __init__(self,
                 in_channels=4,
                 n_class=12):
        super(CRNN, self).__init__()

        self.spec_bn = nn.BatchNorm2d(in_channels)

        # CNN
        self.layer1 = Conv_2d(in_channels, 64, pooling=(2,2))
        self.layer2 = Conv_2d(64, 128, pooling=(3,3))
        self.layer3 = Conv_2d(128, 128, pooling=(4,4))
        self.layer4 = Conv_2d(128, 128, pooling=(4,4))

        # RNN
        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, n_class)

    def forward(self, x):
        # Spectrogram
        x = self.spec_bn(x)

        # CCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        # x.shape = [2, 1, 128]
        x, _ = self.layer5(x)
        # x.shape = [2, 1, 32]
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.GroupNorm(out_channels, out_channels)
        self.bn2 = nn.GroupNorm(out_channels, out_channels)


        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.bn1)
        init_layer(self.bn2)

    def forward(self, x, pool_type='avg', pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'frac':
            fractional_maxpool2d = nn.FractionalMaxPool2d(kernel_size=pool_size, output_ratio=1/np.sqrt(2))
            x = fractional_maxpool2d(x)

        return x


class CRNN10(nn.Module):
    def __init__(self, class_num, in_channels, out_channels=1, pool_type='avg',
                 pool_size=(2, 2), interp_ratio=16, pretrained_path=None, sigmoid=False, multi_track=False):
        super().__init__()

        self.class_num = class_num
        self.out_channels = out_channels
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = interp_ratio

        self.sigmoid = sigmoid
        self.multi_track = multi_track

        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # input_size : the number of expected features
        # hidden_size : hidden features in the gru module
        # input should be : (b, seq_len, H_in)
        self.gru = nn.GRU(input_size=512, hidden_size=256,
                          num_layers=2, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(512, class_num, bias=True)
        if self.multi_track:
            self.xyz_fc = nn.Linear(512, 3 * 3 * class_num, bias=True)
        else:
            self.xyz_fc = nn.Linear(512, 3 * class_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.xyz_fc)

    def forward(self, x):
        x = x.transpose(2, 3)
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        b, c, t, f = x.size()

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1, 2)
        ''' (batch_size, time_steps, feature_maps):'''

        # x = [2, 8, 512]
        #self.gru.flatten_parameters()  # Maybe this is neede when using DataParallel?
        (x, _) = self.gru(x)

        # x.shape = [2, 8, 512]
        if self.out_channels == 1:
            event_output = self.event_fc(x)
            if self.sigmoid:
                event_output = torch.sigmoid(event_output)
            '''(batch_size, time_steps, 1 * class_num)'''
        elif self.out_channels == 3:
            event_output = self.xyz_fc(x)
            '''(batch_size, time_steps, 3 * class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)

        event_output = event_output.transpose(1, 2)
        if self.multi_track:
            event_output = event_output.view(-1, 3, self.out_channels, self.class_num, t)
        else:
            event_output = event_output.view(-1, self.out_channels, self.class_num, t)

        return event_output


def interpolate(x, ratio):
    '''
    Interpolate the x to have equal time steps as targets
    Input:
        x: (batch_size, time_steps, class_num)
    Output:
        out: (batch_size, time_steps*ratio, class_num)
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)

    return upsampled


def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


def init_gru(rnn):
    """Initialize a GRU layer. """
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)



