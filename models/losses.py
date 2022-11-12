import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

##
# Useful repos:
# https://github.com/salu133445/dan/blob/master/src/dan/losses.py
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
#
##

def discriminator_loss(prediction: torch.Tensor, target_is_real: bool, loss_type: str = 'minmax'):
    target_real = 1.0
    target_fake = 0.0

    if target_is_real:
        target_tensor = torch.tensor(target_real)
    else:
        target_tensor = torch.tensor(target_fake)
    target_tensor = target_tensor.expand_as(prediction).to(prediction.get_device())

    if loss_type == 'minmax':
        fn = nn.BCEWithLogitsLoss().to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'non-sat':
        fn = nn.BCEWithLogitsLoss().to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'ls':
        fn = nn.MSELoss().to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'hinge':
        if target_is_real:
            loss = -torch.mean(torch.min(torch.zeros_like(prediction), prediction - 1))
        else:
            loss = -torch.mean(torch.min(torch.zeros_like(prediction), -prediction - 1))
    elif loss_type == 'wass':
        if target_is_real:
            loss = -torch.mean(prediction)
        else:
            loss = torch.mean(prediction)

    return loss

def generator_loss(prediction, loss_type='minmax'):
    eps = 1e-8

    if loss_type == 'minmax':
        fn = nn.BCEWithLogitsLoss().to(prediction.get_device())
        target_real = 1.0
        target_tensor = torch.tensor(target_real)
        target_tensor = target_tensor.expand_as(prediction).to(prediction.get_device())
        loss = fn(prediction, target_tensor)
    elif loss_type == 'non-sat':
        loss = -torch.mean(torch.log(torch.sigmoid(prediction + eps)))
    elif loss_type == 'ls':
        loss = torch.mean(torch.pow(prediction - 1, 2))
    elif loss_type == 'hinge':
        loss = -torch.mean(prediction)
    elif loss_type == 'wass':
        loss = -torch.mean(prediction)

    return loss

class MSELoss_ADPIT(object):
    """ From the DCASE 2022 task 3 baseline"""
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """

        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss

class AccDoaSpectralLoss(nn.Module):
    """ AccDoaSpectralLoss
    Based on the Multi-resolution STFT loss
    Computes the L1 loss of the STFT at different n_fft sizes, of the ACCDOA vectors
    There are two terms, L1loss of the magnitude SFTT, and L1loss of the log magnitude.
    """
    def __init__(self, n_ffts=(2048, 1024, 512, 256, 128, 64), hop_size=0.75, alpha=1.0, device='cpu', distance='l1',
                 return_breakdown=False):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_size = hop_size
        self.alpha = alpha
        self.return_breakdown = return_breakdown
        if distance == 'l1':
            self.crit = nn.L1Loss()
        else:
            self.crit = nn.MSELoss()

        self.log_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80).to(device)
        self.spec_transforms = [torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                                  win_length=n_fft,
                                                                  hop_length=math.floor(n_fft * self.hop_size),
                                                                  power=1,
                                                                  normalized=True).to(device)
                                for n_fft in n_ffts]

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        assert len(output.shape) == 3 or len(output.shape) == 4, f'Tensors should be [3, n_classes, frames] with or without batch. Shape = {output.shape}'
        assert output.shape == target.shape, 'Tensors should have the same shape.'
        
        if len(output.shape) == 3:
            a = None
            b, c, d = output.shape
            shape = (-1, d)
        else:
            a, b, c, d = output.shape
            shape = (a, -1, d)
        output = torch.reshape(output, shape)
        target = torch.reshape(target, shape)

        losses = []
        for spectrogram in self.spec_transforms:
            if spectrogram.n_fft > output.shape[-1]:
                print(f'Warning: Audio too small for n_fft ({spectrogram.n_fft} > {output.shape[-1]}). Skipping this size.')
                continue

            spec_output = spectrogram(output)
            spec_target = spectrogram(target)
            log_spec_output = self.log_transform(spec_output)
            log_spec_target = self.log_transform(spec_target)

            loss = self.crit(spec_output, spec_target) + self.alpha * self.crit(log_spec_output, log_spec_target)
            losses.append(loss)

        if self.return_breakdown:
            return sum(losses), losses
        return sum(losses)

def test_spectral_loss(accdoa1, accdoa2, criterion):
    # Truncate so that both files have the same timesteps
    max_length = min(accdoa1.shape[-1], accdoa2.shape[-1])
    accdoa1 = accdoa1[..., 0:max_length]
    accdoa2 = accdoa2[..., 0:max_length]

    loss1 = criterion(accdoa1, accdoa1)
    loss2 = criterion(accdoa2, accdoa2)
    loss_mix1 = criterion(accdoa1, accdoa2)
    loss_mix2 = criterion(accdoa2, accdoa1)

    assert np.isclose(loss1, 0), 'Loss for audio 1 is too big'
    assert np.isclose(loss2, 0), 'Loss for audio 2 is too big'

    assert loss1 < loss_mix1, 'Self loss for audio 1 is larger than the loss compared to a different audio'
    assert loss2 < loss_mix2, 'Self loss for audio 2 is larger than the loss compared to a different audio'

    assert np.isclose(loss_mix1, loss_mix2), 'Loss should be symmetric'

    # Testing Batches
    batch = torch.stack((accdoa1, accdoa2))
    loss_batch = criterion(batch, batch)

    assert np.isclose(loss_batch, 0), 'Loss for batch should be close to 0'

    print(f'Loss accdoa1 x accdoa 1 : {loss1}')
    print(f'Loss accdoa2 x accdoa 2 : {loss2}')
    print(f'Loss accdoa1 x accdoa 2 : {loss_mix1}')
    print(f'Loss accdoa2 x accdoa 1 : {loss_mix2}')
    print('Test completed')


if __name__ == '__main__':
    channels, n_classes, frames = 3, 12, 256
    accdoa1 = torch.randn((channels, n_classes, frames))
    accdoa2 = torch.randn((channels, n_classes, frames))

    criterion = AccDoaSpectralLoss(n_ffts=[128,64,32,16,8])
    test_spectral_loss(accdoa1, accdoa2, criterion)

