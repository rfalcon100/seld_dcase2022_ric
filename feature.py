import warnings

import torch
import torch.nn as nn
import torchaudio
from dataset.dcase_dataset import DCASE_SELD_Dataset


def BasicSTFT():
    spectrogram_transform = nn.Sequential(
        torchaudio.transforms.Spectrogram(n_fft=512,
                                          hop_length=240),
        torchaudio.transforms.AmplitudeToDB())
    return spectrogram_transform


def my_feature():
    spectrogram_transform = nn.Sequential(
        torchaudio.transforms.Spectrogram(n_fft=512,
                                          hop_length=240,
                                          power=None),
        torchaudio.transforms.AmplitudeToDB())

    return spectrogram_transform

class Feature_StftPlusIV(nn.Sequential):
    def __init__(self, nfft=512):
        super(Feature_StftPlusIV, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                          hop_length=240,
                                          power=None)
        self.eps = 1e-10
        self.clamp_min = -80

    def forward(self, input):
        #tmp = self.stft(input)
        #mag = tmp.abs()
        #phase = tmp.angle()
        #phase_diff = torch.diff(phase, dim=-3)

        tmp = self.stft(input)
        mag = 20 * torch.log10(tmp.abs() + self.eps)
        mag = torch.clamp(mag, self.clamp_min)
        mag = mag / mag.abs().max()
        #phase = tmp.angle()
        #phase_diff = torch.cos(torch.diff(phase, dim=-3))
        foa_iv = _get_foa_intensity_vectors_pytorch(tmp)
        output = torch.concat([mag, foa_iv], dim=-3)
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            warnings.warn('WARNING: NaNs or INFs when computing features.')
        return output


class Feature_MelPlusPhase(nn.Sequential):
    """ Mel Spectoram with interchannel phase differencefeatures"""
    def __init__(self, normalize_specs=True, n_mels=86, nfft=1024):
        super(Feature_MelPlusPhase, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                          hop_length=240,
                                          power=None)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=24000, f_min=0.0, f_max=None, n_stft=nfft // 2 + 1, norm=None)
        self.eps = 1e-10
        self.clamp_min = -80
        self.normalize_specs = normalize_specs

    def forward(self, input):
        tmp = self.stft(input)
        mag = tmp.abs()
        mag = self.mel_scale(mag)
        div = torch.amax(mag, dim=(-3, -2, -1), keepdim=True)  # Singla max across all channels, for each sample in the batch
        mag = 20 * torch.log10((mag + self.eps) / div)
        mag = torch.clamp(mag, self.clamp_min)  # [-80, 0] range, in dB
        if self.normalize_specs:
            t = torch.tensor(self.clamp_min)
            mag = mag / t.abs()  # [-1, 0] range
            mag = mag * 2 + 1  # [-1, 1] range

        phase = tmp.angle()
        phase = self.mel_scale(phase)
        phase_diff = torch.cos(torch.diff(phase, dim=-3))
        output = torch.concat([mag, phase_diff], dim=-3)
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            warnings.warn('WARNING: NaNs or INFs when computing features.')
        return output

class Feature_MelPlusIV(nn.Sequential):
    """ Mel Spectoram with interchannel phase difference features"""
    def __init__(self, normalize_specs=True, n_mels=86, nfft=1024):
        super(Feature_MelPlusIV, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=nfft,
                                          hop_length=240,
                                          power=None)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=24000, f_min=0.0, f_max=None, n_stft=nfft // 2 + 1, norm=None)
        self.eps = 1e-10
        self.clamp_min = -80
        self.normalize_specs = normalize_specs

    def forward(self, input):
        tmp = self.stft(input)
        mag = tmp.abs()
        mag = self.mel_scale(mag)
        div = torch.amax(mag, dim=(-3, -2, -1), keepdim=True)  # Singla max across all channels, for each sample in the batch
        mag = 20 * torch.log10(mag / (div+ self.eps))
        mag = torch.clamp(mag, self.clamp_min)  # [-80, 0] range, in dB
        if self.normalize_specs:
            t = torch.tensor(self.clamp_min)
            mag = mag / t.abs()  # [-1, 0] range
            mag = mag * 2 + 1  # [-1, 1] range

        foa_iv = _get_foa_intensity_vectors_pytorch(tmp)
        foa_iv = self.mel_scale(foa_iv)
        if self.normalize_specs:
            div = torch.amax(foa_iv, dim=(-3, -2, -1), keepdim=True)  # Singla max across all channels, for each sample in the batch
            foa_iv = foa_iv / (div + self.eps)
        output = torch.concat([mag, foa_iv], dim=-3)
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            warnings.warn('WARNING: NaNs or INFs when computing features.')
        return output

def test_my_features():
    import plots
    dataset = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                 list_dataset='dcase2022_devtrain_debug.txt',
                                 chunk_size=int(24000 * 1.27),
                                 chunk_mode='fixed',
                                 trim_wavs=30,
                                 return_fname=False)
    audio, labels = dataset[1]
    plots.plot_labels(labels)

    # Feature_StftPlusIV
    transform = Feature_StftPlusIV()
    feature = transform(audio[None, ...])
    plots.plot_stft_features(feature[0], title='Feature_StftPlusIV')

    # Feature_MelPlusPhase
    transform = Feature_MelPlusPhase()
    audio, labels = dataset[1]
    feature = transform(audio[None, ...])
    plots.plot_stft_features(feature[0], title='Feature_MelPlusPhase')

    # Feature_MelPlusIV
    transform = Feature_MelPlusIV()
    audio, labels = dataset[1]
    feature = transform(audio[None, ...])
    plots.plot_stft_features(feature[0], title='Feature_MelPlusIV')

    # Torchaudio melspec to compare my melspec implementation
    tform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=24000,
                                             n_fft=512,
                                             hop_length=240,
                                             n_mels=64),
        torchaudio.transforms.AmplitudeToDB())
    feature = tform(audio[None, ...])
    torch.max(feature.permute(1, 0, 2, 3).reshape(4, -1), dim=1)  # Get the max of each channel, for noramlization
    plots.plot_stft_features(feature[0], share_colorbar=False, title='TorchaudioMelSpec')

    return 0



def test_spec_augment():
    import plots
    dataset = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                 list_dataset='dcase2022_devtrain_debug.txt',
                                 chunk_size=int(24000 * 1.27),
                                 chunk_mode='fixed',
                                 trim_wavs=30,
                                 return_fname=False)
    audio, labels = dataset[1]
    plots.plot_labels(labels)

    augment0 = nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=24, iid_masks=True, p=0.2),
    )

    augment1 = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=10, iid_masks=True)
    )


    # Feature_StftPlusIV
    transform = Feature_StftPlusIV()
    feature = transform(audio[None, ...])
    feature = augment0(feature)
    plots.plot_stft_features(feature[0], title='Feature_StftPlusIV')

    # Feature_StftPlusIV
    transform = Feature_StftPlusIV()
    feature = transform(audio[None, ...])
    feature = augment1(feature)
    plots.plot_stft_features(feature[0], title='Feature_StftPlusIV')

    return 0

def test_plot_plt():
    import matplotlib.pyplot as plt

    x = input[0,:]

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(x)
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=512,Fs=24000, noverlap=240)
    plt.show()



def _get_foa_intensity_vectors(self, linear_spectra):
    """ From the baseline, copied as refrerence
        #Input is [frames, freqs, channels]
    # I is [frames, freqs, 3], so all the other channels
    """
    W = linear_spectra[:, :, 0]
    I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
    E = self._eps + (np.abs(W) ** 2 + ((np.abs(linear_spectra[:, :, 1:]) ** 2).sum(-1)) / 3.0)

    I_norm = I / E[:, :, np.newaxis]
    I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), self._mel_wts), (0, 2, 1))
    foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
    if np.isnan(foa_iv).any():
        print('Feature extraction is generating nan outputs')
        exit()
    return foa_iv

def _get_foa_intensity_vectors_pytorch(linear_spectra):
    eps = 1e-8
    W = linear_spectra[..., 0, :, :]
    I = torch.real(torch.conj(W)[:, None, ...] * linear_spectra[..., 1:, :, :])
    E = eps + (W.abs()**2 + ((linear_spectra[..., 1:, :, :].abs()**2).sum(dim=-3)) / 3.0)

    I_norm = I / E[:, None, ...]
    foa_iv = I_norm
    if torch.any(torch.isnan(foa_iv)):
        print('Feature extraction is generating nan outputs')
        exit()
    return foa_iv

if __name__ == '__main__':
    test_my_features()  # seems to work ok, but no mels


