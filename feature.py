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

class my_feature(nn.Sequential):
    def __init__(self):
        super(my_feature, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft=512,
                                          hop_length=240,
                                          power=None)
        self.eps = 1e-10
        self.clamp_min = -80

    def forward(self, input):
        #tmp = self.stft(input)
        #mag = tmp.abs()
        #phase = tmp.angle()
        #phase_diff = torch.diff(phase, dim=-3)

        with torch.no_grad():
            tmp = self.stft(input)
            mag = 20 * torch.log10(tmp.abs() + self.eps)
            mag = torch.clamp(mag, self.clamp_min)
            mag = mag / mag.abs().max()
            #phase = tmp.angle()
            #phase_diff = torch.cos(torch.diff(phase, dim=-3))
            foa_iv = _get_foa_intensity_vectors_pytorch(tmp)
            output = torch.concat([mag, foa_iv], dim=-3)
            if torch.any(torch.isnan(output)):
                warnings.warn('WARNING: NaNs when computing features.')
        return output

def test_my_feature():
    dataset = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                 list_dataset='dcase2022_devtrain_debug.txt',
                                 chunk_size=int(24000 * 1.27),
                                 chunk_mode='fixed',
                                 trim_wavs=30,
                                 return_fname=False)
    transform = my_feature()

    audio, labels = dataset[0]
    feature = transform(audio[None,...])
    test_plot_mine(feature[0])

    return 0


def test_plot_mine(feature):
    import plots
    import matplotlib.pyplot as plt
    import seaborn as sns
    #plots.plot_specgram(mag, sample_rate=24000)

    fig, ax = plt.subplots(feature.shape[-3], 1, figsize=(12,12))
    for i in range(feature.shape[-3]):
        aa = ax[i].matshow(feature[i, :, :], aspect='auto', origin='lower', cmap='magma')
        fig.colorbar(aa, ax=ax[i], location='right')
    plt.tight_layout()
    plt.show()


def test_plot_plt():
    import matplotlib.pyplot as plt

    x = input[0,:]

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(x)
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=512,Fs=24000, noverlap=240)
    plt.show()


def test_plot_kinda_workds(audio):
    stft = torchaudio.transforms.Spectrogram(n_fft=512,
                                                  hop_length=240,
                                                  power=None)

    tmp = stft(audio)
    mag = 20 * torch.log10(tmp.abs())
    phase = tmp.angle()
    phase_diff = torch.cos(torch.diff(phase, dim=-3))

    foa_iv = _get_foa_intensity_vectors_pytorch(tmp)

    import plots
    import matplotlib.pyplot as plt
    import seaborn as sns

    # plots.plot_specgram(mag, sample_rate=24000)

    fig, ax = plt.subplots(7, 1, figsize=(12, 12))
    aa = ax[0].matshow(mag[0, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[0], location='right')
    aa = ax[1].matshow(mag[1, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[1], location='right')
    aa = ax[2].matshow(mag[2, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[2], location='right')
    aa = ax[3].matshow(mag[3, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[3], location='right')
    aa = ax[4].matshow(foa_iv[0, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[4], location='right')
    aa = ax[5].matshow(foa_iv[1, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[5], location='right')
    aa = ax[6].matshow(foa_iv[2, :, :], aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(aa, ax=ax[6], location='right')
    plt.tight_layout()
    plt.show()

def _get_foa_intensity_vectors(self, linear_spectra):
    """ From the baseline
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
    I_norm_mel = 0
    foa_iv = I_norm
    if torch.any(torch.isnan(foa_iv)):
        print('Feature extraction is generating nan outputs')
        exit()
    return foa_iv

if __name__ == '__main__':
    test_my_feature()  # seems to work ok, but no mels


