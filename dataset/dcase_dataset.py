
from typing import Iterable, Tuple, TypeVar, Callable, Any, List, Dict, Union
import math
import numpy as np
import os.path
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import pandas as pd

import plots
from utils import validate_audio

# Useful references for the dataloading using iterable datasets:
# https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
# https://discuss.pytorch.org/t/example-for-torch-utils-data-iterabledataset/101175/13
# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

# https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567/4

def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                ele_rad = tmp_val[3] * np.pi / 180.
                azi_rad = tmp_val[2] * np.pi / 180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
    return out_dict

def _read_audio(fname: str, directory_root: str, resampler: Union[torch.nn.Sequential, None], trim_seconds: int = -1) -> Tuple[torch.Tensor, int, float]:
    ''' trim_seconds = to limit how many seconds to load '''
    fpath = os.path.join(directory_root,
                         fname)
    metadata = torchaudio.info(fpath)
    num_frames = trim_seconds if trim_seconds == -1 else trim_seconds * metadata.sample_rate
    this_audio, fs = torchaudio.load(fpath, num_frames=num_frames)
    duration_seconds = this_audio.shape[-1] / fs
    assert validate_audio(this_audio), f'ERROR: {fname}  audio is not valid.'

    if resampler is not None:
        this_audio = resampler(this_audio)

    return torch.tensor(this_audio, dtype=torch.float), fs, duration_seconds

def _read_time_array(fname: str, directory_root: str) -> List:
    ''' Time arrays are the full list of events for a whole audio file.
    This is before any parsing'''
    fpath = os.path.join(directory_root,
                         fname)
    fpath_csv = fpath.replace('mic', 'metadata').replace('foa', 'metadata').replace('wav', 'csv')
    this_time_array = pd.read_csv(fpath_csv, header=None).values

    return this_time_array

def load_output_format_file(fname: str, directory_root: str):
    """
    Adapted from the official baseline.
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """

    fpath = os.path.join(directory_root,
                         fname)
    fpath_csv = fpath.replace('mic', 'metadata').replace('foa', 'metadata').replace('wav', 'csv')

    _output_dict = {}
    _fid = open(fpath_csv, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #polar coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def _add_rotated_label_each_frame(label, time_array4frame_event, start_frame, rotation_pattern=None):
    """ From Sony """
    event_class = time_array4frame_event[1]

    azi_rad = time_array4frame_event[3] / 180 * np.pi
    ele_rad = time_array4frame_event[4] / 180 * np.pi
    if rotation_pattern:
        azi_reflection, azi_rotation, ele_reflection = rotation_pattern
    else:
        azi_reflection, azi_rotation, ele_reflection = [1, 0, 1]  # if None, no rotation
    rotated_azi_rad = azi_reflection * azi_rad + azi_rotation
    rotated_ele_rad = ele_reflection * ele_rad
    x_axis = 1 * np.cos(rotated_ele_rad) * np.cos(rotated_azi_rad)
    y_axis = 1 * np.cos(rotated_ele_rad) * np.sin(rotated_azi_rad)
    z_axis = 1 * np.sin(rotated_ele_rad)

    label[0, event_class, start_frame: start_frame + 10] = x_axis
    label[1, event_class, start_frame: start_frame + 10] = y_axis
    label[2, event_class, start_frame: start_frame + 10] = z_axis

    return (label)

def _get_labels(time_array, start_sec, fs, chunk_size_audio, rotation_pattern=None, multi_track=False, num_classes=12):
    """
    [frame number (int)], [active class index (int)], [track number index (int)], [azimuth (int)], [elevation (int)]
    Frame, class, and track enumeration begins at 0. Frames correspond to a temporal resolution of 100msec.
    Azimuth and elevation angles are given in degrees, rounded to the closest integer value, with azimuth and
    elevation being zero at the front, azimuth ϕ∈[−180∘,180∘], and elevation θ∈[−90∘,90∘]. Note that the azimuth
    angle is increasing counter-clockwise (ϕ=90∘ at the left).
    """

    # This 100 is the sampling frequency of the labels
    # And the 10 for index_diff stuff, is the desired sampling frequency, to match the spectrograms.
    # So the spectrograms use a step_size = 240, with fs = 24000, which is 10 ms
    # Therefore, here they have 100 / 10 = 10
    # My intuition is that a different step_size, would require to change this

    # TODO Is this really ok? Needs verification
    num_axis = 3  # X, Y, Z
    num_class = num_classes
    num_frame = round(chunk_size_audio / fs * 100) + 1  # Each frame == 100 ms (0.1 seconds)
    label = np.zeros([num_axis, num_class, num_frame])

    end_sec = start_sec + chunk_size_audio / fs

    index_diff = int(math.modf(start_sec * 10)[0] * 10)  # get second decimal place
    num_frame_wide = (int(np.ceil(end_sec * 10)) - int(np.floor(start_sec * 10)) + 1) * 10
    # "+ 1" is buffer for numerical error, such as index_diff=3 and num_frame_wide=130

    if not multi_track:
        label_wide = np.zeros([num_axis, num_class, num_frame_wide])
        for index, frame in enumerate(range(int(np.floor(start_sec * 10)), int(np.ceil(end_sec * 10)))):
            time_array4frame = time_array[time_array[:, 0] == frame]
            if time_array4frame.shape == (1, 5):
                label_wide = _add_rotated_label_each_frame(label_wide, time_array4frame[0], index * 10,
                                                           rotation_pattern)
            elif time_array4frame.shape == (2, 5):
                label_wide = _add_rotated_label_each_frame(label_wide, time_array4frame[0], index * 10,
                                                           rotation_pattern)
                label_wide = _add_rotated_label_each_frame(label_wide, time_array4frame[1], index * 10,
                                                           rotation_pattern)
            elif time_array4frame.shape == (3, 5):
                label_wide = _add_rotated_label_each_frame(label_wide, time_array4frame[0], index * 10,
                                                           rotation_pattern)
                label_wide = _add_rotated_label_each_frame(label_wide, time_array4frame[1], index * 10,
                                                           rotation_pattern)
                label_wide = _add_rotated_label_each_frame(label_wide, time_array4frame[2], index * 10,
                                                           rotation_pattern)
        label = label_wide[:, :, index_diff: index_diff + num_frame]
    else:
        # TODO This is not ready
        label_wide_1 = np.zeros([num_axis, num_class, num_frame_wide])
        label_wide_2 = np.zeros([num_axis, num_class, num_frame_wide])
        label_wide_3 = np.zeros([num_axis, num_class, num_frame_wide])
        for index, frame in enumerate(range(int(np.floor(start_sec * 10)), int(np.ceil(end_sec * 10)))):
            time_array4frame = time_array[time_array[:, 0] == frame]
            if time_array4frame.shape == (1, 5):
                label_wide_1 = _add_rotated_label_each_frame(label_wide_1, time_array4frame[0], index * 10,
                                                             rotation_pattern)
            elif time_array4frame.shape == (2, 5):
                label_wide_1 = _add_rotated_label_each_frame(label_wide_1, time_array4frame[0], index * 10,
                                                             rotation_pattern)
                label_wide_2 = _add_rotated_label_each_frame(label_wide_2, time_array4frame[1], index * 10,
                                                             rotation_pattern)
            elif time_array4frame.shape == (3, 5):
                label_wide_1 = _add_rotated_label_each_frame(label_wide_1, time_array4frame[0], index * 10,
                                                             rotation_pattern)
                label_wide_2 = _add_rotated_label_each_frame(label_wide_2, time_array4frame[1], index * 10,
                                                             rotation_pattern)
                label_wide_3 = _add_rotated_label_each_frame(label_wide_3, time_array4frame[2], index * 10,
                                                             rotation_pattern)
        label = np.stack((
            label_wide_1[:, :, index_diff: index_diff + num_frame],
            label_wide_2[:, :, index_diff: index_diff + num_frame],
            label_wide_3[:, :, index_diff: index_diff + num_frame]
        ))
    return (label)

def _read_fnames(directory_root: str, list_dataset: str) -> List:
    """Reads the fnames in the list_dataset.
    Each fname corresponds to a single wav file in the dataset.
    This to prepare the dataset, before loading any audio or labels."""
    fnames = []
    fpath = os.path.join(directory_root,
                         'list_dataset',
                         list_dataset)
    for fname in pd.read_table(fpath, header=None).values.tolist():
        if isinstance(fname, List): fname = fname[0]

        parent_dir = directory_root.split('/')[-1] + '/'
        if parent_dir in fname:
            fname = fname.replace(parent_dir, '')
        fnames.append(fname)
    return fnames

def get_adpit_labels_for_file(_desc_file: Dict, _nb_label_frames: int, num_classes: int = 13) -> np.ndarray:
    """
    
    ADAPATED from csl_feature_class from the baseline, with modifications to remove the dependcy to the class.

    Reads description file and returns classification based SED labels and regression based DOA labels
    for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

    :param _desc_file: metadata description file
    :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
    """

    se_label = np.zeros((_nb_label_frames, 6, num_classes))  # [nb_frames, 6, max_classes]
    x_label = np.zeros((_nb_label_frames, 6, num_classes))
    y_label = np.zeros((_nb_label_frames, 6, num_classes))
    z_label = np.zeros((_nb_label_frames, 6, num_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
            active_event_list_per_class = []
            for i, active_event in enumerate(active_event_list):
                active_event_list_per_class.append(active_event)
                if i == len(active_event_list) - 1:  # if the last
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]

                elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                    active_event_list_per_class = []

    label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]
    return label_mat

def _random_slice(audio: torch.Tensor, fs: int, chunk_size_audio: float, trim_wavs: int, clip_length_seconds: int = 60) \
        -> Tuple[torch.Tensor, int]:
    """Returns a random slice of an audio and the corresponding starting time in sencods (useful to extract labels) """

    # Now we do it in seconds
    if trim_wavs > 0:
        star_min_sec, start_max_sec = 2, math.floor(trim_wavs - (chunk_size_audio/fs + 2))
    else:
        star_min_sec, start_max_sec = 0, math.floor(clip_length_seconds - chunk_size_audio/fs)
    start_sec = np.round(np.random.randint(star_min_sec,
                                           min((audio.shape[-1] - chunk_size_audio / 2) / fs, start_max_sec),
                                           1))
    start_index = start_sec * fs
    sliced_audio = audio[:, start_index[0]: start_index[0] + round(chunk_size_audio)]
    return sliced_audio, start_sec

def _fixed_slice(audio: torch.Tensor, fs: int, chunk_size_audio: float) -> Tuple[torch.Tensor, int]:
    """Returns a fixed slice of an audio and its corresponding time array (label)"""
    start_sec = 5  # Hardcoded start at 5 seconds
    start_sample = start_sec * fs

    sliced_audio = audio[:, start_sample : int(start_sample + chunk_size_audio)]
    return sliced_audio, start_sec


class InfiniteDataLoader(DataLoader):
    ''' DataLoader that keeps returning batches even after the dataset is exhausted.
    Useful when the __getitem__ of the dataset returns a random slice.

    Ref:
    https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class DCASE_SELD_Dataset(Dataset):
    """Dataset for the DCASE SELD (Task3), supports version 2021 and 2022.

    This dataset first loads all the audio and labels to memory.
    In the getitem, it returns a slice, from wavs.
    This dataset is a map dataset, so each "epoch" will see each wav file only once.
    But the slice of each wav can be randomly selected.
    The audios and labels are stored in memory, but the slices are computed at runtime.

    Parameters:
        directory_root - Path to the the directory that contains 'foa_dev', 'metadata', 'list_dataset'
        list_dataset - File with the wav filenames that we want to load. Filesnames are relative to directory_root.
        trim_wavs - Trim wavs to this number of seconds when loading audio, so we load shorter wavs.
        chunk_size - Size of the chunkds (slices) returned in getitem. In samples.
        chuck_mode - {'full', 'fixed', 'random'} Where the getitem:
            - Full - Returns the full wav and labels. Useful for validation, and to compute statistics.
            - Fixed - Returns a slice at fixed start time of each wav. Useful for debugging.
            - Random - Returns a random slice each time.
        return_fname - Returns fname during the getitem
        multi_track - Enables multi-track ACCDOA for the labels
    """
    def __init__(self,
                 directory_root: str = './data/',
                 list_dataset: str = 'dcase2021t3_foa_overfit_vrgpu.txt',
                 trim_wavs: float = -1,  # in seconds
                 chunk_size: int = 48000,  # in samples
                 chunk_mode: str = 'fixed',
                 return_fname: bool = False,
                 multi_track: bool = False,
                 num_classes: int = 13,
                 ignore_labels: bool = False,
                 labels_backend: str = 'sony'):
        super().__init__()
        self.directory_root = directory_root
        self.list_dataset = list_dataset  # list of wav filenames  , e.g. './data_dcase2021_task3/foa_dev/dev-val/fold5_room1_mix001.wav'
        self.chunk_size_audio = chunk_size
        self.chunk_mode = chunk_mode
        self.trim_wavs = trim_wavs  # Trims the inputs wavs to the selected length in seconds
        self.return_fname = return_fname
        self.multi_track = multi_track
        self.num_classes = num_classes
        self.ignore_labels = ignore_labels  # This is to avoid loading labels. Useful when doing evaluation.
        self.labels_backend = labels_backend  # Code to use when extracting labels from CSVs. For multitrack, we need the baseline. {'sony', 'backend'}
        self.resampler = None

        if self.multi_track and self.labels_backend == 'sony':
            warnings.warn('WARNING: When using multi-track labels, we should use the baseline back end.')
        self._fnames = []
        self._audios = {}
        self.durations = {}
        self._fs = {}  # Per wav
        self._time_array_dict = {}  # Per wav

        # Load full wavs and time_arrays to memory
        self._fnames = _read_fnames(directory_root=self.directory_root, list_dataset=self.list_dataset)
        for fname in self._fnames:
            audio, fs, duration = _read_audio(fname=fname, directory_root=self.directory_root,
                                              resampler=self.resampler, trim_seconds=self.trim_wavs)
            if not self.ignore_labels:
                if self.labels_backend == 'sony':
                    time_array = _read_time_array(fname=fname, directory_root=self.directory_root)
                else:
                    time_array = load_output_format_file(fname=fname, directory_root=self.directory_root)
                    time_array = convert_output_format_polar_to_cartesian(time_array)
                    time_array = get_adpit_labels_for_file(_desc_file=time_array, _nb_label_frames=math.ceil(duration * 100),
                                                                   num_classes=self.num_classes)
                self._time_array_dict[fname] = time_array
            self._audios[fname] = audio
            self._fs[fname] = fs
            self.durations[fname] = duration

        self.__validate()
        print(self)

    def __validate(self):
        assert len(self._fnames) == len(self._audios), 'Fnames and audios should have the same count'
        assert len(self._fnames) == len(self.durations), 'Fnames and durations should have the same count'
        assert len(self._fnames) == len(self._fs), 'Fnames and fs should have the same count'
        if not self.ignore_labels:
            assert len(self._fnames) == len(self._time_array_dict), 'Fnames and time_arrays should have the same count'

    def __len__(self):
        return len(self._fnames)

    def get_fnames(self):
        return self._fnames

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of unique wav files : {}\n'.format(len(self._fnames))
        fmt_str += '    Root Location: {}\n'.format(self.directory_root)
        fmt_str += '    List of files: {}\n'.format(self.list_dataset)
        fmt_str += '    Chunk size: {}\n'.format(self.chunk_size_audio)
        fmt_str += '    Chunk Mode: {}\n'.format(self.chunk_mode)
        fmt_str += '    Trim audio: {}\n'.format(self.trim_wavs)
        fmt_str += '    Multi_track: {}\n'.format(self.multi_track)
        fmt_str += '    Ignore labels: {}\n'.format(self.ignore_labels)
        return fmt_str

    def __getitem__(self, item):
        fname = self._fnames[item]
        audio = self._audios[fname]
        fs = self._fs[fname]
        duration = self.durations[fname]
        if not self.ignore_labels:
            time_array = self._time_array_dict[fname]
        else:
            time_array = None

        # Select a slice
        if self.chunk_mode == 'fixed':
            audio, start_sec = _fixed_slice(audio, fs, chunk_size_audio=self.chunk_size_audio)
            labels_duration = self.chunk_size_audio
        elif self.chunk_mode == 'random':
            audio, start_sec = _random_slice(audio, fs, chunk_size_audio=self.chunk_size_audio, trim_wavs=self.trim_wavs, clip_length_seconds=duration)
            labels_duration = self.chunk_size_audio
        elif self.chunk_mode == 'full':
            start_sec = 0
            labels_duration = audio.shape[-1]
        if not self.ignore_labels:
            if self.labels_backend == 'sony':
                label = _get_labels(time_array, start_sec=start_sec, fs=fs, chunk_size_audio=labels_duration, rotation_pattern=None,
                                    multi_track=self.multi_track, num_classes=self.num_classes)
            else:
                if not self.multi_track:
                    raise NotImplementedError
                # TODO Hardcoded fs for laels at 100 ms
                start_frame = int(start_sec) * 10
                end_frame = start_frame + math.ceil(labels_duration / fs * 100) + 1
                #label = get_adpit_labels_for_file(_desc_file=time_array, _nb_label_frames=math.ceil(duration * 10), num_classes=self.num_classes)

                if end_frame > time_array.shape[0]:
                    label = np.concatenate([time_array, np.zeros([end_frame - start_frame - time_array.shape[0], *time_array.shape[1:]])], axis=0)
                else:
                    label = time_array[start_frame: end_frame, ...]
                if label.shape[0] < end_frame - start_frame:
                    label = np.concatenate([label, np.zeros([end_frame - start_frame - label.shape[0], *label.shape[1:]])], axis=0)
        else:
            label = np.empty(1)

        if self.return_fname:
            return audio, torch.from_numpy(label.astype(np.float32)), fname
        else:
            return audio, torch.from_numpy(label.astype(np.float32))


def test_dataset_train_iteration(num_iters=100, batch_size=32, num_workers=4):
    # Here we test a typical train iteration, with the map dataset, but with infinite dataloader
    # The main idea is that we dont have epochs, but iterations.
    # This supports batching, and multiple workers
    # This looks OK, each "epoch" samples each wavs only once, but with infinite dataloader we itierate foreacher
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import islice
    dataset_train = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                       list_dataset='dcase2022_devtrain_all.txt',
                                       chunk_size=int(24000 * 1.27),
                                       chunk_mode='random',
                                       trim_wavs=-1,
                                       return_fname=True)
    loader_train = InfiniteDataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    # Begin iterating
    ctr = 0
    ctr_fnames = {}
    for (x_batch, y_batch, fnames) in islice(loader_train, num_iters):
        if ctr < 5:
            print(f'iter: {ctr}')
            print(f'x_batch.shape: {x_batch.shape}')
            print(f'y_batch.shape: {y_batch.shape}')
            print(torch.mean(x_batch, dim=(-1, -2)))

        for fname in fnames:
            if fname in ctr_fnames:
                ctr_fnames[fname] += 1
            else:
                ctr_fnames[fname] = 1
        ctr += 1
        if ctr > 5:
            break

    # Display counter of how many times each wav was sliced
    print(f'There are {len(ctr_fnames)} unique fnames.')
    f, ax = plt.subplots(figsize=(10, 15))
    df = pd.DataFrame(list(ctr_fnames.items()))
    df.columns = ['fname', 'count']
    sns.barplot(x="count", y="fname", data=df,
                label="count", color="b")
    sns.despine(left=True, bottom=True)
    plt.show()

    # Display wav durations
    f, ax = plt.subplots(figsize=(10, 15))
    df = pd.DataFrame(list(dataset_train.durations.items()))
    df.columns = ['fname', 'duration']
    sns.barplot(x="duration", y="fname", data=df,
                label="duration", color="b")
    sns.despine(left=True, bottom=True)
    plt.show()


def _get_padders(chunk_size_seconds: float = 1.27,
                 duration_seconds: float = 60.0,
                 overlap: float = 0.5,
                 audio_fs=24000, labels_fs=100):
    # Wavs:
    fs = audio_fs
    audio_full_size = fs * duration_seconds
    audio_chunk_size = math.ceil(fs * chunk_size_seconds)
    audio_pad_size = math.ceil(audio_full_size / audio_chunk_size) + math.ceil(audio_fs / labels_fs / overlap)
    audio_padder = nn.ConstantPad1d(padding=(0, audio_pad_size), value=0.0)
    audio_step_size = math.floor(audio_chunk_size * overlap)

    # Labels:
    labels_fs = labels_fs  # 100 --> 10 ms
    labels_full_size = labels_fs * duration_seconds
    labels_chunk_size = math.ceil(labels_fs * chunk_size_seconds) + 1
    labels_pad_size = math.ceil(labels_full_size / labels_chunk_size) + 1
    labels_padder = nn.ConstantPad2d(padding=(0, labels_pad_size, 0, 0), value=0.0)
    labels_step_size = math.floor(labels_chunk_size * overlap)

    audio_padding = {'padder': audio_padder,
                     'chunk_size': audio_chunk_size,
                     'hop_size': audio_step_size}
    labels_padding = {'padder': labels_padder,
                     'chunk_size': labels_chunk_size,
                     'hop_size': labels_step_size}
    return audio_padding, labels_padding

def test_validation_clean():
    # Here I am testing how to do the validation
    # The idea is that I want to iterate the full wavs, to get the predictions
    # So we get full length audio and labels from the dataset
    # Then we split into chunks manually
    # And iterate over wavs, using a dataloader for each one
    # Other useful function, torch.chunks, torch.split

    batch_size = 32  # This depends on GPU memory
    dataset = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                 list_dataset='dcase2022_devtrain_all.txt',
                                 chunk_mode='full',
                                 trim_wavs=-1,
                                 return_fname=True)

    spec = torchaudio.transforms.Spectrogram(
        n_fft=512,
        win_length=512,
        hop_length=240,
    )

    all_labels = []
    print(f'Iterating {len(dataset)} fnames in dataset.')
    for i in range(len(dataset)):
        # Analyze audio in full size
        audio, labels, fname = dataset[i]
        duration = dataset.durations[fname]
        all_labels.append(labels)

        print(f'Full audio:')
        print(audio.shape)
        print(f'Full spec:')
        print(spec(audio).shape)
        print(f'Full labels:')
        print(labels.shape)

        audio_padding, labels_padding = _get_padders(chunk_size_seconds=1.27,
                                                     duration_seconds=math.floor(duration),
                                                     overlap=1,   # Other values e.g. 32/128 are ok,
                                                     audio_fs=24000,
                                                     labels_fs=100)

        # To process audio in GPU, split into chunks (that can be overlapped)
        audio = audio_padding['padder'](audio)
        audio_chunks = audio.unfold(dimension=1, size=audio_padding['chunk_size'],
                                    step=audio_padding['hop_size']).permute((1, 0, 2))
        labels = labels_padding['padder'](labels)
        labels_chunks = labels.unfold(dimension=-1, size=labels_padding['chunk_size'],
                                      step=labels_padding['hop_size']).permute((2,0,1,3))

        print(f'Full padded audio:')
        print(audio.shape)
        print(f'Full padded labels:')
        print(labels.shape)

        tmp = torch.utils.data.TensorDataset(audio_chunks, labels_chunks)
        loader = DataLoader(tmp, batch_size=batch_size, shuffle=False, drop_last=False)  # Loader per wav to get batches
        for ctr, (audio, labels) in enumerate(loader):
            print(f'Processing batch {ctr}')

            outo = spec(audio)
            print(f'Audio shape = {audio.shape}')
            print(f'Spec shape = {outo.shape}')
            print(f'Labels shape = {labels.shape}')

            assert outo.shape[-1] == labels.shape[-1], \
                'Wrong shapes, the spectrogram and labels should have the same number of frames. Check paddings and step size'

    # Analysis of labels
    count_active_classes(all_labels)

    # breaks in wav 43 or 42 with overlap

def test_validation_histograms():
    # Here I am testing how to do the validation
    # The idea is that I want to iterate the full wavs, to get the predictions
    # So we get full length audio and labels from the dataset
    # Then we split into chunks manually
    # And iterate over wavs, using a dataloader for each one
    # Other useful function, torch.chunks, torch.split

    # Update 15.06.2022
    # This is very useful to analyze tbe labels too.

    batch_size = 32  # This depends on GPU memory
    dataset = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                 list_dataset='dcase2022_devtrain_all.txt',
                                 chunk_mode='full',
                                 trim_wavs=-1,
                                 return_fname=True,
                                 num_classes=13,
                                 multi_track=False)

    spec = torchaudio.transforms.Spectrogram(
        n_fft=512,
        win_length=512,
        hop_length=240,
    )

    all_labels = []
    print(f'Iterating {len(dataset)} fnames in dataset.')
    for i in range(len(dataset)):
        # Analyze audio in full size
        audio, labels, fname = dataset[i]
        all_labels.append(labels)

    # Analysis of labels
    count_active_classes(all_labels)
    count_active_classes(all_labels[0:1])

def count_active_classes(all_labels: List, detection_threshold=0.5):
    """ Useful function to get the histogram of active classes per frames.
    Tip: Call it with only one label to get the plot.
        count_active_classes(all_labels[0:1])
    """
    import plots
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(all_labels) == 1:
        plots.plot_labels_cross_sections(all_labels[0], n_classes=list(range(all_labels[0].shape[-2])), plot_cartesian=True)
        plots.plot_labels(all_labels[0], n_classes=list(range(all_labels[0].shape[-2])), savefig=False, plot_cartesian=True)

    all_count_detections = {}
    for i in range(len(all_labels)):
        this_label = all_labels[i]
        vec_norms = torch.linalg.vector_norm(this_label, ord=2, dim=-3)

        for cls in range(this_label.shape[-2]):
            mask_detected_events = vec_norms[cls, :] > detection_threshold  # detected events for this class
            # mask_detected_events = mask_detected_events.repeat(1, 3, 1)
            tmp_events = this_label[..., cls, mask_detected_events]
            # detections = tmp_events[mask_detected_events]
            this_count_detections = mask_detected_events.nonzero(as_tuple=False)
            if cls in all_count_detections.keys():
                all_count_detections[cls] += len(this_count_detections)
            else:
                all_count_detections[cls] = len(this_count_detections)

    f, ax = plt.subplots(figsize=(10, 15))
    df = pd.DataFrame(list(all_count_detections.items()))
    df.columns = ['class_id', 'count']
    g = sns.barplot(x="class_id", y="count", data=df,
                    label="class_id", color="b")
    sns.despine(left=False, bottom=False)
    #g.set_yscale("log")
    plt.show()

def test_multi_track():
    """ HEre I should test (manually):
    - chunk_mode: {'fixed', 'random', 'full'}
    - multi_track: True, False
    - labels_backend: {'sony', 'baseline'}
    """
    dataset = DCASE_SELD_Dataset(directory_root='/m/triton/scratch/work/falconr1/sony/data_dcase2022',
                                 list_dataset='dcase2022_devtrain_debug.txt',
                                 chunk_mode='full',
                                 chunk_size=30480,
                                 trim_wavs=-1,
                                 return_fname=True,
                                 num_classes=13,
                                 multi_track=True,
                                 labels_backend='baseline')  # test sony and baseline
    audio, labels, fname = dataset[0]

    if len(labels.shape) > 3:
        this_label = labels[2]
    else:
        this_label = labels
    plots.plot_labels(this_label)

    raise ValueError
        # This sitll fails when using full wavs, and backend baseline
        # the size is not correct, I guess it is cropping somewhere
    # note that the vanilla multitrack puts all the activity in the first track

    a = 1


if __name__ == '__main__':
    from utils import seed_everything
    seed_everything(1234, mode='balanced')
    test_multi_track()
    test_validation_histograms()
    test_dataset_train_iteration()  # OK, I am happy
    test_validation_clean()  # seems ok, except when using overlaps
    print('End of test')


