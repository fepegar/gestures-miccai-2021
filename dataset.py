import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import sglob


class SudepDataset(torch.utils.data.Dataset):
    def filter_paths(self, root_dir, subject_and_seizure_ids, suffix='.mp4'):
        paths_dict = {}
        if subject_and_seizure_ids is None:
            fps = sglob(root_dir, f'**/*{suffix}')
            if not fps:
                raise FileNotFoundError(f'No files with suffix {suffix} found in {root_dir}')
            paths_dict['all'] = fps
        elif subject_and_seizure_ids is not None:
            for ssid, discard in subject_and_seizure_ids:
                if discard: continue
                video_stem = ssid[:-4]  # Large -> L, Small -> S
                fps = sglob(root_dir, f'{video_stem}/*.mp4')
                if not fps:
                    raise FileNotFoundError(f'No files found for {video_stem}')
                paths_dict[video_stem] = fps
        return paths_dict


class FeaturesSequencesDataset(SudepDataset):
    """
    Get N feature vectors from a seizure, distributed according to the jitter_mode value
    """
    def __init__(
            self,
            root_dir='/home/fernando/sudep/sudep_dataset/',
            frames_per_clip=8,  # this defines the minimum length
            frame_rate=15,
            frames_between_clips=1,
            subject_and_seizure_ids=None,
            num_segments=32,  # this defines the minimum length
            cache_path=None,
            force=False,
            num_debug_seizures=None,
            jitter_mode='normal',
            discard=True,
            ):
        super().__init__()
        self.jitter_mode = jitter_mode
        self.root_dir = Path(root_dir)
        self.num_segments = num_segments
        self.frames_per_clip = frames_per_clip
        self.num_debug_seizures = num_debug_seizures
        self.table = DatasetTable()
        self.frame_rate = frame_rate
        features_dir = self.root_dir / f'features_fpc_{frames_per_clip}_fps_{frame_rate}'
        self.seizure_dirs = []
        if subject_and_seizure_ids is None:
            self.seizure_dirs = sglob(features_dir, '*')
        else:
            for pnt_szr_camera, discard_this in subject_and_seizure_ids:
                if discard_this and discard: continue
                pnt_szr_cam = pnt_szr_camera[:-4]  # 321_14_Large -> 321_14_L
                self.seizure_dirs.append(features_dir / pnt_szr_cam)

        if cache_path is None:
            self.data = self.load()
        else:
            cache_path = Path(cache_path)
            if cache_path.is_file():
                state = torch.load(cache_path)
                self.data = state['data']
                self.seizure_dirs = state['seizures_dirs']
            else:
                self.data = self.load()
                state = dict(data=self.data, seizures_dirs=self.seizure_dirs)
                torch.save(state, cache_path)
        if num_debug_seizures is not None:
            self.seizure_dirs = self.seizure_dirs[:num_debug_seizures]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seizure_dir = self.seizure_dirs[index]
        pnt_szr_cam = seizure_dir.name
        ssid = '_'.join(pnt_szr_cam.split('_')[:2])
        all_features_tensor = self.data[pnt_szr_cam]
        num_vectors = len(all_features_tensor)
        chunks = get_chunks(
            num_vectors,
            self.frames_per_clip,
            self.num_segments,
        )
        jittered_indices = self.get_jittered_indices(
            chunks,
            self.frames_per_clip,
            self.jitter_mode,
            seizure_dir=seizure_dir,
        )
        rows = [all_features_tensor[i] for i in jittered_indices]
        sequence = torch.stack(rows)  # shape (num_segments, num_features)
        sample = dict(
            sequence=sequence,
            pnt_szr_cam=pnt_szr_cam,
            indices=torch.Tensor(jittered_indices),
            gtcs=self.table.is_gtcs(ssid),
        )
        return sample

    @staticmethod
    def get_jittered_indices(chunks, frames_per_clip, mode, seizure_dir=None):
        if mode == 1:
            mode = 'uniform'
        elif mode == 'delta':
            mode = 'middle'
        else:
            try:
                float(mode)
                concentration = mode
                mode = 'beta'
            except ValueError:  # probably 'normal', left for bw compatibility
                pass

        jittered_indices = []
        for indices_in_chunk in chunks:
            if len(indices_in_chunk) <= frames_per_clip:
                message = (
                    f'Chunk length ({len(indices_in_chunk)}) is less than'
                    f' the number of frames per snippet ({frames_per_clip})'
                )
                if seizure_dir is not None:
                    message += f'. Seizure dir: {seizure_dir}'
                raise RuntimeError(message)
            first_possible = int(indices_in_chunk[0])
            last_possible = int(indices_in_chunk[-1] - frames_per_clip + 1)
            if mode == 'uniform':
                index = torch.randint(first_possible, last_possible + 1, (1,))
                index = index.item()
            elif mode == 'normal':
                mean = (last_possible + first_possible) / 2
                indices_range = last_possible - first_possible
                # Values less than three standard deviations from the mean
                # account for 99.73% of the total
                std = indices_range / 6
                index = mean + std * torch.randn(1)
                # We still need to clip for that 0.27%
                index = index.round()
                index = torch.clamp(index, first_possible, last_possible).int()
                index = index.item()
            elif mode == 'middle':
                index = int(round((last_possible + first_possible) / 2))
            elif mode == 'beta':
                m = torch.distributions.beta.Beta(
                    concentration,
                    concentration,
                )
                sample = m.sample()
                sample *= last_possible - first_possible
                sample += first_possible
                index = sample.round().long().item()
            jittered_indices.append(index)
        return jittered_indices

    def load(self):
        seizure_dirs = self.seizure_dirs
        if self.num_debug_seizures is not None:
            seizure_dirs = seizure_dirs[:self.num_debug_seizures]
        progress = tqdm(seizure_dirs, leave=False)
        vectors = {}
        for seizure_dir in progress:
            pnt_szr_cam = self.get_pnt_szr_cam_from_features_dir(seizure_dir)
            progress.set_description(pnt_szr_cam)
            tensor_list = []
            paths = sglob(seizure_dir, '*.pth')
            if not paths:
                raise RuntimeError(f'No .pth files found in {seizure_dir}')
            for feature_path in tqdm(paths, leave=False):
                frame_features = torch.load(feature_path)
                tensor_list.append(frame_features)
            vectors[pnt_szr_cam] = torch.stack(tensor_list)
        return vectors

    def plot_sample(self, sample, out_path=None, show=False):
        pnt_szr_cam = sample['pnt_szr_cam']
        indices = sample['indices']
        times = [i / self.frame_rate for i in indices]
        ssid = '_'.join(pnt_szr_cam.split('_')[:2])
        duration = self.table.get_duration(ssid)
        fig, ax = plt.subplots()
        # Mark segments
        for i in range(self.num_segments + 1):
            if i > 0:
                time = (i - 0.5) * duration / self.num_segments
                ax.axvline(time, linewidth=0.5, alpha=0.5)
            time = i * duration / self.num_segments
            ax.axvline(time, linewidth=2)

        for time in times:
            ax.axvline(time, color=(0.8, 0.2, 0.4), linewidth=0.75)
        if sample['gtcs']:
            gtcs_time = self.table.get_gtcs_time(ssid)
            ax.axvline(gtcs_time, linewidth=3, color=(0.8, 0.1, 0.8))
        plt.title(pnt_szr_cam)
        plt.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=400)
        if show:
            plt.show()

    def get_ssid_from_features_dir(self, features_dir):
        # dir/features/061_05_Large -> 061_05
        pnt_szr_camera = features_dir.name
        ssid = '_'.join(pnt_szr_camera.split('_')[:2])
        return ssid

    def get_pnt_szr_cam_from_features_dir(self, features_dir):
        # dir/features/061_05_L -> 061_05_L
        pnt_szr_cam = features_dir.name
        return pnt_szr_cam

    def get_pnt_szr_cam_from_pnt_szr_camera(self, pnt_szr_camera):
        # 061_05_Large -> 061_05_L
        return pnt_szr_camera[:-4]

    def get_gtcs_ratio(self):
        gtcs = 0
        for sample in self:
            gtcs += int(sample['gtcs'])
        return gtcs / len(self)


def get_seizure_path_(subject, seizure):
    return Path('~/sudep/crops').expanduser() / f'{subject}_{seizure}.mp4'


def get_chunks(num_vectors, frames_per_clip, num_chunks):
    num_frames = num_vectors + frames_per_clip - 1  # frames covered by the feature vectors
    all_indices = np.arange(num_frames)
    chunks = np.array_split(all_indices, num_chunks)
    return chunks


class DatasetTable:
    def __init__(self, root_dir='~/sudep/sudep_dataset'):
        root_dir = Path(root_dir).expanduser()
        self.path = root_dir / 'seizures.csv'
        self.df = pd.read_csv(
            self.path, index_col=0, dtype={'Subject': str, 'Seizure': str})

    def get_gtcs_time(self, ssid):
        subject, seizure = ssid.split('_')[:2]
        row = self.df.query(f'Subject == "{subject}" & Seizure == "{seizure}"')
        ratio = row.RatioGTCS.values[0]
        if pd.isna(ratio):
            time = np.inf
        else:
            time = ratio * row.Duration.values[0]
        return time

    def get_duration(self, ssid):
        subject, seizure = ssid.split('_')[:2]
        row = self.df.query(f'Subject == "{subject}" & Seizure == "{seizure}"')
        return row.Duration.values[0]

    def is_gtcs(self, ssid):
        return self.get_gtcs_time(ssid) < np.inf


def sample_beta(chunks, concentration, frames_per_clip):
    indices = []
    m = torch.distributions.beta.Beta(
        concentration,
        concentration,
    )
    for chunk in chunks:
        first_possible = chunk[0]
        last_possible = chunk[-1] - frames_per_clip
        diff = last_possible - first_possible
        sample = m.sample()
        sample *= diff
        sample += first_possible
        index = sample.round().long().item()
        indices.append(index)
    return indices
