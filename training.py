import logging
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict, deque

import torch
import pandas as pd
from tqdm import tqdm

from utils import sglob


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_num_cpu_cores():
    return mp.cpu_count()

def get_seizure_ids(df):
    sids = []
    for row in df.itertuples():
        sids.append((row.Subject, row.Seizure))
    return sorted(list(set(sids)))

def get_num_files(directory):
    # https://stackoverflow.com/a/8311376/3956024
    import os
    _, _, files = next(os.walk(directory))
    return len(files)

def prepare_df(df, features_dir, frames_per_clip=8, fps=15):
    durations_dict = {}
    for view_dir in tqdm(list(features_dir.iterdir())):
        key = tuple(view_dir.name.split('_')[:2])
        if key in durations_dict: continue
        num_snippets = get_num_files(view_dir)
        duration_one_frame = 1 / fps
        time_last_snippet = (num_snippets - 1) * duration_one_frame
        duration_one_snippet = frames_per_clip * duration_one_frame
        duration = time_last_snippet + duration_one_snippet
        durations_dict[key] = duration
    durations = []
    for row in df.itertuples():
        key = row.Subject, row.Seizure
        try:
            durations.append(durations_dict[key])
        except KeyError:
            logging.warning(f'Features for {key} not found in features directory')
            durations.append(0)
    df['Duration'] = durations
    return df

def get_buckets(
        dataset_dir,
        num_folds,
        min_duration,
        verbose=False,
        ):
    """Put seizures (sorted by duration) in "buckets", one by one. First GTCS, then no GTCS"""
    buckets = defaultdict(list)
    queue = deque(range(num_folds))

    csv_path = dataset_dir / 'seizures.csv'
    df = pd.read_csv(
        csv_path,
        dtype={'Subject': str, 'Seizure': str},
    )

    features_dir = dataset_dir / 'features_fpc_8_fps_15'  # TODO: stop hard-coding this
    df = prepare_df(df, features_dir)
    short = df[df.Duration < min_duration]
    old_df = df
    df = old_df[old_df.Duration >= min_duration]
    if len(short) > 0 and verbose:
        print(
            'Seizures discarded for being shorter than',
            min_duration,
            'seconds:',
        )
        print(short)

    gtcs = df[~df.isna().OnsetClonic].sort_values(by='Duration', ascending=False)
    no_gtcs = df[df.isna().OnsetClonic].sort_values(by='Duration', ascending=False)

    for sdf in gtcs, no_gtcs:
        for subject, seizure in get_seizure_ids(sdf):
            row = sdf.query(
                f'Subject == "{subject}" & Seizure == "{seizure}"'
            )
            discard_large = row.Discard.values[0] in ('Yes', 'Large')
            discard_small = row.Discard.values[0] in ('Yes', 'Small')

            if discard_large and discard_small:
                if verbose:
                    print(f'Discarding {subject} - {seizure} (both views not usable)')
                continue
            index = queue[0]
            vid = f'{subject}_{seizure}_Large'
            buckets[index].append((vid, discard_large))
            vid = f'{subject}_{seizure}_Small'
            buckets[index].append((vid, discard_small))
            queue.rotate(-1)

    for index in buckets:
        buckets[index] = sorted(buckets[index])

    return buckets

def get_fold_split(
        dataset_dir,
        k=None,
        num_folds=10,
        num_holdout_folds=0,
        min_duration=3,
        verbose=False,
        ):
    # E.g. 10 folds, k must be in [0, 7] if num_holdout_folds is 2
    if k >= num_folds - num_holdout_folds:
        raise ValueError(f'Fold {k} is part of the holdout set')

    buckets = get_buckets(
        dataset_dir,
        num_folds=num_folds,
        min_duration=min_duration,
        verbose=verbose,
    )

    validation = buckets[k]
    training = []
    for i in range(num_folds - num_holdout_folds):
        if i != k:
            training.extend(buckets[i])

    test = []
    for i in range(num_folds - num_holdout_folds, num_folds):
        test.extend(buckets[i])

    training = sorted(training)
    validation = sorted(validation)
    test = sorted(test)

    return training, validation, test


def get_checkpoint_path(experiment_version, runs_dir='~/sudep/runs'):
    runs_dir = Path(runs_dir).expanduser()
    experiment_dir = runs_dir / 'lightning_logs' / f'version_{experiment_version}'
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoint_paths = sglob(checkpoints_dir, '*.ckpt')
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    return checkpoint_path


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 1)

    def forward(self, x):
        return self.fc(x)


def get_classifier(checkpoint_path):
    model = Model()
    state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(state_dict)
    return model
