"""
Train LSTM using precomputed clips features.

Command lines generated with:
python ~/git/sudep/scripts/train/generate_tmux_commands_lstm.py > /tmp/run.sh
"""

import time
import logging
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

import pytorch_lightning as pl
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

from models import RecurrentModel, MeanModel
from dataset import FeaturesSequencesDataset
from training import get_num_cpu_cores, get_fold_split

runs_dir = Path(__file__).parent / 'runs'
runs_dir.mkdir(exist_ok=True)

# Create an Experiment instance
ex = Experiment()

file_observer = FileStorageObserver(runs_dir / 'sacred')
ex.observers.append(file_observer)

slack_config_path = Path('~/slack.json').expanduser()
slack_obs = SlackObserver.from_config(str(slack_config_path))
ex.observers.append(slack_obs)

TEMP_DIR = Path(tempfile.gettempdir())


@ex.config
def get_config_data():
    # pylint: disable=unused-variable
    batch_size = 2 ** 6
    batch_size_ratio = 10
    percentage_cores = 25
    num_workers = round(get_num_cpu_cores() * (percentage_cores / 100))
    frames_per_clip = 8
    frame_rate = 15
    num_folds = 10
    num_holdout_folds = 0
    fold = 7
    min_seizure_duration = 15  # use only seizures longer than this
    num_segments = 16


@ex.config
def get_config_optimizer():
    # pylint: disable=unused-variable
    optimizer_name = 'AdamW'
    learning_rate = 1e-2  # 2e-2 found with PT Lightning and 8 frames
    lr = learning_rate  # for LR finder?


@ex.config
def get_config_training():
    # pylint: disable=unused-variable
    gtcs_weight = None
    patience = 100
    monitored_variable, mode = 'val_fscore', 'max'
    hidden_units = 64
    aggregation = 'lstm'

@ex.config
def get_config_system():
    # pylint: disable=unused-variable
    root_dir = Path(__file__).parent / 'dataset'
    experiment_name = str(time.time())
    seed = None


@ex.config
def get_config_testing():
    version_number = -1


@ex.config
def get_trainer_kwargs():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = vars(parser.parse_args([]))
    del parser

    debug = False
    args['fast_dev_run'] = debug

    percent_check = 1
    args['train_percent_check'] = percent_check
    args['val_percent_check'] = percent_check
    args['test_percent_check'] = percent_check

    max_epochs = 400
    args['max_epochs'] = max_epochs

    log_gpu_memory = False
    args['log_gpu_memory'] = log_gpu_memory

    auto_lr_find = False
    args['auto_lr_find'] = auto_lr_find

    auto_scale_batch_size = False
    args['auto_scale_batch_size'] = auto_scale_batch_size

    precision = 32
    args['precision'] = precision

    early_stop_callback = False
    args['early_stop_callback'] = early_stop_callback

    checkpoint_callback = True
    args['checkpoint_callback'] = checkpoint_callback

    gpus = 1
    args['gpus'] = gpus

    auto_select_gpus = gpus > 0
    args['auto_select_gpus'] = auto_select_gpus


class Model(pl.LightningModule):
    @ex.capture
    def __init__(self, hparams, hidden_units):
        super().__init__()
        self.hparams = hparams
        self.model = self.get_model()

    @ex.capture
    def get_model(self, hidden_units, aggregation):
        if aggregation == 'mean':
            model = MeanModel()
        else:
            model = RecurrentModel(
                hidden_size=hidden_units,
                bidirectional=aggregation == 'blstm',
            )
        return model

    def forward(self, x):
        return self.model(x)

    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2484#issuecomment-661277355
    @property
    def batch_size(self):
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.hparams.batch_size = batch_size

    @ex.capture
    def prepare_data(
            self,
            root_dir,
            frames_per_clip,
            frame_rate,
            num_folds,
            fold,
            num_holdout_folds,
            min_seizure_duration,
            num_segments,
            gtcs_weight,
            jitter_mode,
            ):
        dataset_dir = Path(root_dir)
        train_ids, val_ids, test_ids = get_fold_split(
            dataset_dir,
            fold,
            num_folds=num_folds,
            num_holdout_folds=num_holdout_folds,
            min_duration=min_seizure_duration,
        )
        self.train_dataset = FeaturesSequencesDataset(
            root_dir,
            frames_per_clip,
            frame_rate,
            subject_and_seizure_ids=train_ids,
            cache_path=TEMP_DIR / 'dataset_train.pth',
            num_segments=num_segments,
            jitter_mode=jitter_mode,
        )
        print(f'Training dataset: {len(self.train_dataset)} data points')

        self.val_dataset = FeaturesSequencesDataset(
            root_dir,
            frames_per_clip,
            frame_rate,
            subject_and_seizure_ids=val_ids,
            cache_path=TEMP_DIR / 'dataset_val.pth',
            num_segments=num_segments,
            jitter_mode='middle',
        )
        print(f'Validation dataset: {len(self.val_dataset)} data points')

        self.test_dataset = FeaturesSequencesDataset(
            root_dir,
            frames_per_clip,
            frame_rate,
            subject_and_seizure_ids=test_ids,
            cache_path=TEMP_DIR / 'dataset_test.pth',
            num_segments=num_segments,
            jitter_mode='middle',
        )
        print(f'Test dataset: {len(self.test_dataset)} data points')

        if gtcs_weight is None:
            self.gtcs_weight = 1 / self.train_dataset.get_gtcs_ratio()
        else:
            self.gtcs_weight = gtcs_weight
        print('Weight of positive class:', self.gtcs_weight, '\n\n')

    @ex.capture
    def train_dataloader(self, num_workers):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        print(f'Training batches: {len(loader)}\n')
        return loader

    @ex.capture
    def val_dataloader(self, num_workers):
        # pylint: disable=no-value-for-parameter
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.get_validation_batch_size(),
            num_workers=num_workers,
        )
        print(f'Validation batches: {len(loader)}\n')
        return loader

    @ex.capture
    def test_dataloader(self, num_workers):
        # pylint: disable=no-value-for-parameter
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.get_validation_batch_size(),
            num_workers=num_workers,
        )
        print(f'Test batches: {len(loader)}')
        return loader

    @ex.capture
    def get_validation_batch_size(self, batch_size_ratio):
        return batch_size_ratio * self.hparams.batch_size

    @ex.capture
    def configure_optimizers(self, optimizer_name):
        optimizer_class = getattr(torch.optim, optimizer_name)
        return optimizer_class(self.parameters(), lr=self.hparams.lr)

    @ex.capture
    def get_pos_weight(self):
        pos_weight = torch.Tensor((self.gtcs_weight,))
        return pos_weight

    def get_xy(self, batch):
        return batch['features'], batch['gtcs']

    def get_loss(self, logits, y):
        # pylint: disable=no-value-for-parameter
        pos_weight = self.get_pos_weight().type_as(logits)
        y_one_hot = F.one_hot(y.long(), num_classes=2).float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y_one_hot,
            pos_weight=pos_weight,
        )
        return loss

    def get_metrics(self, y, predictions, threshold=0.5):
        y = y.detach().cpu()
        predictions = predictions.detach().cpu()
        predictions = predictions > threshold
        metrics = precision_recall_fscore_support(
            y,
            predictions,
            labels=(0, 1),
            zero_division=1,
        )
        precision, recall, fscore, _ = torch.Tensor(metrics)[:, 1]
        accuracy = accuracy_score(y, predictions)
        accuracy = torch.Tensor((accuracy,))
        return precision, recall, fscore, accuracy

    def training_step(self, batch, batch_index):
        x, y = batch['sequence'], batch['gtcs']
        logits = self(x).squeeze()
        loss = self.get_loss(logits, y)
        tensorboard_logs = dict(train_loss=loss)
        result = dict(loss=loss, log=tensorboard_logs)
        return result

    def validation_step(self, batch, batch_index):
        x, y = batch['sequence'], batch['gtcs']
        logits = self(x).squeeze()
        loss = self.get_loss(logits, y)
        predictions = logits.argmax(dim=1)
        precision, recall, fscore, accuracy = self.get_metrics(y, predictions)
        result = dict(
            val_loss=loss,
            val_precision=precision,
            val_recall=recall,
            val_fscore=fscore,
            val_accuracy=accuracy,
        )
        return result

    def test_step(self, batch, batch_index):
        x, y = batch['sequence'], batch['gtcs']
        logits = self(x).squeeze()
        loss = self.get_loss(logits, y)
        predictions = logits.argmax(dim=1)
        precision, recall, fscore, accuracy = self.get_metrics(y, predictions)
        result = dict(
            test_loss=loss,
            test_precision=precision,
            test_recall=recall,
            test_fscore=fscore,
            test_accuracy=accuracy,
        )
        return result

    @staticmethod
    def stack_mean(outputs, name):
        return torch.stack([x[name] for x in outputs]).mean()

    def validation_epoch_end(self, outputs):
        if not outputs:
            logging.warning('No validation outputs')

        avg_loss = self.stack_mean(outputs, 'val_loss')
        avg_precision = self.stack_mean(outputs, 'val_precision')
        avg_recall = self.stack_mean(outputs, 'val_recall')
        avg_fscore = self.stack_mean(outputs, 'val_fscore')
        avg_accuracy = self.stack_mean(outputs, 'val_accuracy')
        logs = dict(
            val_loss=avg_loss,
            val_precision=avg_precision,
            val_recall=avg_recall,
            val_fscore=avg_fscore,
            val_accuracy=avg_accuracy,
        )
        return dict(val_loss=avg_loss, log=logs, progress_bar=logs)

    def test_epoch_end(self, outputs):
        avg_loss = self.stack_mean(outputs, 'test_loss')
        avg_precision = self.stack_mean(outputs, 'test_precision')
        avg_recall = self.stack_mean(outputs, 'test_recall')
        avg_fscore = self.stack_mean(outputs, 'test_fscore')
        avg_accuracy = self.stack_mean(outputs, 'test_accuracy')
        logs = dict(
            test_loss=avg_loss,
            test_precision=avg_precision,
            test_recall=avg_recall,
            test_fscore=avg_fscore,
            test_accuracy=avg_accuracy,
        )
        return dict(test_loss=avg_loss, log=logs, progress_bar=logs)


@ex.capture
def get_early_callback(patience, monitored_variable, mode):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitored_variable,
        mode=mode,
        patience=patience,
        verbose=True,
    )
    return early_stop_callback


@ex.capture
def get_model_ckpt_callback(experiment_name, fold):
    fold_dir = runs_dir / experiment_name / f'fold_{fold}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=fold_dir,
        monitor='val_fscore',
        mode='max',
    )
    return checkpoint_callback


@ex.capture
def get_model(batch_size, learning_rate):
    config = dict(
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr=learning_rate,
    )
    hparams = argparse.Namespace(**config)
    model = Model(hparams)
    return model


@ex.capture
def get_trainer(args):
    # pylint: disable=no-value-for-parameter
    import copy
    args = copy.deepcopy(args)  # why? because e.g. batch size may be changed?
    if args['early_stop_callback']:
        args['early_stop_callback'] = get_early_callback()
    args['default_root_dir'] = get_default_root_dir()
    # args['logger'] = get_logger()
    trainer = pl.Trainer(**args)
    return trainer


def find_lr(trainer, model, figure_path=None, print_results=False):
    lr_finder = trainer.lr_find(model)

    # Results can be found in
    if print_results:
        print(lr_finder.results)

    # Plot with
    if figure_path is not None:
        fig = lr_finder.plot(suggest=True)
        fig.savefig(figure_path, dpi=400)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    return new_lr


@ex.capture
def get_checkpoint_path(version_number):
    pl_dir = runs_dir / 'lightning_logs'
    cps_dir = pl_dir / f'version_{version_number}' / 'checkpoints'
    checkpoint_paths = list(cps_dir.glob('epoch=*.ckpt'))
    if not checkpoint_paths:
        raise FileNotFoundError('No checkpoints found')
    if len(checkpoint_paths) > 1:
        raise ValueError('More than one checkpoint found')
    return checkpoint_paths[0]


@ex.capture
def get_default_root_dir(experiment_name, fold):
    default_dir = runs_dir / experiment_name / f'fold_{fold}'
    default_dir.mkdir(exist_ok=True, parents=True)
    return default_dir


@ex.capture
def get_logger(experiment_name, fold, _run):
    experiment_dir = runs_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(
        str(experiment_dir),
        f'fold_{fold}',
    )
    return logger


@ex.command
def find_max_batch_size():
    # pylint: disable=no-value-for-parameter
    model = get_model()
    trainer = get_trainer()
    print('Max. batch size:', trainer.scale_batch_size(model))


@ex.command
def find_best_lr():
    # pylint: disable=no-value-for-parameter
    model = get_model()
    trainer = get_trainer()
    lr = find_lr(trainer, model, figure_path='/tmp/lr.png')
    print('Best learning rate:', lr)


@ex.command
def test():
    # pylint: disable=no-value-for-parameter
    model = get_model()
    model.load_from_checkpoint(str(get_checkpoint_path()))
    trainer = get_trainer()
    trainer.test(model)


@ex.automain
def run(_seed, seed):
    # pylint: disable=no-value-for-parameter
    pl.seed_everything(_seed) if seed is None else pl.seed_everything(seed)
    model = get_model()
    trainer = get_trainer()
    trainer.fit(model)
