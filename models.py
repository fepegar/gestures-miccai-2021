import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_max_batch_size(model, frames_per_clip, size, powers_2=True):
    assert isinstance(size, tuple) and len(size) == 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    model.to(device)
    model.eval()
    largest = 1
    while True:
        print(f'Trying batch size {batch_size}...')
        batch = torch.rand(batch_size, 3, frames_per_clip, *size, device=device)
        try:
            with torch.no_grad():
                model(batch)
        except RuntimeError:
            batch_size = largest
            powers_2 = False
        largest = batch_size
        print('OK')
        if powers_2:
            batch_size *= 2
        else:
            batch_size += 1


def get_ig65_model(frames_per_clip, remove_fc=True):
    model_name = f'r2plus1d_34_{frames_per_clip}_kinetics'
    model = torch.hub.load(
        'moabitcoin/ig65m-pytorch',
        model_name,
        num_classes=400,
        pretrained=True,
    )
    if remove_fc:
        model.fc = nn.Identity()
    return model


def get_resnet(layers=50, remove_fc=True):
    model = getattr(torchvision.models, f'resnet{layers}')(pretrained=True)
    if remove_fc:
        model.fc = nn.Identity()
    return model


def get_wide_resnet(layers=50, remove_fc=True):
    name = f'wide_resnet{layers}_2'
    model = getattr(torchvision.models, name)(pretrained=True)
    if remove_fc:
        model.fc = nn.Identity()
    return model


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=fepegar
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MeanModel(nn.Module):
    def __init__(
            self,
            input_size=512,
            output_classes=2,
            final_dropout_probability=0,
            ):
        super().__init__()
        self.dropout = nn.Dropout(final_dropout_probability)
        self.fc = nn.Linear(
            input_size,
            output_classes,
        )

    def extract_features(self, x):
        output = x.mean(dim=1)
        return output

    def forward(self, x):
        features = self.dropout(self.extract_features(x))
        logits = self.fc(features)
        return logits


class RecurrentModel(nn.Module):
    def __init__(
            self,
            input_size=512,
            hidden_size=128,
            output_classes=2,
            num_layers=1,
            bidirectional=False,
            final_dropout_probability=0,
            ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(final_dropout_probability)
        num_directions = 2 if bidirectional else 1
        linear_input_size = num_directions * num_layers * hidden_size
        self.fc = nn.Linear(
            linear_input_size,
            output_classes,
        )

    def extract_features(self, x, state=None):
        all_outputs, (hidden_state, cell_state) = self.lstm(x, hx=state)
        # hidden_state has shape (num_layers * num_directions, batch, hidden_size)
        reshaped = hidden_state.permute(1, 0, 2)
        # reshaped has shape (batch, num_layers * num_directions, hidden_size)
        flattened = reshaped.flatten(start_dim=1)
        # flattened has shape (batch, num_layers * num_directions * hidden_size)
        return flattened

    def forward(self, x):
        features = self.dropout(self.extract_features(x))
        logits = self.fc(features)
        return logits


class TCN_(nn.Module):
    def __init__(self, in_channels, out_classes, num_blocks, reduction=1):
        super().__init__()
        self.blocks = self.get_blocks(in_channels, num_blocks, reduction)
        in_fc_channels = self.blocks[-1].out_channels
        self.fc = nn.Linear(in_fc_channels, out_classes)
        self.in_channels = in_channels

    def get_blocks(self, in_channels, num_blocks, reduction):
        blocks = []
        k = 3
        dilation = 1
        for i in range(num_blocks):
            out_channels = int(round(in_channels * reduction))
            conv = nn.Conv1d(in_channels, out_channels, k, dilation=dilation)
            blocks.append(conv)
            # blocks.append(nn.MaxPool1d(3, 2))
            in_channels = out_channels
            dilation *= 2
            k = dilation * (k - 1) + 1
        blocks.pop()  # remove last pooling
        return nn.Sequential(*blocks)

    def get_receptive_field(self, length=10000):
        tensor = torch.empty(1, self.in_channels, length)
        with torch.no_grad():
            field = length - self.blocks(tensor).shape[-1]
        return field

    def get_temporal_receptive_field(self, fpc=8, fps=15, **kwargs):
        field = self.get_receptive_field(**kwargs)
        snippet_duration = fpc / fps
        return field * snippet_duration

    def global_pool(self, x, type_='max'):
        if type_ == 'max':
            return x.max(dim=-1)[0]
        elif type_ == 'average':
            return x.mean(dim=-1)

    def extract_features(self, x):
        feature_maps = self.blocks(x)
        feature_vector = self.global_pool(feature_maps)
        return feature_vector

    def forward(self, x):
        feature_vector = self.extract_features(x)
        logits = self.fc(feature_vector)
        return logits


# Adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class TemporalBlock(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_outputs,
            kernel_size,
            dilation,
            dropout=0.2,
            pool=True,
            ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            num_inputs,
            num_outputs,
            kernel_size,
            padding=((kernel_size - 1) * dilation) // 2,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(num_outputs)
        self.act1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            num_outputs,
            num_outputs,
            kernel_size,
            padding=((kernel_size - 1) * dilation) // 2,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(num_outputs)
        self.act2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act1,
            self.dropout1,
            self.conv2,
            self.bn2,
            self.act2,
            self.dropout2,
        )
        if num_inputs != num_outputs:
            self.adapt_features = nn.Conv1d(num_inputs, num_outputs, 1)
        else:
            self.adapt_features = None
        self.pool = pool

    def forward(self, x):
        out = self.net(x)
        res = x if self.adapt_features is None else self.adapt_features(x)
        a = out + res
        if self.pool:
            a = F.max_pool1d(a, 2)
        return a


# Adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class TCNBlocks(nn.Module):
    def __init__(
            self,
            num_blocks,
            num_hidden_units,
            num_inputs=512,
            kernel_size=3,
            dropout=0.2,
            pool=False,
            ):
        super().__init__()
        self.num_hidden_units = num_hidden_units
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden_units
            out_channels = num_hidden_units
            block = TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                dropout=dropout,
                pool=pool,
            )
            blocks.append(block)
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, out_classes, *args, **kwargs):
        super().__init__()
        self.blocks = TCNBlocks(*args, **kwargs)
        in_fc_channels = self.blocks.num_hidden_units
        self.fc = nn.Linear(in_fc_channels, out_classes)

    def global_pool(self, x, type_='max'):
        if type_ == 'max':
            return x.max(dim=-1)[0]
        elif type_ == 'average':
            return x.mean(dim=-1)

    def extract_features(self, x):
        feature_maps = self.blocks(x)
        feature_vector = self.global_pool(feature_maps)
        return feature_vector

    def forward(self, x):
        feature_vector = self.extract_features(x)
        logits = self.fc(feature_vector)
        return logits
