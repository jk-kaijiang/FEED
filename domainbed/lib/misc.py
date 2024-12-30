

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm
from collections import Counter
import time
import torch.nn.functional as F

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):

    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.y = underlying_dataset.y[keys]
        self.z = underlying_dataset.z[keys]
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):

    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

class _MergeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(_MergeDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.offsets = np.cumsum([0] + self.lengths[:-1])
        self.y = np.concatenate([dataset.y for dataset in self.datasets])
        self.z = np.concatenate([dataset.z for dataset in self.datasets])

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset + self.lengths[i]:
                return self.datasets[i][index - offset]
        raise IndexError("Index out of range in merged dataset")

    def __len__(self):
        return sum(self.lengths)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def metrics(network, loader, device):
    num_correct_pred = 0
    num_sample = 0
    s_0 = 0.0
    s_1 = 0.0
    yhat_1_s_0 = 0.0
    yhat_1_s_1 = 0.0
    y_0_s_0 = 0.0
    y_0_s_1 = 0.0
    y_1_s_0 = 0.0
    y_1_s_1 = 0.0
    yhat_1_y_0_s_0 = 0
    yhat_1_y_0_s_1 = 0
    yhat_1_y_1_s_0 = 0
    yhat_1_y_1_s_1 = 0
    preds = []
    s_0_indices = []
    s_1_indices = []

    network.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            pred_prob = network.predict(x)
            pred_1_prob = F.softmax(pred_prob, dim=1)[:, 1]

            # accuracy
            if pred_prob.size(1) == 1:
                num_correct_pred += (pred_prob.gt(0).eq(y).float()).sum().item()
            else:
                num_correct_pred += (pred_prob.argmax(1).eq(y).float()).sum().item()
            num_sample += len(x)

            # delta_dp (md), dp
            pred = pred_prob.argmax(1)

            temp_s_1 = z.sum().item()
            temp_s_0 = z.size(0) - temp_s_1
            logic_yhat_1_s_0 = torch.logical_and(pred == 1, z == 0.0)
            logic_yhat_1_s_1 = torch.logical_and(pred == 1, z == 1.0)
            logic_y_0_s_0 = torch.logical_and(y == 0, z == 0.0)
            logic_y_0_s_1 = torch.logical_and(y == 0, z == 1.0)
            logic_y_1_s_0 = torch.logical_and(y == 1, z == 0.0)
            logic_y_1_s_1 = torch.logical_and(y == 1, z == 1.0)
            temp_yhat_1_s_0 = torch.where(logic_yhat_1_s_0, 1.0, 0.0).sum().item()
            temp_yhat_1_s_1 = torch.where(logic_yhat_1_s_1, 1.0, 0.0).sum().item()
            temp_y_0_s_0 = torch.where(logic_y_0_s_0, 1.0, 0.0).sum().item()
            temp_y_0_s_1 = torch.where(logic_y_0_s_1, 1.0, 0.0).sum().item()
            temp_y_1_s_0 = torch.where(logic_y_1_s_0, 1.0, 0.0).sum().item()
            temp_y_1_s_1 = torch.where(logic_y_1_s_1, 1.0, 0.0).sum().item()

            temp_yhat_1_y_0_s_0 = torch.where(torch.logical_and(pred == 1, logic_y_0_s_0), 1.0, 0.0).sum().item()
            temp_yhat_1_y_0_s_1 = torch.where(torch.logical_and(pred == 1, logic_y_0_s_1), 1.0, 0.0).sum().item()
            temp_yhat_1_y_1_s_0 = torch.where(torch.logical_and(pred == 1, logic_y_1_s_0), 1.0, 0.0).sum().item()
            temp_yhat_1_y_1_s_1 = torch.where(torch.logical_and(pred == 1, logic_y_1_s_1), 1.0, 0.0).sum().item()

            s_0 += temp_s_0
            s_1 += temp_s_1
            yhat_1_s_0 += temp_yhat_1_s_0
            yhat_1_s_1 += temp_yhat_1_s_1
            y_0_s_0 += temp_y_0_s_0
            y_0_s_1 += temp_y_0_s_1
            y_1_s_0 += temp_y_1_s_0
            y_1_s_1 += temp_y_1_s_1
            yhat_1_y_0_s_0 += temp_yhat_1_y_0_s_0
            yhat_1_y_0_s_1 += temp_yhat_1_y_0_s_1
            yhat_1_y_1_s_0 += temp_yhat_1_y_1_s_0
            yhat_1_y_1_s_1 += temp_yhat_1_y_1_s_1

            preds.extend(pred_1_prob.cpu().numpy())
            s_0_indices.extend((z == 0).nonzero(as_tuple=True)[0].cpu().numpy())
            s_1_indices.extend((z == 1).nonzero(as_tuple=True)[0].cpu().numpy())
            pass

    network.train()
    acc = num_correct_pred / num_sample
    delta_dp = abs((yhat_1_s_0/s_0) - (yhat_1_s_1/s_1)) # md
    # dp
    if yhat_1_s_0 == 0 or s_0 == 0 or yhat_1_s_1 == 0 or s_1 == 0:
        dp = 0
    else:
        dp = (yhat_1_s_0 / s_0) / (yhat_1_s_1 / s_1)
        if dp > 1.0:
            dp = 1.0 / dp
    # eopp
    if y_1_s_0 == 0 or y_1_s_1 == 0 or yhat_1_y_1_s_1 == 0:
        eopp = 0
    else:
        eopp = (yhat_1_y_1_s_0 / y_1_s_0) / (yhat_1_y_1_s_1 / y_1_s_1)
        if eopp >= 1:
            eopp = 1.0 / eopp
    # delta_eopp
    if y_1_s_0 == 0 or y_1_s_1 == 0:
        delta_eopp = 0
    else:
        delta_eopp = abs((yhat_1_y_1_s_0 / y_1_s_0) - (yhat_1_y_1_s_1 / y_1_s_1))
    # delta_eo
    if y_1_s_0 == 0 or y_1_s_1 == 0:
        delta_eo = 0
    else:
        delta_eo = (abs((yhat_1_y_0_s_0 / y_0_s_0) - (yhat_1_y_0_s_1 / y_0_s_1)) + abs((yhat_1_y_1_s_0 / y_1_s_0) - (yhat_1_y_1_s_1 / y_1_s_1)))/2
    # auc
    preds = torch.tensor(preds, device=device)
    s_0_indices = torch.tensor(s_0_indices, device=device)
    s_1_indices = torch.tensor(s_1_indices, device=device)
    preds_s_0 = preds[s_0_indices]
    preds_s_1 = preds[s_1_indices]
    auc = (preds_s_0.unsqueeze(1) > preds_s_1.unsqueeze(0)).float().mean().item()
    if auc < 0.5:
        auc = 1.0 - auc
    result = {'acc':acc, 'dp':dp, 'Δdp':delta_dp, 'eopp':eopp, 'Δeopp':delta_eopp, 'Δeo':delta_eo, 'auc':auc}
    return result

def accuracy(network, loader, device):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y, z in loader:

            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)

    network.train()
    return correct / total

def md(network, loader, device):

    s_0 = 0.0
    y_1_s_0 = 0.0
    s_1 = 0.0
    y_1_s_1 = 0.0


    network.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            p = network.predict(x)
            p = p.argmax(1)


            temp_s_1 = z.sum().item()
            temp_s_0 = z.size(0) - temp_s_1
            temp_y_1_s_0 = torch.where(torch.logical_and(p == 1, z == 0.0), 1.0, 0.0).sum().item()
            temp_y_1_s_1 = torch.where(torch.logical_and(p == 1, z == 1.0), 1.0, 0.0).sum().item()

            s_0 += temp_s_0
            y_1_s_0 += temp_y_1_s_0
            s_1 += temp_s_1
            y_1_s_1 += temp_y_1_s_1

    network.train()

    result = abs((y_1_s_0/s_0) - (y_1_s_1/s_1))
    return result

def dp(network, loader, device):

    s_0 = 0
    y_1_s_0 = 0
    s_1 = 0
    y_1_s_1 = 0

    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            p = network.predict(x)
            p = p.argmax(1)

            temp_s_1 = z.sum().item()
            temp_s_0 = z.size(0) - temp_s_1
            temp_y_1_s_0 = torch.where(torch.logical_and(p == 1, z == 0.0), 1.0, 0.0).sum().item()
            temp_y_1_s_1 = torch.where(torch.logical_and(p == 1, z == 1.0), 1.0, 0.0).sum().item()
            s_0 += temp_s_0
            y_1_s_0 += temp_y_1_s_0
            s_1 += temp_s_1
            y_1_s_1 += temp_y_1_s_1

    network.train()

    if y_1_s_0 == 0 or s_0 == 0 or y_1_s_1 == 0 or s_1 == 0:
        return 0
    result = (y_1_s_0/s_0) / (y_1_s_1/s_1)
    if result > 1.0:
        return 1.0/result
    return result

def eo(network, loader, device):

    y_1_s_0 = 0
    yhat_1_y_1_s_0 = 0
    y_1_s_1 = 0
    yhat_1_y_1_s_1 = 0

    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            p = network.predict(x)
            p = p.argmax(1)


            logic_y_1_s_0 = torch.logical_and(y == 1, z == 0.0)
            logic_y_1_s_1 = torch.logical_and(y == 1, z == 1.0)
            temp_y_1_s_0 = torch.where(logic_y_1_s_0, 1.0, 0.0).sum().item()
            temp_y_1_s_1 = torch.where(logic_y_1_s_1, 1.0, 0.0).sum().item()
            temp_yhat_1_y_1_s_0 = torch.where(torch.logical_and(p == 1, logic_y_1_s_0 == 1.0), 1.0, 0.0).sum().item()
            temp_yhat_1_y_1_s_1 = torch.where(torch.logical_and(p == 1, logic_y_1_s_1 == 1.0), 1.0, 0.0).sum().item()

            y_1_s_0 += temp_y_1_s_0
            yhat_1_y_1_s_0 += temp_yhat_1_y_1_s_0
            y_1_s_1 += temp_y_1_s_1
            yhat_1_y_1_s_1 += temp_yhat_1_y_1_s_1


    network.train()
    if y_1_s_0 == 0 or y_1_s_1 == 0 or yhat_1_y_1_s_1 == 0:
        return 0
    result = (yhat_1_y_1_s_0/y_1_s_0) / (yhat_1_y_1_s_1/y_1_s_1)
    if result < 1:
        return result
    else:
        return 1.0 / result

def auc(network, loader, device):

    s_0 = []
    s_1 = []
    labels = []
    auc = 0.0
    weights_offset = 0
    network.eval()
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            p = network.predict(x)
            prediction = F.softmax(p, dim=1)
            prediction = prediction[:, 1:]

            for i in range(prediction.size(0)):
                labels.append(prediction[i].item())
                if z[i].item() == 0.0:
                    s_0.append(len(labels)-1)
                elif z[i].item() == 1.0:
                    s_1.append(len(labels)-1)

    for i in range(len(s_0)):
        for j in range(len(s_1)):
            if labels[s_0[i]] > labels[s_1[j]]:
                auc += 1.0
    auc = auc / (len(s_0) * len(s_1))
    network.train()

    if auc < 0.5:
        return 1.0-auc
    return auc

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)