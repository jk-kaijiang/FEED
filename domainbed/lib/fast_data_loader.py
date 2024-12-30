import copy
import random
import torch


class RandomRatioBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_shots):
        self.dataset = dataset
        self.num_shots = num_shots

        labels, attributes = self.dataset.y, self.dataset.z

        self.groups = {}
        for idx, (label, attribute) in enumerate(zip(labels, attributes)):
            if label.item() not in self.groups:
                self.groups[label.item()] = {}
            if attribute.item() not in self.groups[label.item()]:
                self.groups[label.item()][attribute.item()] = []
            self.groups[label.item()][attribute.item()].append(idx)

        self.labels_required_samples = {
            label: num_shots for label in set(labels.tolist())
        }

    def __iter__(self):
        while True:
            groups = copy.deepcopy(self.groups)
            batch = []
            label_counts = {label: 0 for label in self.labels_required_samples}
            for label, value1 in groups.items():
                for _, value2 in value1.items():
                    selected_idx = random.choice(value2)
                    batch.append(selected_idx)
                    label_counts[label] += 1
                    value2.remove(selected_idx)
            for label in label_counts:
                temp_label_dict = groups[label]
                temp_attrs = list(temp_label_dict.keys())
                while label_counts[label] < self.num_shots:
                    temp_attr = random.choice(temp_attrs)
                    selected_idx = random.choice(temp_label_dict[temp_attr])
                    batch.append(selected_idx)
                    label_counts[label] += 1
                    temp_label_dict[temp_attr].remove(selected_idx)
            if sum(label_counts.values()) == sum(self.labels_required_samples.values()):
                yield batch
                batch = []
                label_counts = {label: 0 for label in self.labels_required_samples}

    def __len__(self):
        return len(self.dataset) // sum(self.labels_required_samples.values())

class _InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class MetaDataLoader:
    def __init__(self, dataset, num_shots, num_workers):
        super().__init__()

        batch_sampler = RandomRatioBatchSampler(dataset, num_shots)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError("Infinite loader does not support length querying.")

class MetaFastDataLoader:
    def __init__(self, dataset, num_shots, num_workers):
        super().__init__()

        batch_sampler = RandomRatioBatchSampler(dataset, num_shots)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        # print(sampler)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class FastDataLoader:

    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
