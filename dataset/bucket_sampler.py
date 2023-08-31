import math
import random
from itertools import islice
from typing import List, Iterable, Tuple, Optional, Any, Iterator
from torch.utils.data.sampler import Sampler


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


def lazy_groups_of(iterable: Iterable[Any], group_size: int) -> Iterator[List[Any]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        data,
        batch_size: int,
        sorting_keys: List[str],
        padding_noise: float = 0.1,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        super(BucketBatchSampler, self).__init__(self)
        self.data = data
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if not shuffle:
            self.padding_noise = 0.0

    def _argsort_by_padding(
        self, instances: Iterable[Any]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:  # type: ignore
                lengths.append(len(instance[field_name]))
                noisy_lengths.append(add_noise_to_value(lengths[-1], self.padding_noise))

            instances_with_lengths.append((noisy_lengths, lengths, instance))
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])

        return (
            [instance_with_index[-1] for instance_with_index in with_indices],
            [instance_with_index[0][1] for instance_with_index in with_indices],
        )

    def __iter__(self) -> Iterator[List[int]]:
        return self.get_batch_indices()

    def get_batch_indices(self) -> Iterator[List[int]]:
        indices, _ = self._argsort_by_padding(self.data)
        batches = []
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices = list(group)
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batches.append(batch_indices)
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def get_num_batches(self) -> int:
        batch_count_float = len(self.data) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def get_batch_size(self) -> Optional[int]:
        return self.batch_size

    def __len__(self):
        return self.get_num_batches()
