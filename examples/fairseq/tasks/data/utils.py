# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import collections
from random import Random
from typing import Dict, Iterable, Optional

import numpy as np
from infinibatch import iterators


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if isinstance(x, np.ndarray):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items()
            )
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


class NativeCheckpointableIterator(iterators.CheckpointableIterator):
    def __init__(self, iterable: Iterable):
        self._input_iterable = iterable
        self.setstate(None)

    def getstate(self) -> Dict:
        return {"num_items_yielded": self._num_items_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        self._iterator = iter(self._input_iterable)
        self._num_items_yielded = (
            iterators._advance_iterator(self._iterator, checkpoint["num_items_yielded"])
            if checkpoint is not None
            else 0
        )

    def __next__(self):
        item = next(self._iterator)
        self._num_items_yielded += 1
        return item

    def close(self):
        pass


class WeightIterator(object):
    def __init__(self, weights, seed):
        self.weights = weights
        self.seed = seed
        self.control_index = list(range(len(weights)))
        self.setstate(None)

    def __iter__(self):
        return self

    def getstate(self):
        return {"random_state": self._random_state}

    def setstate(self, checkpoint):
        self._random_state = checkpoint["random_state"] if checkpoint else None
        self._random = (
            None  # this will trigger the lazy initialization in self.__next__
        )

    def __next__(self):
        if self._random is None:
            self._random = Random(self.seed)
            if self._random_state is not None:
                self._random.setstate(self._random_state)
        idx = self._random.choices(self.control_index, self.weights)[0]
        self._random_state = self._random.getstate()
        return idx

    def close(self):
        pass
