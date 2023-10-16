from typing import Iterator, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset, DistributedSampler, WeightedRandomSampler


class DistributedWeightedRandomSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        replacement=True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.weights = np.array(weights)
        self.replacement = replacement

    def __iter__(self):
        indices = np.array(list(super().__iter__()))
        weights = self.weights[indices]
        weighted_indices = list(
            WeightedRandomSampler(weights, len(weights), self.replacement)
        )
        indices = [indices[wi] for wi in weighted_indices]
        return iter(indices)
