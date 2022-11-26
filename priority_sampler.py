import torch
from torch.utils.data import Sampler, RandomSampler, BatchSampler


class PrioritySampler(Sampler):
    def __init__(self, data_source, batch_size, overhead=2):
        self.batch_size = batch_size
        self.data_source = data_source
        self.overhead = overhead
        self.num_losses = 1
        self.batch_sampler = BatchSampler(RandomSampler(range(len(data_source))), overhead*batch_size, True)
        self.epoch = 0

    def __len__(self):
        return len(self.data_source) // self.batch_size

    def __iter__(self):
        sampler_iter = iter(self.batch_sampler)
        losses = torch.zeros(self.overhead * self.batch_size)
        obs_indices = torch.zeros(self.batch_size)
        while True:
            overhead_batch = next(sampler_iter)
            for i in range(len(overhead_batch)):
                losses[i] = self.data_source.get_loss(overhead_batch[i])
            indices = torch.argsort(losses, descending=True)
            next_index = 0
            for i in range(len(overhead_batch)):
                if indices[i] < self.batch_size:
                    obs_indices[next_index] = overhead_batch[indices[i]]
                    next_index += 1
            del next_index
            del indices
            del overhead_batch
            yield obs_indices
