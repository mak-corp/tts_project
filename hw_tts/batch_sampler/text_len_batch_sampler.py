from torch.utils.data import Sampler, BatchSampler, SequentialSampler, RandomSampler
import numpy as np


class TextLenBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batch_expand_size):
        # drop_last = True
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batch_expand_size = batch_expand_size
        self.sampler = RandomSampler(data_source)
        self.len = len(data_source) // batch_size

    def get_sorted_grouped_samples(self, extended_batch):
        extended_batch = np.array(extended_batch)
        len_arr = np.array([self.data_source[idx]["text"].size(0) for idx in extended_batch])
        sorted_extended_batch = extended_batch[np.argsort(-len_arr)]
        return sorted_extended_batch.reshape((-1, self.batch_size)).tolist()

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        while True:
            extended_batch = []
            stop = False
            for _ in range(self.batch_expand_size):
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    extended_batch.extend(batch)
                except StopIteration:
                    stop = True
                    break
            if len(extended_batch) == 0:
                break

            sorted_grouped = self.get_sorted_grouped_samples(extended_batch)
            yield from sorted_grouped

            if stop:
                break

    def __len__(self):
        return self.len
