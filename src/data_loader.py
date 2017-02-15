import numpy as np


class DataSet(object):
    def __init__(self, data):
        self.data = data
        self.epochs_completed = 0
        self._index_in_epoch = 0
        self.num_examples = data.shape[0]
        # random shuffle the data
        np.random.shuffle(data)

    def next_batch(self, batch_size):
        """
        Gives the next batch in this dataset
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            permutation = np.arange(self.num_examples)
            np.random.shuffle(permutation)
            self.data = self.data[permutation]
            self._index_in_epoch = batch_size
            self.epochs_completed += 1
            start = 0
        end = self._index_in_epoch
        return self.data[start: end]


