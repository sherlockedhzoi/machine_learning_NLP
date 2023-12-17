import pandas as pd
import numpy as np
from utils import detag, Base

class Dataset(Base):
    def __init__(self, _train_paths, _test_paths, _atom='letter', _supervised=True):
        self.save_hyperparameters()
        lines=[]
        for path in _train_paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines+=f.readlines()
        self.train_data=[line.strip() for line in lines if line.strip()!='']
        lines=[]
        for path in _test_paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines+=f.readlines()
        self.test_data=[line.strip() for line in lines if line.strip()!='']

        self.train_size=len(self.train_data)
        self.test_size=len(self.test_data)
        self.maxlen=max([len(detag(x)) for x in self.train_data+self.test_data])

        if not _supervised:
            self.train_data=[detag(sentence, sep=('' if _atom=='letter' else ' ')) for sentence in self.train_data]
        self.test_data=[(sentence, detag(sentence)) for sentence in self.test_data]

        print('training lines of data: ', self.train_size)
        print('test lines of data: ', self.test_size)
        print('maxlen: ', self.maxlen)

    def __len__(self):
        return self.train_size+self.test_size
    def __getitem__(self, idx):
        return self.data[idx]

    def get_train_data(self):
        return self.train_data
    def get_test_data(self):
        return self.test_data

    def get_maxlen(self):
        return self.maxlen
    def get_train_size(self):
        return self.train_size
    def get_test_size(self):
        return self.test_size

    def get_train_batch(self, batch_size):
        batch=[]
        for i in range(batch_size):
            batch.append(self.train_data[np.random.randint(0, self.train_size)])
        return batch
    def get_test_batch(self, batch_size):
        batch=[]
        for i in range(batch_size):
            batch.append(self.test_data[np.random.randint(0, self.test_size)])
        return batch