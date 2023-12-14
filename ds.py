import pandas as pd
import numpy as np
from utils import detag, Base, sentence2states

class Dataset(Base):
    def __init__(self, _paths, not_divided=False, with_tag=True, test_size=100):
        self.save_hyperparameters()

        for path in _paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines+=f.readlines()
        datas=[line.strip() for line in lines if line.strip()!='']
        self.train_data, self.test_data=datas[test_size:], datas[:test_size]
        self.test_data=[(sentence, detag(sentence)) for sentence in self.test_data]

        self.train_length=len(self.train_data)
        self.maxlen=max(len(detag(x)) for x in self.train_data+self.test_data)
        print('training lines of data: ', self.train_length)
        print('test lines of data: ', self.test_length)
        print('maxlen: ', self.maxlen)

    def __len__(self):
        return self.train_length+self.test_length
    def __getitem__(self, idx):
        return self.data[idx]

    def get_train_data(self):
        return self.train_data
    def get_test_data(self):
        return self.test_data

    def get_maxlen(self):
        return self.maxlen
    def got_train_size(self):
        return self.train_size
    def get_test_size(self):
        return self.test_size

    def get_train_batch(self, batch_size):
        batch=[]
        for i in range(batch_size):
            batch.append(self.train_data[np.random.randint(0, self.train_length)])
        return batch
    def get_test_batch(self, batch_size):
        batch=[]
        for i in range(batch_size):
            batch.append(self.test_data[np.random.randint(0, self.length)])
        return batch