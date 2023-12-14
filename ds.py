import pandas as pd
import numpy as np
from utils import detag, Base, sentence2states

class Dataset(Base):
    def __init__(self, _paths, encoder_decoder, not_divided=False, test_length=50):
        self.save_hyperparameters()
        self.train_data, self.test_data=[], []
        for path in _paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines+=f.readlines()
        lines=[line.strip() for line in lines if line.strip()!='']

        for line in lines[:test_length]: 
            self.test_data.append((detag(line), line))
        for line in lines[test_length:]:
            encoded_sentence=self.encoder_decoder.encode_states(detag(line), 
                    states=sentence2states(line) if self.not_divided else None, not_divided=self.not_divided)
            assert not np.array_equal(encoded_sentence, np.array([])), 'encoded_sentence cannot be empty, '+line
            self.train_data.append(encoded_sentence)
        self.train_length=len(self.train_data)
        self.maxlen=max(len(x) for x in self.data)
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
    def got_train_length(self):
        return self.train_length
    def get_test_length(self):
        return self.test_length

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