import pandas as pd
import numpy as np
from utils import detag, Base

class Dataset(Base):
    def __init__(self, _paths, encoder_decoder, not_divided=False, train=True):
        self.save_hyperparameters()
        self.data=[]
        for path in _paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip()
                    if line=='': continue
                    if train:
                        encoded_sentence=self.encoder_decoder.encode_sentence(line, not_divided=self.not_divided)
                        assert not np.array_equal(encoded_sentence, np.array([])), 'encoded_sentence cannot be empty, '+line
                        self.data.append(encoded_sentence)
                    else:
                        sentence=detag(line)
                        self.data.append((sentence, line))
        self.length=len(self.data)
        self.maxlen=max(len(x) for x in self.data)
        print('lines of data: ', self.length)
        print('maxlen: ', self.maxlen)

    def __len__(self):
        return self.length

    def get_data(self):
        return self.data
    
    def __getitem__(self, idx):
        return self.data[idx]

    def get_maxlen(self):
        return self.maxlen

    def get_batch(self, batch_size):
        batch=[]
        for i in range(batch_size):
            batch.append(self.data[np.random.randint(0, self.length)])
        return batch