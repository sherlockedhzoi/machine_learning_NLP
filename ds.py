import pandas as pd
import numpy as np
from param import Base

class Dataset(Base):
    def __init__(self, _paths, encoder_decoder, not_divided=True):
        # print(glob(ds_url))
        self.save_hyperparameters()
        self.data=[]
        for path in _paths:
            with open(path, 'r', encoding='utf-8') as f:
                for i in range(1000):
                    line=f.readline()
                    if line.strip()=='': continue
                    encoded_sentence=self.encoder_decoder.encode_sentence(line, not_divided=self.not_divided)
                    assert not np.array_equal(encoded_sentence, np.array([]))
                    self.data.append(encoded_sentence)
        self.maxlen=max(len(x) for x in self.data)

    def get_data(self):
        return self.data