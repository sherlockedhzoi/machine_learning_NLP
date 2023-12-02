from glob import glob
import pandas as pd
import numpy as np
from utils import encode_sentence

class Dataset:
    def __init__(self, ds_url, letter_dict, word_dict, without_mark=True):
        # print(glob(ds_url))
        self.without_mark=without_mark
        self.data=[]
        self.letter_dict=letter_dict
        self.word_dict=word_dict
        paths=glob(ds_url)
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines=f.readlines()
                for line in lines:
                    if line.strip()=='': continue
                    encoded_sentence=encode_sentence(line, letter_dict, without_mark=self.without_mark)
                    assert not np.array_equal(encoded_sentence, np.array([]))
                    self.data.append(encoded_sentence)
        self.maxlen=max(len(x) for x in self.data)

    def get_data(self):
        return self.data