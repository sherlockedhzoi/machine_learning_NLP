from glob import glob
import pandas as pd
import numpy as np
from utils import is_seperator

class Dataset:
    def __init__(self, ds_url, letter_dict, without_mark=True):
        # print(glob(ds_url))
        self.without_mark=without_mark
        self.data=[]
        paths=glob(ds_url)
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines=f.readlines()
                for line in lines:
                    sentence=''.join(line.strip().split('  '))
                    encoded_sentence=[]
                    if self.without_mark:
                        for letter in sentence:
                            encoded_sentence.append(letter_dict.get_id(letter))
                    else:
                        words=line.strip().split()
                        for word in words:
                            if word is None: continue
                            elif is_seperator(word): 
                                encoded_sentence.append(letter_dict.get_id(word),3)
                            else:
                                encoded_sentence.append((letter_dict.get_id(word[0]),0))
                                for letter in word[1:-1]:
                                    encoded_sentence.append(letter_dict.get_id(letter),1)
                                encoded_sentence.append((letter_dict.get_id(word[-1]),2))
                    self.data.append(encoded_sentence)

    def get_data(self):
        return self.data