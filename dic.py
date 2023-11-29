import pandas as pd
import numpy as np

class Trie:
    def __new_node():
        return {
            "son": {},
            "is_end": False,
            attr: []
        }

    def __init__():
        root=__new_node()

    def reset():
        last=root
    
    def step(letter):
        if last.son[letter] is None: return False
        last=last.son[letter]
        return True

    def reach_end():
        return last.is_end
    
    def get_attr():
        assert last.is_end
        return last.attr

    def insert(word, attr):
        reset()
        for letter in word:
            if not step(letter):
                last.son[letter]=__new_node()
                step(letter)
        assert last.is_end==False
        last.is_end=True
        last.attr=attr
    
    def search(word):
        reset()
        for letter in word:
            if not step(letter):
                return None
        return last.attr if last.is_end else None

    def __build(datas):
        for data in datas:
            attr={
                'freq': data[1],
                'parts-of-speech': data[2],
                'prop': data[3:],
            }
            trie.insert(data[0], attr)

class WordDict(Trie):
    def __init__(self, path):
        super().__init__()
        words=pd.read_csv(path, header=None)
        self.total_freq=words.iloc[:,1].sum()
        super().__build(words.to_numpy())

class LetterDict:
    def __init__(self, path):
        self.letter2id, self.id2letter={}, {}
        with open(path,'r', encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                if line:
                    idx, letter=line.strip('\n').split('\t')
                    self.letter2id[letter]=int(idx)
                    self.id2letter[int(idx)]=letter
        self.total_letter=len(self.letter2id)
        print(self.total_letter)
    
    def __len__(self):
        return self.total_letter

    def get_id(self, letter):
        return self.letter2id.get(letter, self.total_letter-1)
    
    def get_letter(self, idx):
        return self.id2letter.get(idx, 'OOV')