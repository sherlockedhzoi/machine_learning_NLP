import pandas as pd
import numpy as np

class Trie:
    def __new_node(self):
        return {
            'son': {},
            'is_end': False,
            'attr': []
        }

    def __init__(self):
        self.root=self.__new_node()

    def reset(self):
        self.last=self.root
    
    def step(self, letter):
        if self.last['son'].get(letter) is None: return False
        self.last=self.last['son'][letter]
        return True

    def reach_end(self):
        return self.last['is_end']
    
    def get_attr(self):
        assert self.last['is_end']
        return self.last['attr']

    def insert(self, word, attr):
        # print(word, attr)
        self.reset()
        for letter in word:
            if not self.step(letter):
                self.last['son'][letter]=self.__new_node()
                self.step(letter)
        if self.last['is_end']==True:
            raise RuntimeError(word,'value existed,',attr,self.last['attr'])
        self.last['is_end']=True
        self.last['attr']=attr
    
    def search(self, word):
        self.reset()
        for letter in word:
            if not self.step(letter):
                return None
        return self.last['attr'] if self.last['is_end'] else None

class WordDict(Trie):
    def __init__(self, path):
        super().__init__()
        words=pd.read_csv(path, header=None, encoding='utf-8').drop_duplicates([0]).fillna('undefined')
        self.total_freq=sum(map(int,words.iloc[:,1]))
        self.build(words.to_numpy())

    def build(self, datas):
        for data in datas:
            attr={
                'freq': data[1],
                'parts-of-speech': data[2],
                'prop': data[3:],
            }
            self.insert(data[0], attr)

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