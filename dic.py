import pandas as pd
import numpy as np

class Trie:
    def __new_node(self):
        return {
            'son': {},
            'ID': None,
            'freq': 0,
            'tag': None,
            'prop': None
        }
    def __init__(self):
        self.root=self.__new_node()
        self.length=0
        self.insert('<unk>')

    def reset(self):
        self.last=self.root
    def step(self, letter):
        if self.last['son'].get(letter) is None: return False
        self.last=self.last['son'][letter]
        return True
    @property
    def reach_end(self):
        return self.last['ID'] is not None
    def get_now_freq(self):
        assert self.reach_end, 'freq only meaningful when reached end'
        return self.last['freq']
    
    def insert(self, word, freq=0, tag=None, prop=None):
        self.reset()
        for letter in word:
            if not self.step(letter):
                self.last['son'][letter]=self.__new_node()
                self.step(letter)
        if self.reach_end:
            raise RuntimeError(word, f'value existed with ID={self.last["ID"]}')
        self.last['ID']=self.length+1
        self.last['freq']=freq
        self.last['tag']=tag
        self.last['prop']=prop
        self.length+=1
    
    def get_id(self, word):
        self.reset()
        for letter in word:
            if not self.step(letter):
                return self.unk
        return self.last['ID'] if self.reach_end else self.unk
    def get_freq(self, word):
        self.reset()
        for letter in word:
            if not self.step(letter):
                return 0
        return self.last['freq']
    def get_attr(self, word, attr='tag'):
        assert attr in ['tag', 'prop'], f'attr must be "tag" or "prop", but received {attr}'
        self.reset()
        for letter in word:
            if not self.step(letter):
                return None
        return self.last[attr]

    def __len__(self):
        return self.length
    @property
    def unk(self):
        return self.get_id('<unk>')

class WordDict(Trie):
    def __init__(self, path='data/words.csv'):
        super().__init__()
        self.words=['<unk>']
        datas=pd.read_csv(path, header=None, encoding='utf-8', dtype=str).drop_duplicates([0]).fillna('<unk>').to_numpy()
        self.total_freq=sum(map(int,datas[:,1]))
        # print(datas)
        for i, data in enumerate(datas):
            # data=data[0:]
            while data[-1]=='<unk>':
                data=data[:-1]
            word, freq, tag, prop=data[0], int(data[1]), data[2], (data[3:].tolist() if len(data)>3 else '<unk>')
            # print(data, word, freq, tag, prop)
            self.words.append(word)
            self.insert(word, freq, tag, prop)
        print('total words:', self.length)
        
    def get_total_freq(self):
        return self.total_freq
    def __getitem__(self, ID):
        return self.words[ID]

class LetterDict:
    def __init__(self, path='data/letters.dic'):
        self.letter2id, self.id2letter={}, {}
        with open(path,'r', encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                if line:
                    idx, letter=line.strip('\n').split('\t')
                    self.letter2id[letter]=int(idx)
                    self.id2letter[int(idx)]=letter
        # print(self.letter2id)
        self.length=len(self.letter2id)
        print('total_letter:', self.length)
    
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.id2letter.get(idx, 'OOV')
    def get_id(self, letter):
        return self.letter2id.get(letter, self.length-1)

class TagDict:
    def __init__(self, path='data/tag.csv'):
        self.tag2id, self.id2tag={}, {}
        tag_dict=pd.read_csv(path)['tag'].to_numpy()
        self.tag2id=dict(zip(tag_dict, range(len(tag_dict))))
        self.id2tag=dict(zip(range(len(tag_dict)), tag_dict))
        self.length=len(tag_dict)
        print('tag_cnt:',self.length)

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.id2tag.get(idx, '<unk>')
    def get_id(self, tag):
        return self.tag2id.get(tag, None)