import pandas as pd
import string
from param import Base
from utils import detag

class Coder(Base):
    def __init__(self, letter_dict, word_dict, with_tag=False):
        self.save_hyperparameters()        
        if self.with_tag:
            self.tag_dict=pd.read_csv('./data/tag.csv')
            self.tag_cnt=len(self.tag_dict['tag'].to_list())
            print('tag_cnt:',self.tag_cnt)
        else:
            self.tag_cnt=1

    def encode_pos(self, pos) -> int:
        assert pos in ['M', 'B', 'E', 'S'], 'state must be M, B, E, or S'
        return {'M': 0, 'B': 1, 'E': 2, 'S': 3}[pos]
    def encode_tag(self, tag) -> int:
        assert tag in self.tag_dict['tag'].to_list(), 'tag %s not defined'%tag
        return (self.tag_dict['tag']==tag).to_numpy().squeeze().nonzero()[0]
    def encode_word(self, word, pos, tag=None):
        return {
            'id': self.letter_dict.get_id(word),
            'word': word,
            'tag': int(self.encode_pos(pos) if not self.with_tag else self.encode_tag(tag)*4+self.encode_pos(pos))
        }
    def encode_sentence(self, line, not_divided=False):
        encoded_sentence=[]
        if not_divided:
            sentence=detag(line)
            for letter in sentence:
                encoded_sentence.append(self.letter_dict.get_id(letter))
        else:
            words=line.split(' ')
            for w in words:
                if w.strip()=='': continue
                if self.with_tag:
                    word, tag=w.strip().split('[')[-1].split(']')[0].split('/')
                    if len(word)==1: 
                        encoded_sentence.append(self.encode_word(word, 'S', tag))
                    else:
                        encoded_sentence.append(self.encode_word(word[0], 'B', tag))
                        for letter in word[1:-1]:
                            encoded_sentence.append(self.encode_word(letter, 'M', tag))
                        encoded_sentence.append(self.encode_word(word[-1], 'E', tag))
                else:
                    word=w.strip()
                    if len(word)==1: 
                        encoded_sentence.append(self.encode_word(word, 'S'))
                    else:
                        encoded_sentence.append(self.encode_word(word[0], 'B'))
                        for letter in word[1:-1]:
                            encoded_sentence.append(self.encode_word(letter, 'M'))
                        encoded_sentence.append(self.encode_word(word[-1], 'E'))
        return encoded_sentence

    def is_begin(self, tag):
        return self.decode_state(tag)['pos'] in ['B', 'S']
    def get_all_ends(self):
        ends=[]
        for pos in ['E','S']:
            for tag in range(self.tag_cnt):
                ends.append(tag*4+self.encode_pos(pos))
        return sorted(ends)

    def decode_pos(self, state) -> str:
        assert state in range(4)
        return ['M', 'B', 'E', 'S'][state]
    def decode_tag(self, state):
        assert state in range(110) and self.tag_dict.iloc[state].loc['tag'] is not None, 'tag has to exist'
        return self.tag_dict.iloc[state].loc['tag']
    def decode_state(self, state):
        return {
            'pos': self.decode_pos(state%4),
            'tag': self.decode_tag(state//4) if self.with_tag else None
        }
    def decode_sentence(self, state, sentence):
        assert len(state)==len(sentence), "state and sentence must have the same length"
        pre=0
        words=[]
        state=list(map(self.decode_state, state))
        for i in range(1,len(state)):
            # print(state[i]['pos'])
            if state[i]['pos']=='B' or state[i]['pos']=='S':
                # assert state[i-1]['pos']=='E' or state[i-1]['pos']=='S'
                word=sentence[pre:i]
                words.append({'word': word, 'tag': state[i-1]['tag'], 'prop': 'undefined' if self.with_tag else None})
                pre=i
        word=sentence[pre:]
        words.append({'word': word, 'tag': state[-1]['tag'], 'prop': 'undefined' if self.with_tag else None})
        return words