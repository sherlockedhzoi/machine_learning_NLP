import pandas as pd

class Coder:
    def __init__(self, letter_dict, with_tag=False):
        self.letter_dict = letter_dict
        self.with_tag=with_tag
        if self.with_tag:
            self.tag_dict=pd.read_csv('./data/tag.csv')

    def encode_pos(self, pos):
        assert pos in ['M', 'B', 'E', 'S'], 'state must be M, B, E, or S'
        return {'M': 0, 'B': 1, 'E': 2, 'S': 3}[pos]
    def encode_tag(self, tag):
        assert tag in self.tag_dict['tag'].to_list(), 'tag not defined'
        return (self.tag_dict['tag']==tag).to_numpy().squeeze().nonzero()[0]
    def encode_word(self, word, pos, tag):
        return {
            'id': self.letter_dict.getid(word),
            'tag': encode_pos(pos) if not self.with_tag else encode_tag(tag)*4+encode_pos(pos),
        }
    def encode_sentence(self, line, letter_dict, not_divided=False):
        encoded_sentence=[]
        if not_divided:
            sentence=''.join(line.strip().split('  '))
            # print(line, sentence)
            for letter in sentence:
                encoded_sentence.append(letter_dict.get_id(letter))
        else:
            words=line.strip().split()
            for w in words:
                word, tag=w.split('/')
                if word is None: continue
                elif len(word)==1: 
                    encoded_sentence.append(encode_word(word, 'S', tag))
                else:
                    encoded_sentence.append(encode_word(word[0], 'B', tag))
                    for letter in word[1:-1]:
                        encoded_sentence.append(encode_word(letter, 'M', tag))
                    encoded_sentence.append(encode_word(word[-1], 'E', tag))

        return encoded_sentence

    def decode_pos(self, state):
        assert state in range(4)
        return ['M', 'B', 'E', 'S'][state]
    def decode_tag(self, state):
        assert state in range(39)
        return self.tag_dict.loc[state]['tag']
    def decode_state(self, state):
        return {
            'pos': decode_pos(state%4),
            'tag': decode_tag(state//4) if self.with_tag else None
        }
    def decode_sentence(self, state, word_dict, sentence):
        assert len(state)==len(sentence), "state and sentence must have the same length"
        pre=0
        words=[]
        state=map(decode_state, state)
        for i in range(len(state)):
            if state[i]['pos']==2 or state[i]['pos']==3:
                assert i+1==len(state) or state[i+1]==1 or state[i+1]==3
                word=sentence[pre:i+1]
                words.append((word,))
                pre=i+1
        return words