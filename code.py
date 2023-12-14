import pandas as pd
import string
from utils import Base, detag

class Sentence22Words:
    def sentence2words(self, sentence):
        words=sentence.strip().split()
        for w in words:
            if w.strip()=='': continue
            if '/' in w:
                word, tag=w.strip().split('[')[-1].split(']')[0].split('/')
                prop='undefined'
            else:
                word, tag=w.strip(), None
                prop=None
            yield {'word': word, 'tag': tag, 'prop': prop}

    def words2sentence(self, words):
        line=''
        for word in words:
            line+=word['word']+('/'+word['tag'] if word['tag'] else '')+(
                '('+','.join(word['prop'])+')' if word['prop'] and word['prop']!='undefined' else '')+' '
        return line

                
class Words22Letters:
    def words2letters(self, words):
        letter2dict=lambda letter, pos, tag, prop: {'letter': letter, 'pos': pos, 'tag': tag, 'prop': prop}
        for w in words:
            word, tag, prop=w.values()
            for letter in word[1:-1]:
                if len(word)==1: 
                    yield letter2dict(word, 'S', tag, prop)
                else:
                    yield letter2dict(word[0], 'B', tag, prop)
                    for letter in word[1:-1]:
                        yield letter2dict(letter, 'M', tag, prop)
                    yield letter2dict(word[-1], 'E', tag, prop)

    def letters2words(self, letters):
        word2dict=lambda word, tag, prop: {'word': word, 'tag': tag, 'prop': prop}
        buf=''
        for l in letters:
            letter, pos, tag, prop=l.values()
            if pos=='S':
                yield word2dict(letter, tag, prop)
            else:
                buf+=letter
                if pos=='E':
                    yield word2dict(buf, tag, prop)
                    buf=''

class Sentence22Letters:
    def sentence2letters(self, sentence):
        sentence=sentence.strip()
        assert len(sentence.split())==1, 'the sentence must be not splitable'
        for letter in sentence:
            yield {'letter': letter, 'pos': None, 'tag': None, 'prop': None}
    def letters2sentence(self, letters):
        return ''.join([l['letter'] for l in letters])
    
def Coder(Sentence22Words, Words22Letters, Sentence22Letters, Base):
    def __init__(self, letter_dict, word_dict, with_tag=True):
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
        if tag is None: return 0
        else:
            assert tag in self.tag_dict['tag'].to_list(), 'tag %s not defined'%tag
            return (self.tag_dict['tag']==tag).to_numpy().squeeze().nonzero()[0]
    def encode_words(self, word):
        word, tag, prop=word.values()
        return {
            'ID': self.word_dict.get_id(word),
            'tag': self.encode_tag(tag)
        }
    def encode_letter(self, letter):
        letter, pos, tag, prop=letter.values()
        return {
            'ID': self.letter_dict.get_id(letter),
            'tag': self.encode_tag(tag)*4+self.encode_pos(pos)
        }
    def encode_sentence(self, sentence, train=True, not_divided=False, atom='letter'):
        assert atom in ['letter','word'], f'Don\'t accept atoms other than word/letter, but received {atom}'
        if train:
            words=self.sentence2words(sentence)
            letters=self.words2letters(words)
            if atom=='word':
                return list(map(self.encode_words, words))
            else:
                return list(map(self.encode_letter, letters))
        else:
            if atom=='word':
                words=self.sentence2words(sentence)
                words=list(map(self.encode_word, words))
                return [word['ID'] for word in words]
            else:
                letters=self.sentence2letters(sentence)
                letters=list(map(self.encode_letter, letters))
                return [letter['ID'] for letter in letters]

    def decode_pos(self, state) -> str:
        assert state in range(4)
        return ['M', 'B', 'E', 'S'][state]
    def decode_tag(self, state):
        assert state in range(110) and self.tag_dict.iloc[state].loc['tag'] is not None, 'tag has to exist'
        return self.tag_dict.iloc[state].loc['tag']
    def decode_words(self, word, with_tag=True):
        ID, tag=word.values()
        return {
            'word': self.word_dict[ID],
            'tag': self.decode_tag(state) if self.with_tag else None,
            'prop': 'undefined' if self.with_tag else None
        }
    def decode_letter(self, letter, with_tag=True):
        ID, tag=letter.values()
        return {
            'letter': self.letter_dict[ID],
            'pos': self.decode_pos(state%4),
            'tag': self.decode_tag(state//4) if self.with_tag else None,
            'prop': 'undefined' if self.with_tag else None
        }
    def decode_sentence(self, sentence, atom='letter'):
        assert atom in ['letter', 'word'], f'frm has to be either "letter" or "word", but received {atom}'
        assert end in ['word', 'sentence'], f'atom has to be either "word" or "sentence", but received {end}'
        if atom=='letter':
            letters=self.decode_letters(sentence)
            words=self.letters2words(letters)
        else:
            letters=None
            words=self.decode_words(sentence)
        return letters, words, self.words2sentence(words)

    def is_begin(self, tag):
        return self.decode_state(tag)['pos'] in ['B', 'S']
    def get_all_ends(self):
        ends=[]
        for pos in ['E','S']:
            for tag in range(self.tag_cnt):
                ends.append(tag*4+self.encode_pos(pos))
        return sorted(ends)