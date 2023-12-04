from pred import HMMPredictor
from ds import Dataset
from dic import LetterDict, WordDict
from param import HyperParam
from seg import Segmentor
from code import Coder
from glob import glob
from extractor import TextRank
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--ds_url', default=['data/PeopleDaily199801.txt'], help='dataset url')
parser.add_argument('--with_tag', default=True, help='whether dataset contain tags')
arg=parser.parse_args()

letter_dict_url='data/word.dic'
word_dict_url='data\\dict.csv'
ds_url=arg.ds_url

letter_dict=LetterDict(letter_dict_url)
word_dict=WordDict(word_dict_url)
encoder_decoder=Coder(letter_dict, word_dict, with_tag=arg.with_tag)
# print(encoder_decoder.decode_tag(32))
ds=Dataset(ds_url, encoder_decoder, not_divided=False)
hyper_param=HyperParam(T=ds.maxlen, N=4*(110 if arg.with_tag else 1), M=len(letter_dict))
print('Data load complete.')

# print(letter_dict.get_id('我'), letter_dict.get_letter(1000))
# print(ds.get_data()[0])
pred=HMMPredictor(ds, hyper_param.N, hyper_param.M, hyper_param.T)
pred.train()
print('Predictor Training Complete.')
seg=Segmentor(pred, word_dict)
extr=TextRank(lim=2)

with open('test.txt','r', encoding='utf-8') as f:
    lines=f.readlines()
    for line in lines:
        words=seg.forward(line)
        print(words)
        # extr.load(words)
        # print(words, extr.get_rank()[:3], sep='\n')