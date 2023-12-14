from pred import HMMPredictor
from ds import Dataset
from dic import LetterDict, WordDict
from seg import Segmentor
from code import Coder
from glob import glob
from extractor import TextRank
from utils import to_sentence, evaluate, HyperParam
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--with_tag', action='store_true', help='whether dataset contain tags')
parser.add_argument('--not_divided', action='store_true', help='whether dataset separated')
parser.add_argument('--retrain', action='store_true', help='whether retrain model')
arg=parser.parse_args()
assert not (arg.with_tag and arg.not_divided), 'cannot use both --with_tag and --not_divided'

letter_dict_url='data/letters.dic'
word_dict_url='data/words.csv'
ds_url=['data/PeopleDaily199801.txt'] if arg.with_tag else glob('data/training/*.utf8')

letter_dict=LetterDict(letter_dict_url)
word_dict=WordDict(word_dict_url)
encoder_decoder=Coder(letter_dict, word_dict, with_tag=arg.with_tag)
if arg.retrain:
    train_ds=Dataset(ds_url, encoder_decoder, not_divided=arg.not_divided, test_length=100)
    hyper_param=HyperParam(T=ds.get_train_length, N=4*(110 if arg.with_tag else 1), M=len(letter_dict))
    print('Data load complete.')
    pred=HMMPredictor(hyper_param.N, hyper_param.M, hyper_param.T, encoder_decoder, train_ds)
    pred.train()
    print('Predictor training complete.')
    pred.save()
else:
    hyper_param=HyperParam(T=724, N=4*(110 if arg.with_tag else 1), M=len(letter_dict))
    pred=HMMPredictor(hyper_param.N, hyper_param.M, hyper_param.T, encoder_decoder)
    print('Model load complete.')

seg=Segmentor(pred, word_dict, with_tag=arg.with_tag)
extr=TextRank(window=2)

print('loss: ', evaluate(pred, test_ds))

with open('test.txt','r', encoding='utf-8') as f:
    lines=f.readlines()
    for line in lines:
        words=seg.forward(line)
        print(to_sentence(words))
        extr.load(words)
        print(extr.get_rank(3))