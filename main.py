from pred import HMMPredictor
from ds import Dataset
from dic import LetterDict, WordDict, TagDict
from seg import Segmentor
from code import Coder
from glob import glob
from extractor import TextRank
from utils import evaluate, HyperParam
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--with_tag', action='store_true', help='whether dataset contain tags')
parser.add_argument('--not_divided', action='store_true', help='whether dataset separated')
parser.add_argument('--retrain', action='store_true', help='whether retrain model')
parser.add_argument('--atom', default='letter', help='whether predict on letters')
parser.add_argument('--eval', action='store_true', help='if output evaluate result')
arg=parser.parse_args()
assert arg.atom in ['letter', 'word'], 'atom must be letter or word'
assert not(not arg.with_tag and not arg.not_divided and arg.atom=='letter'), 'state not supported'
assert not(arg.not_divided and arg.atom=='word'), 'cannot predict on words when not divided'
supervised=(arg.with_tag and not arg.not_divided)

letter_dict_url='data/letters.dic'
word_dict_url='data/words.csv'
tag_dict_url='data/tag.csv'
if supervised:
    ds_url=['data/PeopleDaily199801.txt']
else:
    ds_url=['data/self_made.txt']

letter_dict=LetterDict(letter_dict_url)
word_dict=WordDict(word_dict_url)
tag_dict=TagDict(tag_dict_url)
encoder_decoder=Coder(letter_dict, word_dict, tag_dict if arg.with_tag else None)
ds=Dataset(ds_url, not_divided=arg.not_divided, with_tag=arg.with_tag, test_size=100)
print('Data load complete.')

if arg.retrain:
    if arg.atom=='letter':
        hyper_param=HyperParam(T=ds.get_train_size(), N=4*(len(tag_dict) if arg.with_tag else 1), M=len(letter_dict))
    else:
        hyper_param=HyperParam(T=ds.get_train_size(), N=len(tag_dict), M=len(word_dict))
    pred=HMMPredictor(hyper_param.N, hyper_param.M, hyper_param.T, encoder_decoder, ds=ds, atom=arg.atom, supervised=supervised)
    pred.train(1)
    print('Predictor training complete.')
    pred.save()
else:
    if arg.atom=='letter':
        hyper_param=HyperParam(T=724, N=4*(len(tag_dict) if arg.with_tag else 1), M=len(letter_dict))
    else:
        hyper_param=HyperParam(T=724, N=len(tag_dict), M=len(word_dict))
    pred=HMMPredictor(hyper_param.N, hyper_param.M, hyper_param.T, encoder_decoder, atom=arg.atom)
    print('Model load complete.')

seg=Segmentor(pred, word_dict, with_tag=arg.with_tag)
extr=TextRank(window=2)

if arg.eval:
    print('Average sentence accuracy:', evaluate(seg, ds, encoder_decoder))


test_datas=ds.get_test_data()
for sentence, detagged in test_datas:
    # print(line)
    words=seg.predict(detagged)
    print(encoder_decoder.words2sentence(words))
    extr.load(words)
    print(extr.get_rank(3))
