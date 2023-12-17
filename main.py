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
parser.add_argument('--retrain', action='store_true', help='whether retrain model')
parser.add_argument('--atom', default='letter', help='whether predict on letters')
parser.add_argument('--eval', action='store_true', help='if output evaluate result')
parser.add_argument('--supervised', action='store_true', help='if use supervised model')
parser.add_argument('--only_pred', action='store_true', help='if evaluate on the predict model or segmentation model')
arg=parser.parse_args()
assert arg.atom in ['letter', 'word'], 'atom must be letter or word'

letter_dict_url='data/letters.dic'
word_dict_url='data/words.csv'
tag_dict_url='data/tag.csv'
if arg.supervised:
    train_ds_url=['data/PeopleDaily199801.txt']
    test_ds_url=['data/self_made_train.txt','data/self_made_test.txt']
else:
    train_ds_url=['data/self_made_train.txt']
    test_ds_url=['data/self_made_test.txt']

letter_dict=LetterDict(letter_dict_url)
word_dict=WordDict(word_dict_url)
tag_dict=TagDict(tag_dict_url)
encoder_decoder=Coder(letter_dict, word_dict, tag_dict if arg.with_tag else None)
ds=Dataset(train_ds_url, test_ds_url, arg.atom, arg.supervised)
print('Data load complete.')

if arg.atom=='letter':
    hyper_param=HyperParam(T=ds.get_maxlen(), N=4*(len(tag_dict) if arg.with_tag else 1), M=len(letter_dict))
else:
    hyper_param=HyperParam(T=ds.get_maxlen(), N=len(tag_dict), M=len(word_dict))
if arg.retrain:
    pred=HMMPredictor(hyper_param.N, hyper_param.M, hyper_param.T, encoder_decoder, ds=ds, atom=arg.atom, supervised=arg.supervised)
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
    print('Average sentence accuracy:', evaluate(pred if arg.only_pred else seg, ds, encoder_decoder))


test_datas=ds.get_test_data()
for sentence, detagged in test_datas:
    # print(sentence, detagged)
    words=seg.predict(detagged)
    print(encoder_decoder.words2sentence(words))
    extr.load(words)
    print(extr.get_rank(3))
