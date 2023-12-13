from dic import LetterDict, WordDict
from pred import HHMPredictor
from code import Coder
from ds import Dataset
from glob import glob
from utils import HyperParam

letter_dict_url='data/word.dic'
word_dict_url='data/dict.csv'
ds_url=glob('data/training/*.utf8')
letter_dict=LetterDict(letter_dict_url)
word_dict=WordDict(word_dict_url)
encoder_decoder=Coder(letter_dict, word_dict, with_tag=False)
ds=Dataset(ds_url, encoder_decoder, not_divided=arg.not_divided)
hyper_param=HyperParam(T=ds.maxlen, N=4, M=len(letter_dict))
print('Data load complete.')
pred=HHMPredictor(hyper_param.N, hyper_param.M, hyper_param.T, encoder_decoder, ds)
pred.train()
print('Predictor training complete.')
