from pred import HMMPredictor
from ds import Dataset
from dic import LetterDict
from param import HyperParam

ds_url='data\\training\\*_training.utf8'
letter_dict_url='data\\word.dic'
letter_dict=LetterDict(letter_dict_url)
# print(letter_dict.get_id('我'), letter_dict.get_letter(1000))
segmentor_hyper_param=HyperParam(T=1000, N=4, M=len(letter_dict))
ds=Dataset(ds_url, letter_dict, without_mark=False)
# print(ds.get_data()[0])
pred=HMMPredictor(ds, segmentor_hyper_param)
pred.train()
seg=Segmentor(pred, dict_url='data/dict.csv')
print(seg.forward('我想穿蓝色的衣服去散步。'))



