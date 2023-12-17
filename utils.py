from editdistance import eval as levenshtein_distance
import inspect

class HyperParam:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
class Base:
    def save_hyperparameters(self, ignore=[]):
        frame=inspect.currentframe().f_back
        _, _, _, values = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in values.items() if k not in set(ignore+['self']) and not k.startswith('_')}
        for k,v in self.hparams.items():
            setattr(self, k, v)

err=1e-2
def detag(sentence, sep=''):
    return ''.join([word.split('[')[-1].split(']')[0].split('/')[0] for word in sentence.split()])

important_pairs={
    'v->n': 2, 'n->v': 2,
    'a->n': 2, 'n->a': 2,
    'ad->v': 2, 'v->ad': 2,
    'ad->a': 2, 'a->ad': 2,
    'nx->n': 0, 'nx->v': 0, 'nx->a': 0, 'nx->ad': 0, 'nx->prep': 0,
}
def categorize(tag):
    if tag in ['in', 'jn', 'ln', 'm', 'mq', 'Ng', 'n', 'nr', 'nrf', 'nrg', 'ns', 'nt', 'nx', 'nz', 'Rg', 'r', 'rr', 'ry', 'ryw', 'rz', 'rzw', 's', 'Tg', 't', 'tt']:
        return 'n'
    elif tag in ['in', 'jv', 'lv', 'Vg', 'v', 'vd', 'vi', 'vl', 'vn', 'vq', 'vu', 'vx']:
        return 'v'
    elif tag in ['Ag', 'a', 'ad', 'an', 'ia', 'ja', 'la']:
        return 'a'
    elif tag in ['Dg', 'd', 'dc', 'df', 'f', 'id', 'jd', 'ld']:
        return 'ad'
    elif tag in ['c', 'p']:
        return 'prep'
    elif tag in ['nx', 'o', 'u', 'ud', 'ue', 'ui', 'ul', 'uo', 'us', 'uz', 'w', 'wd', 'wf', 'wj', 'wk', 'wky', 'wkz', 'wm', 'wp', 'ws', 'wt', 'wu', 'ww', 'wy', 'wyy', 'wyz', 'x', 'y', 'z']:
        return 'nx'
    else:
        return tag

def evaluate_sentence(sentence, correct_sentence):
    return 1-levenshtein_distance(sentence, correct_sentence)/len(correct_sentence)

def evaluate(seg, ds, code):
    test_datas=ds.get_test_batch(batch_size=50)
    sentence_acc, pos_acc, tag_acc=0, 0, 0
    for right_sentence, detagged in test_datas:
        # print(detagged)
        words=seg.predict(detagged)
        sentence=code.words2sentence(words)
        letters=list(code.words2letters(words))
        tags=[word['tag'] for word in words]
        pos=[letter['pos'] for letter in letters]
        
        right_words=list(code.sentence2words(right_sentence))
        right_letters=list(code.words2letters(right_words))
        right_tags=[word['tag'] for word in right_words]
        right_pos=[letter['pos'] for letter in right_letters]

        # print('sentence:', sentence, 'right:', right_sentence)
        # print('words:', words, 'right_words:', right_words)
        # print('letters:', letters, 'right_letters:', right_letters)
        # print('pos:', pos, 'right_pos:', right_pos)
        # print('tags:', tags, 'right_tags:', right_tags)

        pos_acc+=evaluate_sentence(pos, right_pos)
        tag_acc+=evaluate_sentence(tags, right_tags)
        sentence_acc+=evaluate_sentence(sentence, right_sentence)

    pos_acc/=len(test_datas)
    tag_acc/=len(test_datas)
    sentence_acc/=len(test_datas)
    return pos_acc, tag_acc, sentence_acc