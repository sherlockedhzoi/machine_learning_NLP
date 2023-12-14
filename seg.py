from math import log
from utils import Base

class Segmentor(Base):
    def __init__(self, predictor, word_dict, with_tag=True):
        self.save_hyperparameters()
        self.predict=(predictor is not None)
        
    
    def get_DAG(self, sentence):
        to=[]
        for i in range(len(sentence)):
            self.word_dict.reset()
            now=i
            to.append([])
            while now<len(sentence) and self.word_dict.step(sentence[now]):
                if self.word_dict.reach_end():
                    to[i].append((now, self.word_dict.get_now_freq()))
                now+=1
        return to

    def calc_max(self, to):
        log_total=log(self.word_dict.get_total_freq())
        max_to=[None]*len(to)
        for u in range(len(to)-1,-1,-1):
            max_to[u]=min((-log(freq)+log_total+(max_to[v+1][0] if v+1<len(to) else 0),v) for v,freq in to[u]) if len(to[u]) else (0,u)
        return max_to

    def forward(self, line):
        line=line.strip()
        l=0
        N=len(line)
        to=self.get_DAG(line)
        max_to=self.calc_max(to)
        buf=''
        words=[]
        while l<N:
            r=max_to[l][1]+1
            now_word=line[l:r]
            if self.word_dict.find(now_word) is not None:
                if buf:
                    predict_result=self.predictor.predict(buf)
                    words+=predict_result
                    buf=''
                words.append({
                    'word': now_word,
                    'tag': self.word_dict.get_tag(now_word) if self.with_tag else None,
                    'prop': self.word_dict.get_prop(now_word) if self.with_tag else None
                })
            else:
                buf+=now_word
            l=r
        if buf:
            predict_result=self.predictor.predict(buf)
            words+=predict_result
            buf=''
        return words