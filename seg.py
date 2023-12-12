from math import log
from param import Base

class Segmentor(Base):
    def __init__(self, predictor, word_dict):
        self.save_hyperparameters()
        self.predict=(predictor is not None)
        
    
    def get_DAG(self, sentence):
        to=[]
        for i in range(len(sentence)):
            self.word_dict.reset()
            now=i
            to.append([])
            while self.word_dict.step(sentence[now]):
                if self.word_dict.reach_end():
                    to[i].append((now, self.word_dict.get_attr()['freq']))
                now+=1
        return to

    def calc_max(self, to):
        log_total=log(self.word_dict.total_freq)
        max_to=[None]*len(to)
        for u in range(len(to)-1,-1,-1):
            max_to[u]=min((-log(freq)+log_total+max_to[v+1][0],v) for v,freq in to[u]) if len(to[u]) else (0,u)
        return max_to

    def forward(self, sentence):
        l=0
        N=len(sentence)
        to=self.get_DAG(sentence)
        max_to=self.calc_max(to)
        buf=''
        words=[]
        p=0
        while l<N:
            r=max_to[l][1]+1
            p+=max_to[l][0]
            now_word=sentence[l:r]
            if self.word_dict.find(now_word) is not None:
                if buf:
                    predict_result=self.predictor.predict(buf)
                    words.append(word for word in predict_result)
                    buf=''
                words.append({
                    'word': now_word,
                    'tag': self.word_dict.get_tag(now_word),
                    'prop': self.word_dict.get_prop(now_word)
                })
            else:
                buf+=now_word
            l=r
        if buf:
            predict_result=self.predictor.predict(buf)
            print(predict_result)
            words+=predict_result
            buf=''
        print(words)
        return words