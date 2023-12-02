from math import log
from dic import WordDict

class Segmentor:
    def __init__(self, predictor, dict_url='data/dict.csv'):
        self.dict=WordDict(dict_url)
        self.predict=(predictor is not None)
        self.predictor=predictor
        
    
    def get_DAG(self, sentence):
        to=[]
        for i in range(len(sentence)):
            self.dict.reset()
            now=i
            to.append([])
            while self.dict.step(sentence[now]):
                if self.dict.reach_end():
                    to[i].append((now, self.dict.get_attr()['freq']))
                now+=1
        return to

    def calc_max(self, to):
        log_total=log(self.dict.total_freq)
        max_to=[None]*len(to)
        for u in range(len(to)-1,-1,-1):
            # print(u, to[u])
            max_to[u]=min((-log(freq)+log_total+max_to[v+1][0],v) for v,freq in to[u]) if len(to[u]) else (0,len(to))
        return max_to

    def forward(self, sentence):
        l=0
        N=len(sentence)
        to=self.get_DAG(sentence)
        max_to=self.calc_max(to)
        buf=''
        words=[]
        while l<N:
            r=max_to[l][1]+1
            now_word=sentence[l:r]
            if r==l+1:
                buf+=now_word
            else:
                if buf:
                    if self.dict.search(buf) is not None:
                        words.append(buf)
                    elif self.predict:
                        predict_datas=self.predictor.predict(buf)
                        for predict_data in predict_datas:
                            words.append(predict_data)
                    else:
                        for word in buf:
                            words.append(word)
                    buf=''
                words.append(now_word)
            l=r
        return words