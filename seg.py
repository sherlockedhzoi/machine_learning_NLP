from math import log
from dic import Dict

class Segmentor:
    def __init__(self, predictor, dict_url='data/dict.csv'):
        self.dict=Dict(dict_url)
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
        for u in range(len(to),-1,-1):
            max_to[u]=min((-log(freq)+log_total+max_to[v][1],v) for v, freq in to[u])

    def forward(self, sentence):
        l=0
        N=len(sentence)
        to=get_DAG(sentence)
        max_to=calc_max(to)
        while l<N:
            r=max_to[l][1]+1
            now_word=sentence[l:r]
            if r==l+1:
                buf+=now_word
            else:
                if buf:
                    if self.dict.search(buf) is not None:
                        words.append(tuple(buf,self.dict.search(buf)))
                    elif self.predict:
                        predict_datas=self.predictor.predict(buf)
                        for predict_data in predict_datas:
                            words.append(predict_data)
                    else:
                        for word in buf:
                            words.append(tuple(word,self.dict.search(word)))
                words.append(tuple(now_word,self.dict.search(now_word)))
            l=r
        return words