from param import Base
import numpy as np
from utils import important_pairs, categorize

class TextRank(Base):
    def __init__(self, window=2, loop_lim=50):
        self.save_hyperparameters()

    def load(self, words):
        self.n=len(words)
        R=np.diag([self.r(word) for word in words])
        A=np.ones((self.n,self.n))
        D=np.array([self.w(word1,word2) for word1 in words for word2 in words]).reshape((self.n,self.n))
        W=np.array([D[i]/sum(D[i]) for i in range(self.n)])
        self.M=R@(W-1/self.n*A)+1/self.n*A

        self.p=np.ones(self.n)/self.n
        for i in range(self.loop_lim):
            self.p=self.p@self.M
        
        self.ranked=sorted([(p,word['word']) for p,word in zip(self.p, words)], reverse=True)
        
    def get_rank(self, num=10):
        return self.ranked[:num]
    
    def r(self, word):
        return 0.9 if word['prop']=='undefined' else 0.5
    
    def w(self, word1, word2):
        return important_pairs.get(categorize(word1['tag'])+'->'+categorize(word2['tag']), 1) if word1['tag'] and word2['tag'] else 1
