from param import Base
from utils import important_pairs
import editdistance as word_dis

class TextRank(Base):
    def __init__(self, lim=5, d=0.85, maxt=50):
        self.save_hyperparameters()
    
    def init(self, words):
        self.N=len(words)
        self.words=[word['word'] for word in words]
        self.in_edge=[{} for _ in range(self.N)]
        self.out_edge=[{} for _ in range(self.N)]
        self.in_deg=[0]*self.N
        self.out_deg=[0]*self.N

    def get_weight(self, word_from, word_to):
        return word_dis.eval(word_from, word_to)

    def add(self, i, j, word_i, word_j):
        self.in_edge[i][j]=self.out_edge[j][i]=self.get_weight(word_j, word_i)
        self.in_edge[j][i]=self.out_edge[i][j]=self.get_weight(word_i, word_j)
        self.in_deg[i]+=self.get_weight(word_j, word_i)
        self.out_deg[i]+=self.get_weight(word_i, word_j)
        self.in_deg[j]+=self.get_weight(word_i, word_j)
        self.out_deg[j]+=self.get_weight(word_j, word_i)

    def load(self, words):
        self.init(words)
        for i in range(len(words)):
            last=self.lim
            j=i-1
            while j>=0 and last>0:
                self.add(i,j, words[i]['word'], words[j]['word'])
                last-=len(words[i]['word'])
                j-=1
        self.rank()
    
    def rank(self):
        scores=[1]*self.N
        for t in range(self.maxt):
            tmp_scores=scores.copy()
            for i in range(self.N):
                nowsum=0
                for j in self.in_edge[i].keys():
                    print(self.words[i], self.words[j], self.out_edge[j][i], self.out_edge[i][j])
                    nowsum+=(self.out_edge[j][i]/self.out_deg[j])*scores[j]
                tmp_scores[i]=nowsum*self.d+1-self.d
            scores=tmp_scores.copy()
        self.ranked_words=sorted([(scores[i], self.words[i]) for i in range(len(scores))], reverse=True)
            
    def get_rank(self):
        return self.ranked_words