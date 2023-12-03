import numpy as np
from ds import Dataset
from numpy.random import standard_normal
import numpy as np
from utils import err
from math import log, exp
from param import Base

class HMMPredictor(Base):
    def __init__(self, ds, N, M, T):
        self.save_hyperparameters()
        self.letter_dict=self.ds.encoder_decoder.letter_dict
        if self.ds.not_divided:
            self.A=standard_normal((self.N, self.N))
            self.B=standard_normal((self.N, self.M))
            self.pi=standard_normal(self.N)
        else:
            self.A=np.ones((self.N, self.N))
            self.B=np.ones((self.N, self.M))
            self.pi=np.ones(self.N)

    def get_alpha(self, O):
        alpha=[]
        alpha.append([self.pi[i]*self.B[i][O[0]] for i in range(self.N)])
        for t in range(1,self.T+1):
            alpha.append([self.B[i][O[t]]*sum([alpha[t-1][j]*self.A[j][i] for j in range(self.N)]) for i in range(self.N)])
        return alpha

    def get_beta(self, O):
        beta=[]
        beta.append([1 for i in range(self.N)])
        for t in range(self.T-1, -1, -1):
            beta.append([sum([self.A[i][j]*self.B[j][O[t+1]]*beta[-1][j] for j in range(self.N)]) for i in range(self.N)])
        return beta.reverse()

    def get_gamma(self, alpha, beta, O):
        gamma=[]
        for t in range(self.T):
            s=sum([alpha[t][i]*beta[t][i] for i in range(self.N)])
            gamma.append([alpha[t][i]*beta[t][i]/s[t] for i in range(self.N)])
        return gamma

    def get_xi(self, alpha, beta, O):
        xi=[]
        for t in range(self.T-1):
            s=sum([alpha[t][i]*self.A[i][j]*self.B[j][O[t+1]]*beta[t+1][j] for i,j in range(self.N,self.N)])
            xi.append([[alpha[t][i]*self.A[i][j]*self.B[j][O[t+1]]*beta[t+1][j]/s for j in range(self.N)] for i in range(self.N)])
        return xi

    def step(self, O):
        alpha=self.get_alpha(O)
        beta=self.get_beta(O)
        gamma=self.get_gamma(alpha, beta, O)
        xi=get_xi(alpha, beta, O)
        self.A, self.B=[], []
        for i in range(self.N):
            self.A.append([sum([xi[t][i][j] for t in range(self.T)])/sum([gamma[t][i] for t in range(self.T)]) for j in range(self.N)])
        for j in range(self.N):
            self.B.append([sum([(gamma[t][j] if O[t]==k else 0) for t in range(self.T)])/sum([gamma[t][j] for t in range(self.T)]) for k in range(self.M)])
        self.pi=gamma[1]
        print(self.A, self.B, self.pi, sep='\n')

    def train(self):
        datas=self.ds.get_data()
        # print(self.A.shape, self.B.shape, self.pi.shape)
        if self.ds.not_divided:
            for sentence in datas:
                self.step(np.array(sentence))
        else:
            for sentence in datas:
                # print(len(sentence))
                self.pi[sentence[0]['tag']]+=1
                for i in range(len(sentence)-1):
                    # print(i, sentence[i]['tag'],sentence[i+1]['tag'],self.A[sentence[i]['tag']][sentence[i+1]['tag']])
                    self.A[sentence[i]['tag']][sentence[i+1]['tag']]+=1
                for i in range(len(sentence)):
                    self.B[sentence[i]['tag']][sentence[i]['id']]+=1
            
            self.A=np.array([(self.A[i]/sum(self.A[i]) if sum(self.A[i]) else self.A[i]) for i in range(self.N)])
            self.B=np.array([(self.B[i]/sum(self.B[i]) if sum(self.B[i]) else self.B[i]) for i in range(self.N)])
            self.pi=self.pi/sum(self.pi)
    
    def predict(self, sentence, begin_pos=0):
        # print(sentence)
        O=self.ds.encoder_decoder.encode_sentence(sentence, not_divided=True)
        # print(O)
        # print(self.A, self.B[:,O[0]], self.pi, sep='\n')
        logp=[np.array([-np.log(self.pi[i])-np.log(self.B[i][O[0]]) for i in range(self.N)])]
        frm=[np.array([])]
        for t in range(1, len(O)):
            b=self.B[:,O[t]]
            nowlogp=(np.expand_dims(logp[t-1], axis=0)-np.log(self.A.T)-np.expand_dims(np.log(b),axis=1))
            logp.append(nowlogp.min(axis=1))
            frm.append(nowlogp.argmin(axis=1))
        
        # for i in range(len(logp[-1])):
        #     if logp[-1][i]!=logp[-1].max():
        #         print(logp[-1])
        endstate=logp[-1].argmin()
        I=[endstate]
        for t in range(len(O)-1,0,-1):
            endstate=frm[t][endstate]
            I.append(endstate)
        I=I[::-1]
        # for i in I:
        #     print(self.ds.encoder_decoder.decode_state(i), end=' ')
        # print()
        return self.ds.encoder_decoder.decode_sentence(I, sentence)