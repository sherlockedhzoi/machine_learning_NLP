import numpy as np
from ds import Dataset
from numpy.random import standard_normal

class HMMPredictor:
    def __init__(self, ds, hyper_param):
        self.ds=ds
        self.hyper_param=hyper_param
        self.A=standard_normal((self.hyper_param.N, self.hyper_param.N))
        self.B=standard_normal((self.hyper_param.N, self.hyper_param.M))
        self.pi=standard_normal(self.hyper_param.N)

    def get_alpha(self, O):
        alpha=[]
        alpha.append([self.pi[i]*self.B[i][O[0]] for i in range(self.hyper_param.N)])
        for t in range(1,self.hyper_param.T+1):
            alpha.append([self.B[i][O[t]]*sum([alpha[t-1][j]*self.A[j][i] for j in range(self.hyper_param.N)]) for i in range(self.hyper_param.N)])
        return alpha

    def get_beta(self, O):
        beta=[]
        beta.append([1 for i in range(self.hyper_param.N)])
        for t in range(self.hyper_param.T-1, -1, -1):
            beta.append([sum([self.A[i][j]*self.B[j][O[t+1]]*beta[-1][j] for j in range(self.hyper_param.N)]) for i in range(self.hyper_param.N)])
        return beta.reverse()

    def get_gamma(self, alpha, beta, O):
        gamma=[]
        for t in range(self.hyper_param.T):
            s=sum([alpha[t][i]*beta[t][i] for i in range(self.hyper_param.N)])
            gamma.append([alpha[t][i]*beta[t][i]/s[t] for i in range(self.hyper_param.N)])
        return gamma

    def get_xi(self, alpha, beta, O):
        xi=[]
        for t in range(self.hyper_param.T-1):
            s=sum([alpha[t][i]*self.A[i][j]*self.B[j][O[t+1]]*beta[t+1][j] for i,j in range(self.hyper_param.N,self.hyper_param.N)])
            xi.append([[alpha[t][i]*self.A[i][j]*self.B[j][O[t+1]]*beta[t+1][j]/s for j in range(self.hyper_param.N)] for i in range(self.hyper_param.N)])
        return xi

    def step(self, O):
        alpha=self.get_alpha(O)
        beta=self.get_beta(O)
        gamma=self.get_gamma(alpha, beta, O)
        xi=get_xi(alpha, beta, O)
        self.A=[[sum([xi[t][i][j] for t in range(self.hyper_param.T)])/sum([gamma[t][i] for t in range(self.hyper_param.T)]) for j in range(self.hyper_param.N)] for i in range(self.hyper_param.N)]
        self.B=[[sum([(gamma[t][j] if O[t]==k else 0) for t in range(self.hyper_param.T)])/sum([gamma[t][j] for t in range(self.hyper_param.T)]) for k in range(self.hyper_param.M)] for j in range(self.hyper_param.N)]
        self.pi=gamma[1]

    def train(self):
        datas=self.ds.get_data()
        if self.ds.without_mark:
            for sentence in datas:
                self.step(np.array(sentence))
        else:
            for sentence in datas:
                self.step(np.array(sentence))
                self.pi[sentence[0][1]]+=1
                for i in range(len(sentence)-1):
                    self.A[sentence[i][1]][sentence[i+1][1]]+=1
                for i in range(len(sentence)):
                    self.B[sentence[i][1]][sentence[i][0]]+=1
            self.A=[self.A[i]/sum(self.A[i]) for i in range(self.hyper_param.N)]
            self.B=[self.B[i]/sum(self.B[i]) for i in range(self.hyper_param.N)]
            self.pi=self.pi/sum(self.pi)
        
    
    def predict(self, O):
        p=np.ones([len(O), self.hyper_param.N])
        p=[[self.pi[i]*self.B[i][O[0]] for i in range(self.hyper_param.N)]]
        backpointers=[[]]
        for t in range(1, len(O)):
            b=self.B[:,O[t]]
            v=np.expand_dims(p[t-1], axis=1)*self.A
            p.append(np.max(v,0)*b)
            backpointers.append(np.argmax(v,0))
        
        viterbi=[np.argmax(p[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi=[self.map_i[v] for v in viterbi]
        viterbi_score=np.max(p[-1])
        return viterbi, viterbi_score
