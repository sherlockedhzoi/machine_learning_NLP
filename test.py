import pandas as pd

letter=[]
words=pd.read_csv("data/dict.csv", header=None, encoding='utf-8').iloc[:,0]
for word in words:
    for letter in word:
        letters.append(letter)
with open("data/training/*.utf8", 'r', encoding='utf-8') as f:
    
