err=1e-100

def encode_divide(state):
    assert state in ['M', 'B', 'E', 'S'], 'state must be M, B, E, or S'
    return 0 if state=='M' else (1 if state=='B' else (2 if state=='E' else 3))
def encode_sentence(line, letter_dict, without_mark=False):
    encoded_sentence=[]
    if without_mark:
        sentence=''.join(line.strip().split('  '))
        # print(line, sentence)
        for letter in sentence:
            encoded_sentence.append(letter_dict.get_id(letter))
    else:
        words=line.strip().split()
        for word in words:
            if word is None: continue
            elif len(word)==1: 
                encoded_sentence.append((letter_dict.get_id(word),encode_divide('S')))
            else:
                encoded_sentence.append((letter_dict.get_id(word[0]),encode_divide('B')))
                for letter in word[1:-1]:
                    encoded_sentence.append((letter_dict.get_id(letter),encode_divide('M')))
                encoded_sentence.append((letter_dict.get_id(word[-1]),encode_divide('E')))
    return encoded_sentence

def decode_sentence(state, sentence):
    assert len(state)==len(sentence), "state and sentence must have the same length"
    pre=0
    words=[]
    for i in range(len(state)):
        if state[i]==2 or state[i]==3:
            assert i+1==len(state) or state[i+1]==1 or state[i+1]==3
            words.append(sentence[pre:i+1])
            pre=i+1
    return words