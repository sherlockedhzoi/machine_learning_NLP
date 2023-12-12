err=1e-10
def to_sentence(words):
    line=''
    for word in words:
        line+=word['word']+'/'+word['tag']+(
            '('+','.join(word['prop'])+')' if word['prop']!='undefined' else '')+' '
    return line

important_pairs={
    'a->n': 10,
    'uj->n': 10,
    'n->uj': 0,
    # ('ad', 'a'): 5,
    # ('ad', 'v'): 5,
    # ('v', 'n'): 2,
    # ('zg', 'n'): 2
}
