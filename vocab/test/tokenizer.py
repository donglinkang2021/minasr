from vocab.tokenizer import Tokenizer
tokenizer = Tokenizer('char')
print(tokenizer.encode('hello world'))

"""
(GPT) root@asr:~/minasr# python dev2.py 
[-1, 3, 10, 4, 14, 14, 7, 3, 18, 7, 12, 14, 13, 1]
"""

from vocab.tokenizer import Tokenizer
tokenizer = Tokenizer('char')
print(tokenizer.encode(''))

"""
(GPT) root@asr:~/minasr# python dev2.py 
[-1, 1]
"""