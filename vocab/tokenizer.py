import sentencepiece as spm

VOCAB_PREFIX = "/opt/data/private/linkdom/data/libritext/vocab/librispeech"

class BaseTokenizer:
    def encode(self, s):
        raise NotImplementedError
    
    def decode(self, ids, ignore_repeat=False):
        raise NotImplementedError
    
    @property
    def vocab_size(self):
        raise NotImplementedError
    
    @property
    def token_type(self):
        raise NotImplementedError
    
    
    @property
    def pad_idx(self):
        return 0
    
    @property
    def eos_idx(self):
        return 1
    
    @property
    def unk_idx(self):
        return 2
    
    @property
    def bos_idx(self):
        return 3
    
    def __repr__(self):
        return "<{} vocab_size={} token_type={}>".format(type(self).__name__, self.vocab_size, self.token_type)

class Tokenizer(BaseTokenizer):
    def __init__(self, model_type: str):
        """
        Usage
        -----

        >>> from vocab.tokenizer import Tokenizer
        >>> tokenizer = Tokenizer('char')
        >>> print(tokenizer.encode('hello world'))
        [3, 10, 4, 14, 14, 7, 3, 18, 7, 12, 14, 13]
        """
        assert model_type in ['bpe', 'char', 'word', 'unigram']
        self.model_type = model_type
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(f'{VOCAB_PREFIX}_{model_type}.model')
        """
        Please train with the following settings:
        --pad_id=0 --eos_id=1 --unk_id=2 --bos_id=3 --eos_piece=<eos>
        """

    def encode(self, s):
        # add bos and eos tokens
        return [self.bos_idx] + self.spm.encode_as_ids(s) + [self.eos_idx]

    def decode(self, idxs):
        return self.spm.decode_ids(idxs)

    @property
    def vocab_size(self):
        return len(self.spm)

    @property
    def token_type(self):
        return self.model_type
    
