import sentencepiece as spm

text_file = '/opt/data/private/linkdom/data/libritext/train/transcript.txt'
model_type = 'char'
model_prefix = f'/opt/data/private/linkdom/data/libritext/vocab/librispeech_{model_type}'

# Load the sentencepiece model
sp = spm.SentencePieceProcessor()

# Load the trained model
sp.load(f'{model_prefix}.model')

test_samples = [
    "hello world",
    "this is a test",
]

# Encode the text
encoded_ids = sp.encode_as_ids(test_samples)
print(f"Encoded text ids   : {encoded_ids}")
encoded_pieces = sp.encode_as_pieces(test_samples)
print(f"Encoded text pieces: {encoded_pieces}")

# Decode the text
decoded_text = sp.decode_ids(encoded_ids)
print(f"Decoded text: {decoded_text}")

# Get the size of the vocabulary
vocab_size = sp.get_piece_size()
print(f"Vocabulary size: {vocab_size}")

# Get the list of the vocabulary
# vocab_list = [sp.id_to_piece(i) for i in range(vocab_size)]
# print(vocab_list)

"""
(GPT) root@asr:~/minasr# python dev.py 
Encoded text ids   : [[3, 10, 4, 14, 14, 7, 3, 18, 7, 12, 14, 13], [3, 5, 10, 9, 11, 3, 9, 11, 3, 6, 3, 5, 4, 11, 5]]
Encoded text pieces: [['▁', 'h', 'e', 'l', 'l', 'o', '▁', 'w', 'o', 'r', 'l', 'd'], ['▁', 't', 'h', 'i', 's', '▁', 'i', 's', '▁', 'a', '▁', 't', 'e', 's', 't']]
Decoded text: ['hello world', 'this is a test']
Vocabulary size: 30
"""