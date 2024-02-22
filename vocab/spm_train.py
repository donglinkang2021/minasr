import sentencepiece as spm

SPM_TYPE=[
    'unigram',
    'bpe',
    'char',
    'word',
]

text_file = '/opt/data/private/linkdom/data/libritext/transcript.txt'
model_type = SPM_TYPE[3]
model_prefix = f'/opt/data/private/linkdom/data/libritext/vocab/librispeech_{model_type}'
idx_setting = '--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 --eos_piece=<eos>'

# Train the sentencepiece model
spm.SentencePieceTrainer.train(f'--input={text_file} --model_prefix={model_prefix} --model_type={model_type} {idx_setting} --vocab_size=1000')