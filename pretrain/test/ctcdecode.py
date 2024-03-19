import torch
from torchaudio.models.decoder import cuda_ctc_decoder
from vocab.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)
vocab_size = tokenizer.vocab_size
tokens = [tokenizer.spm.id_to_piece(id) for id in range(vocab_size)]

nbest = 3
cuda_decoder = cuda_ctc_decoder(tokens, nbest=nbest, beam_size=10, blank_skip_threshold=0.95)
print(len(tokens))


T = 20      # Input sequence length
C = vocab_size   # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

log_prob = torch.randn(N, T, C).log_softmax(-1).detach().requires_grad_() # (T, N, C)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long) # (N, S)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long) # (N,)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long) # (N,)

log_prob = log_prob.to(device)
input_lengths = input_lengths.to(torch.int32).to(device)

# note that log_prob is expected to be in shape (N, T, C)
results = cuda_decoder(log_prob, input_lengths)

for sample in range(N):
    for i in range(nbest):
        transcript = tokenizer.decode(results[sample][i].tokens).lower()
        score = results[0][i].score
        print(f"sample {sample}, nbest {i}, transcript: {transcript}, score: {score}, words: {words}")

"""
(GPT) root@asr:~/minasr# python ctcdecode.py 
1000
sample 0, nbest 0, transcript: mind op cle somet ream whoosed childever contentort very beh other gi stoodeening, score: -86.61978149414062
sample 0, nbest 1, transcript: mind pleas cle somet ream whoosed childever contentort very beh other gi stoodeening, score: -86.6214370727539
sample 0, nbest 2, transcript: mind op cle somet ream whoosedpleever contentort very beh other gi stoodeening, score: -86.6270980834961
sample 1, nbest 0, transcript: a before min is lawignause being same arm sou inc down howeverough lo blackange may then, score: -86.61978149414062
sample 1, nbest 1, transcript: a before min is lawignause being same arm sou inc down howeverough lo blackuck may then, score: -86.6214370727539
sample 1, nbest 2, transcript: a beforeall is lawignause being same arm sou inc down howeverough lo blackange may then, score: -86.6270980834961
sample 2, nbest 0, transcript: allyaken ansou suppopselfterself replddenour high light prallol still de gra, score: -86.61978149414062
sample 2, nbest 1, transcript: allyaken ansou suppopselfterself replddenever high light prallol still de gra, score: -86.6214370727539
sample 2, nbest 2, transcript: allyaken ansou suppopselfterselfaringddenour high light prallol still de gra, score: -86.6270980834961
sample 3, nbest 0, transcript: ast king hisause countryut guisslt ha unc otherunginedosed hadich daysause going, score: -86.61978149414062
sample 3, nbest 1, transcript: ast king hisause countryut guisslt ha unc otherunginedosed hadich days mat going, score: -86.6214370727539
sample 3, nbest 2, transcript: ast king hisause countryut guisslf ha unc otherunginedosed hadich daysause going, score: -86.6270980834961
sample 4, nbest 0, transcript: har hel so lady god understilyvenop until hundred even that fallissngatedouseult did, score: -86.61978149414062
sample 4, nbest 1, transcript: har hel so lady god understilyvenop until hundred even that fallissngatedryult did, score: -86.6214370727539
sample 4, nbest 2, transcript: har hel so lady take understilyvenop until hundred even that fallissngatedouseult did, score: -86.6270980834961
sample 5, nbest 0, transcript: hundred itland certain sker pli to himself soon belie law neverac of high pro felt, score: -86.61978149414062
sample 5, nbest 1, transcript: hundred it pur certain sker pli to himself soon belie law neverac of high pro felt, score: -86.6214370727539
sample 5, nbest 2, transcript: hundred ifland certain sker pli to himself soon belie law neverac of high pro felt, score: -86.6270980834961
sample 6, nbest 0, transcript: donsowishonelingiver som though momu smallqulandaint beg behtersulman, score: -86.61978149414062
sample 6, nbest 1, transcript: donsowishonelingiver som though momu smallqulandaint beg behters airman, score: -86.6214370727539
sample 6, nbest 2, transcript: donsowishonelingiver som though momu smallultlandaint beg behtersulman, score: -86.6270980834961
sample 7, nbest 0, transcript: full eng pointip con de quvery lay repullceptx un hor donab go justage, score: -86.61978149414062
sample 7, nbest 1, transcript: full eng pointip con de quvery lay repullceptx un horolab go justage, score: -86.6214370727539
sample 7, nbest 2, transcript: full eng pointip great de quvery lay repullceptx un hor donab go justage, score: -86.6270980834961
sample 8, nbest 0, transcript: is harorn will justoneyph thhed water stand fact can herselfans cameen what first being, score: -86.61978149414062
sample 8, nbest 1, transcript: is harorn will justoneyph thhed water stand fact can herselfans cameen whatance being, score: -86.6214370727539
sample 8, nbest 2, transcript: is harorn will justoneyph layhed water stand fact can herselfans cameen what first being, score: -86.6270980834961
sample 9, nbest 0, transcript: oub cr sypp patways orideittle whe right car beg bro tre called bec criedactthing, score: -86.61978149414062
sample 9, nbest 1, transcript: oub se sypp patways orideittle whe right car beg bro tre called bec criedactthing, score: -86.6214370727539
sample 9, nbest 2, transcript: oub cr sypp patways orideittle wheition car beg bro tre called bec criedactthing, score: -86.6270980834961
sample 10, nbest 0, transcript: ved by stoodath momited happ car interest small disried other cogedqu leftake between, score: -86.61978149414062
sample 10, nbest 1, transcript: ved by stoodath momited happ car ret small disried other cogedqu leftake between, score: -86.6214370727539
sample 10, nbest 2, transcript: ved by stoodath momited happ car interestad disried other cogedqu leftake between, score: -86.6270980834961
sample 11, nbest 0, transcript: prinissject mother r childaken instween he let lawheft cur other famseint ple, score: -86.61978149414062
sample 11, nbest 1, transcript: prinissject mother r childaken instween he let lawheft comp other famseint ple, score: -86.6214370727539
sample 11, nbest 2, transcript: prinissject mother r childaken instween he gone lawheft cur other famseint ple, score: -86.6270980834961
sample 12, nbest 0, transcript: apsies herselfmost ent capalsens se called with work so madephount whileghwn after, score: -86.61978149414062
sample 12, nbest 1, transcript: apsiesissmost ent capalsens se called with work so madephount whileghwn after, score: -86.6214370727539
sample 12, nbest 2, transcript: apsies herselfmost ent capalsens se called with work so madephount while repliedwn after, score: -86.6270980834961
sample 13, nbest 0, transcript: waou womanpt outothernded most followween gi pres f any natountedft by side, score: -86.61978149414062
sample 13, nbest 1, transcript: waou womanpt outothernded most followween giip f any natountedft by side, score: -86.6214370727539
sample 13, nbest 2, transcript: waou womanpt outothernded most followween bo pres f any natountedft by side, score: -86.6270980834961
sample 14, nbest 0, transcript: whichouenceiver interest souaut whichs another cons new light woman began impasondden morningaught, score: -86.61978149414062
sample 14, nbest 1, transcript: whichouenceiver interest souaut whichs another cons new light woman began impasondden morningfect, score: -86.6214370727539
sample 14, nbest 2, transcript: whichouenceiver interest souaut whichsfect cons new light woman began impasondden morningaught, score: -86.6270980834961
sample 15, nbest 0, transcript: verertving arm feltark sn spend st con poss best ter armale incple world after, score: -86.61978149414062
sample 15, nbest 1, transcript: veriving arm feltark sn spend st con poss best ter armale incple world after, score: -86.6214370727539
sample 15, nbest 2, transcript: verertving arm feltark sn spend st con poss best ter armale incple worldful, score: -86.6270980834961
"""
