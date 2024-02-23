import torch
from lm.model import GPT
from vocab.tokenizer import Tokenizer

tokenizer = Tokenizer('bpe')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gpt_kwargs = {
    "vocab_size": tokenizer.vocab_size,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "block_size": 512,
    "dropout": 0.2
}

model = GPT(**gpt_kwargs)
model.to(device)
model.load_state_dict(torch.load("checkpoints/best_bpe_gpt.pth"))

# sample config
sample_kwargs = {
    "max_new_tokens": 500, # max number of tokens to generate
    "temperature": 0.8, # > 1.0 = more exploratory, < 1.0 = more conservative
    "top_k": 200 # consider only the top_k most likely tokens, clamp others to have 0 probability
}

# generate from the model
# just gen from bos token
# context = torch.tensor([tokenizer.spm.bos_id()], dtype=torch.long, device=device).unsqueeze(0)
# print(tokenizer.decode(model.generate(context, **sample_kwargs)[0].tolist()))

# gen from a prompt
text = "how old"
context = torch.tensor(tokenizer.encode(text)[:-1], dtype=torch.long, device=device).unsqueeze(0)
print(tokenizer.decode(model.generate(context, **sample_kwargs)[0].tolist()))