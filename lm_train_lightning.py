import torch
import pytorch_lightning as L
from lm.model import LibriGPT
from lm.datasets import LibriTextDataModule
from vocab.tokenizer import Tokenizer
from pytorch_lightning.loggers import TensorBoardLogger

vocab_type = 'bpe'
tokenizer = Tokenizer(vocab_type)

gpt_kwargs = {
    "vocab_size": tokenizer.vocab_size,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "block_size": 512,
    "dropout": 0.2
}

# train config
num_epochs = 20
eval_interval = 500
eval_iters = 200
save_begin = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = f"{vocab_type}_gpt"

loader_kwargs = {
    "tokenizer": tokenizer,
    "batch_size": 128,
    "num_workers": 4
}

datasets = {
    "train_set": "train",
    "dev_set": "dev",
    "test_set": "test"
}

# sample config
sample_kwargs = {
    "max_new_tokens": 200, # max number of tokens to generate
    "temperature": 0.8, # > 1.0 = more exploratory, < 1.0 = more conservative
    "top_k": 200 # consider only the top_k most likely tokens, clamp others to have 0 probability
}

# ------------
torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
model = LibriGPT(tokenizer, gpt_kwargs, sample_kwargs, learning_rate=learning_rate)
dm = LibriTextDataModule(**datasets, **loader_kwargs)
logger = TensorBoardLogger("logs", name="libritext")
# training
trainer = L.Trainer(
    accelerator="gpu",
    devices=[0, 1],
    precision="16-mixed",
    logger=logger,
    num_nodes=1,
    max_epochs=20,
    profiler="simple"
)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

