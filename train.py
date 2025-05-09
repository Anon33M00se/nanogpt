#
#
#  Let's build GPT from scratch
#
#  https://www.youtube.com/watch?v=kCc8FmEb1nY
#
#  Data comes from:
#
#    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

# print(f"length of dataset in characters is {len(text)}")
# print(text[:1000])


#
#  Get unique characters (we are building a character based NN)
#

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)


#
#  chars to ints and vice versa
#


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(x):
    return ''.join([itos[i] for i in x])


# print(encode("hii there"))
# print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

#
# Split tensor encoded dataset into training and validation portions
#

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#
# Set the block size (context size) to 8 and see that plus 1
#

block_size = 8
train_data[:block_size + 1]

#
# See how input and output training data relates to each other
#

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
#    print(f"when input is {context} the target is {target}")


#
# Getting slightly more serious, let's batch this together
#

torch.manual_seed(1337)
# batch_size = 4  # how many independent sequences in parallel
batch_size = 32  # how many independent sequences in parallel


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)

print('targets:')
print(yb.shape)
print(yb)

print("----")

#
# Simplest neural network you can feed this to is a Bigram
#


import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# batch_size = 32

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets == None:
            loss = None
        else:
            # Futz with how torch represents things to massage into cross_entropy input
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # How good was the logits to the expected output?
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


#
# Ok, let's generate something (we haven't done any training ...)
#

idx = torch.zeros((1, 1), dtype=torch.long)  # the most cmplex repr of zero :-)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


#
# oh man did that suck ... we need some training ... start with an optimizer
#

optimizer = torch.optim.AdamW(m.parameters(), lr =1e-3)

# batch_size = 32

for steps in range(10000):
    xb, xy = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())

#
# Let's try that output again
#

idx = torch.zeros((1, 1), dtype=torch.long)  # the most cmplex repr of zero :-)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))



