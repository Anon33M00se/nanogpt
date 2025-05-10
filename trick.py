#
# This is the mathematical "trick" at the core of attention
#

import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2  #Batch, Time, Channel
x = torch.randn(B,T,C)
x.shape
