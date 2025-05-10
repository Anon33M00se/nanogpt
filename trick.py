#
# This is the mathematical "trick" at the core of attention
#

import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2  #Batch, Time, Channel
x = torch.randn(B,T,C)
x.shape
print(x)


#
# Lump together whatever came before into an average
#


xbow = torch.zeros(B,T,C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)


print("xbow:")
print(xbow[0])

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
# print(wei)

xbow2 = wei @ x

print(xbow2[0])

print(torch.allclose(xbow, xbow2, 0.001))


# print(xbow - xbow2)
#
# Ok, so the "trick" is that we can be very efficient about doing this
# using matrix multiplication
#

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a/torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b


# print("a=")
# print(a)
# print("---")
#
# print("b")
# print(b)
# print("---")

# print("c=")
# print(c)
# print("---")


#
#  Rewritten for the 3rd time, but here we use softmax
#
from torch.nn import functional as F

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

print("xbow3:")
print(xbow3[0])

