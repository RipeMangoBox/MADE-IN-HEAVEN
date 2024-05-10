import torch
import torch.nn.functional as F

a = torch.rand(2, 3, 2, 3)
a[:, :, 1, 1] = 5

b = F.softmax(a, dim=-1)

print(a)
print(b)


prob_dist = F.softmax(b[:, :, i, j], -1)
pixel = torch.multinomial(prob_dist, 1)
x[:, i, j] = pixel[:, 0]