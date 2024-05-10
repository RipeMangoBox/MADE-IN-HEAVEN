import torch

def log(vars):
    for var in vars:
        print(var.stride(), var, var.is_contiguous())
        print('-' * 40)

a = t = torch.arange(12).reshape(3,4)
b = a.transpose(0, 1)
c = a.permute(1, 0)

log([a ,b ,c])

v_a = a.view(-1)
log([v_a])

v_b = b.view(-1) # 会报错
log([v_b])

v_c = c.view(-1)
log([v_c])