import torch.nn as nn
import torch
from typing import List, Tuple


def merge(*args):
    return torch.cat(args, dim=-1)


def mlp(input_mlp: List[Tuple[int, int]]) -> nn.Sequential:
    if not input_mlp:
        return nn.Sequential()
    mlp_list = []
    for input_dim, out_put_dim, af in input_mlp:
        mlp_list.append(nn.Linear(input_dim, out_put_dim, bias=True))
        if af == "relu":
            mlp_list.append(nn.ReLU())
        if af == 'sigmoid':
            mlp_list.append(nn.Sigmoid())
    return nn.Sequential(*mlp_list)

# linear = mlp([(4, 64)])
# print(linear(torch.zeros([4,4])).shape)