import torch
from torch.nn.modules.utils import _pair
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.modules.utils import _pair

from mamba_ssm import Mamba


class DWConv_dch2(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_dch2, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_dch2(hidden_features//2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        res = torch.clone(x)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x + res
        return x



class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:

        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
# Space_Mamba
class MultiHeadMamba_S(nn.Module):
    def __init__(self, d_model, head=8):
        super().__init__()
        self.mamba_1 = Mamba(d_model=d_model // 8)
        self.mamba_2 = Mamba(d_model=d_model // 8)
        self.mamba_3 = Mamba(d_model=d_model // 8)
        self.mamba_4 = Mamba(d_model=d_model // 8)
        self.mamba_5 = Mamba(d_model=d_model // 8)
        self.mamba_6 = Mamba(d_model=d_model // 8)
        self.mamba_7 = Mamba(d_model=d_model // 8)
        self.mamba_8 = Mamba(d_model=d_model // 8)
        self.mamba = [self.mamba_1, self.mamba_2, self.mamba_3, self.mamba_4, self.mamba_5, self.mamba_6, self.mamba_7, self.mamba_8]
        self.head = head

    def forward(self, x):
        B, L, C = x.shape
        x = x.view(B, L, self.head, C // self.head)
        self.head_list = []
        for i in range(self.head):
            input = (x[:, :, i, :].view(B, L, C // self.head))
            output = self.mamba[i](input)

            self.head_list.append(output)
        y = torch.concat(self.head_list, dim=2)
        return y






class VSSBlock_S(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        # self.ln_1 = norm_layer(hidden_dim)
        # self.ln_2 = norm_layer(hidden_dim)
        # self.ln_3 = norm_layer(hidden_dim)
        self.self_attention_1 = MultiHeadMamba_S(d_model=hidden_dim)
        self.self_attention_2 = MultiHeadMamba_S(d_model=hidden_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        B, L, C = input.shape
        # x = self.ln_1(input)
        x1 = input.view(B, -1, C)
        x2 = torch.flip(x1, dims=[1])


        y1 = self.self_attention_1(x1)
        # y1 = y1.permute(0, 2, 1)
        y2 = self.self_attention_2(x2)
        y2 = torch.flip(y2, dims=[-1])

        # y2 = y2.view(B, H, W, C)
        # y2 = F.silu((self.ln_2(y2)))
        ## 不同序列信息如何融合是一个问题？
        x = y1 + y2
        input = input.view(B, L, C)
        x = input + x
        return x
        
class Space_Mamba(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.space_mamba_1 = VSSBlock_S(hidden_dim=96)
        # self.space_mamba_1 = VSSBlock_S(hidden_dim=96)
        # self.projection_in = nn.Linear(in_features=1440, out_features=196)
        # self.projection_out = nn.Linear(in_features=196, out_features=1440)

    def forward(self, x_list_1):
        x = self.space_mamba_1(x_list_1)
        return x



class Global_Infor(torch.nn.Module):
    def __init__(self, dims=96):
        super().__init__()
        self.SM = Space_Mamba()

    def forward(self, x):
        B, H, W, C = x[0].shape
        x_embed_0 = x[0].view(B, -1, C)
        x_embed_1 = x[1].view(B, -1, C)
        x_embed_2 = x[2].view(B, -1, C)
        x_embed_3 = x[3].view(B, -1, C)
        x = torch.cat([x_embed_0, x_embed_1, x_embed_2, x_embed_3], -2)
        y = x + self.SM(x)
        return x, y

class Local_Infor(torch.nn.Module):
    def __init__(self, dims=96):
        super().__init__()
        # self.mixffn1 = MixFFN_skip(dims,dims//8)
        # self.mixffn2 = MixFFN_skip(dims*2,dims//4)
        # self.mixffn3 = MixFFN_skip(dims*4,dims//2)
        # self.mixffn4 = MixFFN_skip(dims*8,dims)
        self.mixffn1 = ConvolutionalGLU(dims,dims//16)
        self.mixffn2 = ConvolutionalGLU(dims*2,dims//8)
        self.mixffn3 = ConvolutionalGLU(dims*4,dims//4)
        self.mixffn4 = ConvolutionalGLU(dims*8,dims//2)
        # self.mixffn1 = MLP_FFN(dims,dims*2)
        # self.mixffn2 = MLP_FFN(dims*2,dims*4)
        # self.mixffn3 = MLP_FFN(dims*4,dims*8)
        # self.mixffn4 = MLP_FFN(dims*8,dims*16)

    def forward(self, x):
        B, L, C = x.shape
        tem1 = x[:,:3136,:].reshape(B, -1, C)
        tem2 = x[:,3136:4704,:].reshape(B, -1, C*2)
        tem3 = x[:,4704:5488,:].reshape(B, -1, C*4)
        tem4 = x[:,5488:5880,:].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        # m1f = self.mixffn1(tem1).reshape(B, -1, C)
        # m2f = self.mixffn2(tem2).reshape(B, -1, C)
        # m3f = self.mixffn3(tem3).reshape(B, -1, C)
        # m4f = self.mixffn4(tem4).reshape(B, -1, C)
        y = torch.cat([m1f, m2f, m3f, m4f], -2)


        return y

    def forward(self, x):
        B, _, _, C = x[0].shape
        tem1 = x[0].reshape(B, -1, C)
        tem2 = x[1].reshape(B, -1, C*2)
        tem3 = x[2].reshape(B, -1, C*4)
        tem4 = x[3].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 56, 56).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 28, 28).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 14, 14).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 7, 7).reshape(B, -1, C)

        # m1f = self.mixffn1(tem1).reshape(B, -1, C)
        # m2f = self.mixffn2(tem2).reshape(B, -1, C)
        # m3f = self.mixffn3(tem3).reshape(B, -1, C)
        # m4f = self.mixffn4(tem4).reshape(B, -1, C)
        y = torch.cat([m1f, m2f, m3f, m4f], -2)


        return y


class Global_Local(torch.nn.Module):
    def __init__(self, dims=96):
        super().__init__()
        self.global_infor = Global_Infor()
        self.norm = nn.LayerNorm(dims)
        self.local_infor = Local_Infor()

    def forward(self, x):
        res, x = self.global_infor(x)
        x = self.norm(x)
        res_1 = x
        y = self.local_infor(x)
        y = res + res_1 + y
        B, L, C = y.shape
        y0 = y[:,:3136,:].reshape(B, 56, 56, C)
        y1 = y[:,3136:4704,:].reshape(B, 28, 28, C*2)
        y2 = y[:,4704:5488,:].reshape(B, 14, 14, C*4)
        y3 = y[:,5488:5880,:].reshape(B, 7, 7, C*8)
        return [y0, y1, y2, y3]
