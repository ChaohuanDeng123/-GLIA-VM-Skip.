import torch
import torch.nn as nn
from mamba_ssm import Mamba

class DFR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba_att_1 = Mamba(d_model=1)
        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x1, x2):
        B, H, W, C = x1.shape
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.GAP(x1)
        x1 = x1.view(B, C, -1)
        att_1 = self.mamba_att_1(x1)

        att_1 = att_1.view(B, 1, 1, C)
        out = x2 * att_1

        return out

if __name__ == "__main__":

    model = DFR()
    model.cuda()
    # 初始化总参数量为0

    input_1 = torch.randn(1, 7, 7, 768).cuda()
    input_2 = torch.randn(1, 7, 7, 768).cuda()
    output = model(input_1, input_2)
    pass