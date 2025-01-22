# from vmamba import VSSM
from vmamba_unet_skip import VSSM
import torch
from torch import nn
from torchinfo import summary

class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=9,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = VMUNet(depths=[2, 2, 2, 1], depths_decoder=[1, 2, 2, 2], drop_path_rate=0.2, load_ckpt_path='/home/ubuntu/dch/VM-UNet/pre_trained_weights/vmamba_small_e238_ema.pth')
    model = model.to(device)
    model.load_from()
    # 初始化总参数量为0
    from thop import profile  ## 导入thop模块

    summary(model, input_size=(1, 3, 224, 224))
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input,))
    print('模型计算量：flops', flops / 1e9)  ## 打印计算量
    print('模型可训练参数量：params', params / 1e6)  ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("模型总参数量: %.2fM" % (total / 1e6))

    print(model(input).shape)



    iterations = 400  # 重复计算的轮次


    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(100):
        _ = model(input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f} ms, FPS: {} ".format(mean_time, 1000 / mean_time))
