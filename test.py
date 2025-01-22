import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset import RandomGenerator
from engine_synapse import *

from models.vmunet.vmunet import VMUNet

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1, 2, 3"
# torch.cuda.set_device(1)
from utils import *
from configs.config_setting_synapse import setting_config

import warnings

warnings.filterwarnings("ignore")

def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')


    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')


    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                            batch_size=1,  # if config.distributed else config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            sampler=val_sampler,
                            drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else:
        raise ('Please prepare a right net!')

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model.cuda()





    print('#----------Testing----------#')
    best_weight = torch.load("/home/ubuntu/dch/VM-UNet/results/vmunet_synapse_baseline(en+de)+[vmamba_unet_skip]/checkpoints/best-epoch315-mean_dice0.8198-mean_hd9521.4543.pth", map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    model.eval()
    with torch.no_grad():
        data = next(iter(val_loader))


        img, msk, case_name = data['image'], data['label'], data['case_name'][0]

        image, label = img.squeeze(0).cpu().detach().numpy(), msk.squeeze(0).cpu().detach().numpy()
        if len(image.shape) == 3:
            prediction = np.zeros_like(label)
            ind = 90
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != 224 or y != 224:
                slice = zoom(slice, (224 / x, 224 / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            model.eval()
            with torch.no_grad():
                outputs = model(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != 224 or y != 224:
                    pred = zoom(out, (x / 224, y / 224), order=0)
                else:
                    pred = out
                prediction[ind] = pred





if __name__ == '__main__':
    config = setting_config
    main(config)