import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
# from lib.SDRNet import Network
from baseline_mssep import PVTv2_MSSEP_Fusion
from torch import nn
from Src.utils.Dataloader import test_dataset1
import cv2
import imageio
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from torch import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 测试时使用单GPU即可
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='')
opt = parser.parse_args()

# 初始化模型
model = model().cuda()

# 加载模型权重并处理多GPU训练的前缀问题
state_dict = torch.load(opt.pth_path)
# 移除参数名中的'module.'前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 从第7个字符开始截取，移除'module.'
    else:
        new_state_dict[k] = v

# 加载处理后的状态字典
model.load_state_dict(new_state_dict)
model.cuda()
model.eval()

# 以下部分保持不变
S = 0
for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
    data_path = ''.format(_data_name)
    save_path = ''.format(opt.pth_path.split('\\')[-2], _data_name+'_S')
    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root', image_root, gt_root)
    test_loader = test_dataset1(
        image_root=image_root,
        gt_root=gt_root,
        edge_root=gt_root,
        testsize=opt.testsize
    )
    print('****', test_loader.size)
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()
            cam= model(image)
            cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=False)
            res = cam.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imsave(save_path + name, img_as_ubyte(res))
