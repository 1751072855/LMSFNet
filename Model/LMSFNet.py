import numbers
import os
import random
from datetime import datetime
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from Src.utils.Dataloader import get_loader, test_dataset
from utils.utils import clip_gradient
from lib.pvtv2 import pvt_v2_b2


# -----------------------------
# SELA 注意力
# -----------------------------
class SELA(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(SELA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        weight = self.se(x)
        return x * weight


# -----------------------------
# MS_SSEP 小目标增强模块
# -----------------------------
class MS_SSEP(nn.Module):
    def __init__(self, in_channels, out_channels, sobel=True, lambda_edge=0.2):
        super(MS_SSEP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.se = SELA(out_channels, out_channels)

        self.sobel = sobel
        self.lambda_edge = lambda_edge

        # Sobel 卷积核（L1 归一化）
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32) / 8.0
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32) / 8.0
        self.register_buffer('sobel_x', kx.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', ky.view(1, 1, 3, 3))

    def forward(self, x_low, x_high):
        """
        和你原来的调用方式一致：
            x2_t = x2_t + self.ssep_2(x1_down, x2_t)
        所以这里参数顺序是 (x_low, x_high)
        """
        x = torch.cat([x_low, x_high], dim=1)  # [B, 2C, H, W]
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.se(x)

        if self.sobel:
            # 边缘引导
            gray = x.mean(1, keepdim=True)
            kx = self.sobel_x.to(x.dtype)
            ky = self.sobel_y.to(x.dtype)
            gray_pad = F.pad(gray, (1, 1, 1, 1), mode="replicate")
            gx = F.conv2d(gray_pad, kx)
            gy = F.conv2d(gray_pad, ky)
            edge = torch.sqrt(gx * gx + gy * gy + 1e-6)

            B = edge.shape[0]
            e_flat = edge.view(B, -1)
            k = (e_flat.shape[1] * 95) // 100 + 1
            q95 = e_flat.kthvalue(k, dim=1).values.view(B, 1, 1, 1)
            edge_n = (edge / (q95 + 1e-6)).clamp(0, 1).detach()

            x = x * (1.0 + self.lambda_edge * edge_n)

        return x


class GLCF(nn.Module):
    def __init__(self, channel):
        super(GLCF, self).__init__()
        self.att_local = nn.Sequential(
            nn.Conv2d(channel * 4, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.att_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel * 4, channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel * 4, 1),
            nn.Sigmoid()
        )
        self.global_reduce = nn.Conv2d(channel * 4, channel, 1)  # 👈 添加这一层

    def forward(self, x4, x3, x2, x1):
        x_all = torch.cat([
            F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=True),
            F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True),
            F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True),
            x1
        ], dim=1)  # shape: [B, channel*4, H, W]

        local_feat = self.att_local(x_all)
        global_weight = self.att_global(x_all)
        global_feat = x_all * global_weight  # [B, channel*4, H, W]

        global_feat = self.global_reduce(global_feat)  # 👈 降维到 channel

        fused = local_feat + global_feat
        return fused


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        # 保持你原来的初始化方式
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps  # 添加epsilon参数，避免除零

    def forward(self, x):
        """
        前向传播：带可学习偏置和权重的LayerNorm
        输入x的shape为 [B, N, C]（来自LayerNorm类的reshape）
        """
        # 对最后一维（特征维度C）进行归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # LayerNorm核心计算
        x = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习的权重和偏置
        x = self.weight * x + self.bias

        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # 当前的x.shape的值是 [16, 256, 96, 96]
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class Interaction(nn.Module):
    # dim 是输入特征图的通道数 256
    # num_heads 注意力头的数量 4
    # bias 就是卷积层是否使用偏置项  false
    def __init__(self, dim, num_heads, bias):
        super(Interaction, self).__init__()
        self.num_heads = num_heads
        # 这里定义了一个用于控制注意力值的缩放因子
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 这里就是进行三个1*1的卷积层 分别用来生成 K,Q,V特征图
        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 这里的三个卷积层是为了对查询、K,Q,V进行更高层次的特征转换，使用groups=dim
        # 意味着每个通道独立进行卷积操作。
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 将输出特征图的通道数恢复到dim

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        # 减少输出的通道数，压缩为原来的四分之一
        self.compress = nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, bias=bias)
        # 层归一化，用来对输入特征图进行标准化
        self.norm = LayerNorm(dim)

    def forward(self, x):
        # 这是Interaction中的x.shape torch.Size([12, 256, 96, 96])
        b, c, h, w = x.shape
        # x是输入的特征图，大小为[batchsize,channels,height,width] 对其进行归一化
        x = self.norm(x)
        # 输出x还是[12,256,96,96]，但是每个通道的每个像素值会根据该通道的均值和标准差
        # 进行归一化处理

        # 通过qkv_0、qkv_1、qkv_2卷积层 分别查询q 、k 、 v 特征图
        # 这里的x还是 12，256，96，96 经过1*1的卷积，然后经过3*3的卷积
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        # 调整张量形状
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 注意力加权
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 残差连接 + 压缩
        out = self.project_out(out) + x
        out = self.compress(out)

        return out


class LMSFNet(nn.Module):
    """
    - backbone: pvt_v2_b2，输出 x1,x2,x3,x4
    - Translayer*：统一到 fuse_channels 维度
    - MSSEP：x1->x2, x1->x3 的小目标增强
    - 融合：x1, x2', x3', x4' 上采样到同一尺寸 concat，再 conv 融合 -> seg_head 输出
    - 最后再上采样到输入大小
    """

    def __init__(self,
                 pretrained_pvt_path=None,
                 out_channels=1,
                 fuse_channels=64,
                 use_mssep=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        self.use_mssep = use_mssep
        self.fuse_channels = fuse_channels

        # --- 加载 PVT 预训练 ---
        if pretrained_pvt_path is not None:
            try:
                state_dict = torch.load(pretrained_pvt_path, map_location="cpu")
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
                print(f"[Backbone] Loaded PVTv2-B2 weights from {pretrained_pvt_path}")
            except Exception as e:
                print(f"[Backbone] WARNING: failed to load PVTv2-B2 weights: {e}")

        # pvt_v2_b2 输出通道: [64, 128, 320, 512]
        self.trans1 = nn.Conv2d(64, fuse_channels, kernel_size=1)
        self.trans2 = nn.Conv2d(128, fuse_channels, kernel_size=1)
        self.trans3 = nn.Conv2d(320, fuse_channels, kernel_size=1)
        self.trans4 = nn.Conv2d(512, fuse_channels, kernel_size=1)

        # MSSEP（只在 use_mssep=True 的时候用）
        if self.use_mssep:
            self.ssep_2 = MS_SSEP(fuse_channels, fuse_channels)
            self.ssep_3 = MS_SSEP(fuse_channels, fuse_channels)

        # 多尺度融合：4 个尺度 concat 后卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_channels * 5, fuse_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.intra = Interaction(64 * 4, 4, False)
        self.glcf = GLCF(64)
        # segmentation head
        self.seg_head = nn.Conv2d(fuse_channels, out_channels, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # 1) backbone 提取多尺度特征
        x1, x2, x3, x4 = self.backbone(x)  # [B,64,H/4,W/4], [B,128,H/8,W/8]...

        # 2) 通道统一
        x1_t = self.trans1(x1)  # [B,C,H/4,W/4]
        x2_t = self.trans2(x2)  # [B,C,H/8,W/8]
        x3_t = self.trans3(x3)  # [B,C,H/16,W/16]
        x4_t = self.trans4(x4)  # [B,C,H/32,W/32]

        # 3) MSSEP 小目标增强
        if self.use_mssep:
            # x1 -> x2
            x1_down2 = F.interpolate(x1_t, size=x2_t.shape[2:], mode='bilinear', align_corners=True)
            x2_t = x2_t + self.ssep_2(x1_down2, x2_t)

            # x1 -> x3
            x1_down3 = F.interpolate(x1_t, size=x3_t.shape[2:], mode='bilinear', align_corners=True)
            x3_t = x3_t + self.ssep_3(x1_down3, x3_t)

        # 4) 多尺度交互：全部上采样到 x1_t 的分辨率，再 concat
        x_qkv = self.intra(torch.cat((x1, F.interpolate(x2_t, size=x1.size()[2:], mode='bilinear'),
                                      F.interpolate(x3_t, size=x1.size()[2:], mode='bilinear'),
                                      F.interpolate(x4_t, size=x1.size()[2:], mode='bilinear')), 1))
	#5)上下文融合
        x_share = self.glcf(x4_t, x3_t, x2_t, x1)
        x_share = x_qkv + x_share
        target_size = x1_t.shape[2:]  # H/4, W/4
        f1 = x1_t
        f2 = F.interpolate(x2_t, size=target_size, mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3_t, size=target_size, mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4_t, size=target_size, mode='bilinear', align_corners=True)

        feats_cat = torch.cat([f1, f2, f3, f4, x_share], dim=1)  # [B, 4C, H/4, W/4]
        fused = self.fuse_conv(feats_cat)  # [B, C, H/4, W/4]

        # 6) 输出预测 + 上采样到输入大小
        logits = self.seg_head(fused)  # [B, 1, H/4, W/4]
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)  # [B,1,H,W]

        return logits
