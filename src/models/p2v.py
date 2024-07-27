# TODO: make norm3d module types changeable in temporal branch.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import resnet
from ._blocks import Conv1x1, Conv3x3, MaxPool2x2, get_norm_layer
from ._utils import Identity, KaimingInitMixin
import matplotlib.pyplot as plt
import numpy as np
import cv2

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))


class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        return self.conv_fuse(x)


class BasicConv3D(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            kernel_size,
            bias='auto',
            bn=False, act=False,
            **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.ConstantPad3d(kernel_size // 2, 0.0))
        seq.append(
            nn.Conv3d(
                in_ch, out_ch, kernel_size,
                padding=0,
                bias=(False if bn else True) if bias == 'auto' else bias,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.BatchNorm3d(out_ch))
        if act:
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv3x3x3(BasicConv3D):
    def __init__(self, in_ch, out_ch, bias='auto', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, bias=bias, bn=bn, act=act, **kwargs)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Conv3x3x3(itm_ch, itm_ch, bn=True, act=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y + res)
        return y


class Backbone(nn.Module, KaimingInitMixin):
    def __init__(
            self,
            in_ch, out_ch=32,
            arch='resnet18',
            pretrained=True,
            n_stages=5
    ):
        super().__init__()

        expand = 1
        strides = (2, 1, 2, 1, 1)
        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self.n_stages = n_stages

        if self.n_stages == 5:
            itm_ch = 512 * expand
        elif self.n_stages == 4:
            itm_ch = 256 * expand
        elif self.n_stages == 3:
            itm_ch = 128 * expand
        else:
            raise ValueError

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_out = Conv3x3(itm_ch, out_ch)

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        y = self.resnet.conv1(x)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)

        y = self.resnet.layer1(y)
        y = self.resnet.layer2(y)
        y = self.resnet.layer3(y)
        y = self.resnet.layer4(y)

        y = self.upsample(y)

        return self.conv_out(y)

    def _trim_resnet(self):
        if self.n_stages > 5:
            raise ValueError

        if self.n_stages < 5:
            self.resnet.layer4 = Identity()

        if self.n_stages <= 3:
            self.resnet.layer3 = Identity()

        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


class PairEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(16, 32, 64), add_chs=(0, 0)):
        super().__init__()

        self.n_layers = 3

        self.conv1 = SimpleResBlock(2 * in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = SimpleResBlock(enc_chs[0] + add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock(enc_chs[1] + add_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()

        self.backbone = Backbone(3, 128, arch='resnet34', n_stages=5)

    def forward(self, x1, x2, add_feats=None):
        x = torch.cat([x1, x2], dim=1)
        feats = [x]

        for i in range(self.n_layers):
            conv = getattr(self, f'conv{i + 1}')
            if i > 0 and add_feats is not None:
                add_feat = F.interpolate(add_feats[i - 1], size=x.shape[2:])
                x = torch.cat([x, add_feat], dim=1)
            x = conv(x)
            pool = getattr(self, f'pool{i + 1}')
            x = pool(x)
            feats.append(x)

        y1 = self.backbone(x1)
        y2 = self.backbone(x2)
        y = F.interpolate(torch.abs(y1 - y2), size=[32, 32])

        # vis=y.mean(dim=1, keepdim=True)
        # vis = F.interpolate(vis, size=(256, 256), mode='bilinear', align_corners=False)
        # print(vis.shape)
        # for i in range(8):
          #  array = vis[i]
          #  maxValue = array.max()
          #  minValue = array.min()
          # array = (array - minValue) / (maxValue - minValue)
          # array = array * 255
          # array1 = array.clone().cpu()
          # array1 = array1.numpy()
          # mat = np.uint8(array1)
          # mat = mat.transpose(1, 2, 0)
          #  mat = cv2.applyColorMap(mat, cv2.COLORMAP_JET)
          # cv2.imwrite(f'{i}.jpg', mat)
          # a = input("zt")

        return feats, y

class PairEncoder2(nn.Module):
    def __init__(self, in_ch, enc_chs=(16, 32, 64), add_chs=(0, 0)):
        super().__init__()

        self.n_layers = 3

        self.conv1 = SimpleResBlock(2 * in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = SimpleResBlock(enc_chs[0] + add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock(enc_chs[1] + add_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()


    def forward(self, x1, x2, mask, add_feats=None):

        x = torch.cat([x1, x2], dim=1)
        x = x * mask
        feats = [x]

        for i in range(self.n_layers):
            conv = getattr(self, f'conv{i + 1}')
            if i > 0 and add_feats is not None:
                add_feat = F.interpolate(add_feats[i - 1], size=x.shape[2:])
                x = torch.cat([x, add_feat], dim=1)
            x = conv(x)
            pool = getattr(self, f'pool{i + 1}')
            x = pool(x)
            feats.append(x)

        return feats

class VideoEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64, 128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = 4
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3, 9, 9), stride=(1, 4, 4), padding=(1, 4, 4), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU()
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(
                enc_chs[0],
                enc_chs[0] * exps,
                enc_chs[0],
                ds=BasicConv3D(enc_chs[0], enc_chs[0] * exps, 1, bn=True)
            ),
            ResBlock3D(enc_chs[0] * exps, enc_chs[0] * exps, enc_chs[0])
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(
                enc_chs[0] * exps,
                enc_chs[1] * exps,
                enc_chs[1],
                stride=(2, 2, 2),
                ds=BasicConv3D(enc_chs[0] * exps, enc_chs[1] * exps, 1, stride=(2, 2, 2), bn=True)
            ),
            ResBlock3D(enc_chs[1] * exps, enc_chs[1] * exps, enc_chs[1])
        )

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i + 1}')
            x = layer(x)
            feats.append(x)

        return feats


class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):
        super().__init__()

        enc_chs = enc_chs[::-1]
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs, (itm_ch,) + dec_chs[:-1], dec_chs)
        ])
        self.conv_out = Conv1x1(dec_chs[-1], 1)

    def forward(self, x, feats):
        feats = feats[::-1]

        x = self.conv_bottom(x)

        for feat, blk in zip(feats, self.blocks):
            x = blk(feat, x)

        y = self.conv_out(x)

        return y


class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32, 64, 128), enc_chs_v=(64, 128), dec_chs=(256, 128, 64, 32)):
        super().__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.encoder_v = VideoEncoder(in_ch, enc_chs=enc_chs_v)
        enc_chs_v = tuple(ch * self.encoder_v.expansion for ch in enc_chs_v)
        self.encoder_p1 = PairEncoder(in_ch, enc_chs=enc_chs_p, add_chs=enc_chs_v)
        self.encoder_p2 = PairEncoder2(in_ch, enc_chs=enc_chs_p, add_chs=enc_chs_v)
        self.conv_out_v = Conv1x1(enc_chs_v[-1], 1)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2 * ch, ch, norm=True, act=True)
                for ch in enc_chs_v
            ]
        )
        self.decoder1 = SimpleDecoder(enc_chs_p[-1], (2 * in_ch,) + enc_chs_p, dec_chs)
        self.decoder2 = SimpleDecoder(enc_chs_p[-1], (2 * in_ch,) + enc_chs_p, dec_chs)

    def create_mask(self, cm):
        B, H, W = cm.shape
        # 黑色是0  白色缺失部分是255   实际使用时，需要将255变为1。 因为需要和原图做加减运算
        mask = torch.zeros((B, H, W))  # 生成一个覆盖全部大小的全黑mask
        for i in range(B):
            if cm[i].sum():
                b = torch.nonzero(cm[i])
                x, _ = b.min(0)
                y, _ = b.max(0)
                mask[i][x[0]:y[0] + 1, x[1]:y[1] + 1] = 1
            else:
                mask[i] = mask[i] + 1
        return mask

    def forward(self, t1, t2, return_aux=True):
        frames = self.pair_to_video(t1, t2)
        #frames1 = frames[0]
        #print(frames1.shape)
        #image = frames1[7].permute(1, 2, 0).cpu().numpy()
        #image = (image * 255).astype(np.uint8)
        #plt.imshow(image)
        #plt.axis('off')
        #height, width, _ = image.shape
        #dpi = 100  # 假设 dpi 为 100
        #fig = plt.gcf()  # 获取当前图形
        #fig.set_size_inches(width / dpi, height / dpi)  # 设置图形大小为图片大小
        #plt.savefig('output_image.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        #plt.close(fig)  # 关闭图形以释放内存
        #print(zt)

        feats_v = self.encoder_v(frames.transpose(1, 2))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))
        pred_v = F.interpolate(self.conv_out_v(feats_v[-1]), [256, 256])
        mask = torch.sigmoid(pred_v)
        mask = (mask > 0.5) + 0.0

        feats_p1, diff1 = self.encoder_p1(t1, t2, feats_v)
        feats_p2 = self.encoder_p2(t1, t2, mask, feats_v)
        # feats_p, diff = self.encoder_p(t1, t2, feats_v)

        pred1 = self.decoder1(diff1, feats_p1)
        pred2 = self.decoder2(feats_p2[-1], feats_p2)
        # pred = 0.5 * pred1 + 0.5 * pred2  #SVCD
        pred = 0.67 * pred1 + 0.33 * pred2  #LEVIR
        # pred = 0.7 * pred1 + 0.3 * pred2  #WHU
        # pred = self.decoder(diff, feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            return pred, pred_v
        else:
            return pred

    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0 / (len - 1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1, -1, 1, 1, 1)
            interped = im1.unsqueeze(1) + ((im2 - im1) * delta_map).unsqueeze(1) * steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:, 0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        return frames

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)
