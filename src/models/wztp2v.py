# TODO: make norm3d module types changeable in temporal branch.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ._blocks import Conv1x1, Conv3x3, MaxPool2x2
from models.help_funcs import Transformer


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
        return feats


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

    def forward(self, x):   #([8, 3, 8, 256, 256])
        feats = [x]


        x = self.stem(x)  #[8, 64, 8, 64, 64]
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i + 1}')
            x = layer(x) # [8, 256, 8, 64, 64]) [8, 512, 4, 32, 32]
            feats.append(x)

        return feats # len = 3


class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):  # 128  (6, 32, 64, 128)  (256, 128, 64, 32)

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
        # x = self.conv_bottom(x)
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
        self.transformer = Transformer(dim=128, depth=2, heads=8, dim_head=64, mlp_dim=256, dropout=0, softmax=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, 32, 32))

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

        frames = self.pair_to_video(t1, t2)  #[8, 8, 3, 256, 256]
        feats_v = self.encoder_v(frames.transpose(1, 2))

        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        pred_v = F.interpolate(self.conv_out_v(feats_v[-1]), [256, 256])

        mask = torch.sigmoid(pred_v)

        mask = (mask > 0.5) + 0.0
        # mask = (mask > 0.4) + 0.0
        # mask = ((mask > 0.5) + 0).squeeze(dim=1)  # 阈值分割 -> mask
        # mask = self.create_mask(mask).unsqueeze(dim=1).cuda() # 包围盒

        feats_p0 = self.encoder_p1(t1, t2, feats_v)
        feats_p1 = self.encoder_p2(t1, t2, mask, feats_v)

        max_pooling = nn.MaxPool2d(kernel_size=32)
        B, C, H, W = feats_p0[-1].shape
        meta0 = max_pooling(feats_p0[-1]).expand(B,C,H,W)
        meta1 = max_pooling(feats_p1[-1]).expand(B,C,H,W)
        # print(a.shape)
        # print(b.shape)
        # print(zt)

        # feats_p0[-1] = 0.2 * self.CrossAttention(feats_p0[-1], feats_p1[-1]) + 0.8 * feats_p0[-1]
        # feats_p1[-1] = 0.2 * self.CrossAttention(feats_p1[-1], feats_p0[-1]) + 0.8 * feats_p1[-1]

        # pred0 = self.decoder1(feats_p0[-1], feats_p0)
        # pred1 = self.decoder2(feats_p1[-1], feats_p1)
        pred0 = self.decoder1(meta1, feats_p0)
        pred1 = self.decoder2(meta0, feats_p1)
        pred = 0.5 * pred0 + 0.5 * pred1

        if return_aux:
            return pred, pred_v
        else:
            return pred

    def CrossAttention(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding
        # elif self.with_decoder_pos == 'learned':
        #     x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        m = rearrange(m, 'b c h w -> b (h w) c')
        x = self.transformer(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

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
