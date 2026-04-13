import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ChannelSqueezeBlock(nn.Module):
    """ Squeeze-and-Excitation """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # 输出 0 到 1 的通道权重
        )

    def forward(self, x):
        """ x: (B, C, H, W) """
        B, C, _, _ = x.shape
        # Squeeze: (B, C, H, W) -> (B, C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B, C) -> (B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # Scale: 通道加权
        return x * y.expand_as(x)


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_group = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):



        local_feat = self.conv_group(x)  # (B, C, H, W)


        gate = self.sigmoid(self.gate_conv(x))  # (B, C, H, W)


        gated_feat = local_feat * gate


        out = self.final_conv(gated_feat)

        return out


class GLSSA_UBranch(nn.Module):
    def __init__(self, in_channels, out_channels_r, C=128):
        super().__init__()
        self.r = out_channels_r

        # --- (P -> C) ---
        self.proj_in = ConvBlock(in_channels, C)
        self.initial_attention = ChannelSqueezeBlock(C, reduction=8)  # 初始注意力

        # --- Encoder (Downsampling) ---
        self.enc1 = ConvBlock(C, C)
        self.pool1 = nn.MaxPool2d(2)  # H/2, W/2

        self.enc2 = ConvBlock(C, C * 2)
        self.pool2 = nn.MaxPool2d(2)  # H/4, W/4

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(

            ConvBlock(C * 2, C*4),
            nn.BatchNorm2d(C*4),
            nn.LeakyReLU(inplace=True),

            GatedConvBlock(C * 4, C * 2),
            nn.BatchNorm2d(C*2),
            nn.LeakyReLU(inplace=True),
        )



        # --- Decoder (Upsampling) ---
        # ConvTranspose2d
        self.up2 = nn.ConvTranspose2d(C * 2, C, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(C * 2, C)  # C*2 from concatenation (u2 + e1)

        self.up1 = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(C * 2, C)  # C*2 from concatenation (u1 + x0)

        # --- Final Projection ---
        self.final_attention = ChannelSqueezeBlock(C, reduction=8)  # 最终注意力
        self.final_proj = nn.Conv2d(C, self.r, kernel_size=1)



    def forward(self, Y):
        """ Y: (B, P, H, W) -> HSI Input """

        # 0. Initial Projection + Attention
        x = self.proj_in(Y)
        x0 = self.initial_attention(x)  # 初始特征，也是第一个 Skip Connection

        # --- Encoder ---
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # --- Bottleneck ---
        b = self.bottleneck(p2)
        # b = p2

        # --- Decoder ---
        # 1. Level 2
        u2 = self.up2(b)

        if u2.shape[-2:] != e1.shape[-2:]:

            u2 = F.interpolate(u2, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        x2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(x2)

        # 2. Level 1
        u1 = self.up1(d2)

        if u1.shape[-2:] != x0.shape[-2:]:

            u1 = F.interpolate(u1, size=x0.shape[-2:], mode='bilinear', align_corners=True)
        x1 = torch.cat([u1, x0], dim=1)
        d1 = self.dec1(x1)

        # --- Final Projection ---
        d1 = self.final_attention(d1)
        U = self.final_proj(d1)

        # Abundance Constraints (Non-negativity)
        U = F.relu(U)



        return U


class SELayer1D(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        # Squeeze: (B, C, L) -> (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # 1. Squeeze: (B, C, 1)
        y = self.avg_pool(x).view(b, c)
        # 2. Excitation: (B, C, 1)
        y = self.fc(y).view(b, c, 1)
        # 3. Scale
        return x * y.expand_as(x)


class FourierVBranch_Attn(nn.Module):
    def __init__(self, p, r, hidden_dim=64, reduction=4):
        super().__init__()
        self.p = p
        self.r = r


        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.pre_mlp = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1)
        )


        self.freq_len = p // 2 + 1

        self.complex_weight = nn.Parameter(
            torch.randn(hidden_dim, self.freq_len, 2, dtype=torch.float32) * 0.02
        )


        self.channel_attn = SELayer1D(channel=hidden_dim, reduction=reduction)

        self.post_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, r, 1)  # 输出 R 个端元
        )

    def forward(self, Y):
        """ Y: (B, P, H, W) """
        B, P, H, W = Y.shape

        # 1. (B, P, 1, 1) -> (B, P) -> (B, 1, P)
        x = self.spatial_pool(Y).view(B, 1, P)

        # 2. (B, hidden_dim, P)
        x = self.pre_mlp(x)

        # 3. FFT -> Filtering -> iFFT
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')


        weight = torch.view_as_complex(self.complex_weight)


        x_fft = x_fft * weight


        x_filtered = torch.fft.irfft(x_fft, n=P, dim=-1, norm='ortho')


        x_refined = self.channel_attn(x_filtered)

        V = self.post_mlp(x_refined)


        V = V.permute(0, 2, 1)

        V = F.softplus(V)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-8)

        return V







class Model(nn.Module):
    def __init__(self, p, r, base_dim=128):
        super().__init__()

        self.Abundance_Module = GLSSA_UBranch(in_channels=p, out_channels_r=r, C=base_dim)

        self.SpectralBasis_Module = FourierVBranch_Attn(p=p, r=r, hidden_dim=base_dim, reduction=4)



    def forward(self, y):

        U = self.Abundance_Module(y)
        V = self.SpectralBasis_Module(y)
        Y_hat = torch.einsum('bpr,brhw->bphw', V, U)

        return {'U': U, 'V': V, 'Y_hat': Y_hat}