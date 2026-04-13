import os
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch import nn
from visualization import painting, plot_spectral_curve, plot_spectral_curve1, painting_rl
import matplotlib.pyplot as plt


def _to_numpy_image(X):
    X_np = (
        X.cpu()  # 第一步：将 Tensor 从 GPU 复制到 CPU
        .squeeze(0)  # 第二步：去除批次维度（B=1），形状变为 (C, H, W)
        .permute(1, 2, 0)  # 第三步：调整维度为 (H, W, C)
        .numpy()  # 第四步：转为 NumPy 数组
    )
    return X_np.astype(np.float32)

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * torch.log10(target.max() / torch.sqrt(mse))

def spectral_angle_mapper(pred, target):
    # pred, target: (B, P, H, W)
    B, P, H, W = pred.shape
    pred_flat = pred.reshape(B, P, -1)
    target_flat = target.reshape(B, P, -1)
    dot = (pred_flat * target_flat).sum(dim=1)
    denom = (pred_flat.norm(dim=1) * target_flat.norm(dim=1) + 1e-8)
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang.mean() * 180 / np.pi

def ssim(pred, target, window_size=11):
    """
    Compute SSIM for hyperspectral image (band-wise average).
    pred, target: [B, P, H, W]
    """
    B, P, H, W = pred.shape
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu_x2
    sigma_y2 = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


def spectral_tv_loss(Y_hat):

    diff1 = torch.abs(Y_hat[:, 1:, :, :] - Y_hat[:, :-1, :, :])
    spectral_tv = torch.mean(diff1)

    return spectral_tv


import torch
from torch.optim import Adam

def train_SSFD(model, Y_noisy, Y_clean, alpha=0.1, mask_ratio=0.5,lr=1e-3,
               gamma=0.9, max_iter=2000, tol=1e-4):
    """
    Y:  [B, P, H, W]
    """

    optimizer_U = Adam(model.Abundance_Module.parameters(), lr=lr)
    optimizer_V = Adam(model.SpectralBasis_Module.parameters(), lr=lr)

    out_prev = Y_noisy.clone()
    Y_tgt= Y_noisy.clone()
    loss_log, psnr_log, ssim_log, sam_log = [], [], [], []

    max_psnr = 0
    max_ssim = 0
    min_sam = 0
    iter = 0

    for t in range(max_iter):
        optimizer_U.zero_grad()
        optimizer_V.zero_grad()





        mask = (torch.rand_like(Y_noisy) < mask_ratio).float()
        Y_masked = Y_noisy * (1 - mask)

        model.train()
        out = model(Y_masked)
        Y_hat = out['Y_hat']
        U, V = out['U'], out['V']







        loss_mask = F.l1_loss(Y_hat[mask.bool()], Y_tgt[mask.bool()])
        image_spectral_tv = spectral_tv_loss(Y_hat)
        loss = (
                loss_mask +
                alpha * image_spectral_tv
        )


        loss.backward()

        optimizer_U.step()
        optimizer_V.step()


        out_smooth = gamma * out_prev + (1 - gamma) * Y_hat.detach()
        rel_err = torch.norm(out_smooth - out_prev) / (torch.norm(out_prev) + 1e-8)
        out_prev = out_smooth

        cur_psnr = psnr(out_smooth, Y_clean).item()
        cur_sam = spectral_angle_mapper(out_smooth, Y_clean).item()
        cur_ssim = ssim(out_smooth, Y_clean).item()
        if cur_psnr>=max_psnr:
            max_psnr = cur_psnr
            max_ssim = cur_ssim
            min_sam = cur_sam
            iter  = t + 1

        loss_log.append(loss.item())
        psnr_log.append(cur_psnr)
        sam_log.append(cur_sam)
        ssim_log.append(cur_ssim)

        if (t + 1) % 100  == 0 :
            print(f"[Iter {t + 1:4d}] "
                  f"Loss={loss.item():.6f} | PSNR={cur_psnr:.2f} | "
                  f"SSIM={cur_ssim:.3f} | SAM={cur_sam:.3f}° | RelErr={rel_err:.2e} | loss_mask={loss_mask.item():.5f}"
                  f" | image_spectral_tv={image_spectral_tv.item():.5f} ")


        if rel_err < tol:
            print(f"Converged early at iter {t + 1} (RelErr={rel_err:.2e})")
            break

    print(f"[Iter {iter :4d}] "
          f"MaxPSNR={max_psnr:.2f} | MaxSSIM={max_ssim:.3f} | MinSAM={min_sam:.3f}° ")


    return out_prev, U, V,loss_log, psnr_log, ssim_log, sam_log




if __name__ == "__main__":
    import numpy as np
    from model import Model
    import scipy.io as sio
    import time



    mat_path = 'data/noisy/WDC_f.mat'

    mat = sio.loadmat(mat_path)
    print(mat.keys())
    Y_clean = mat['gt']
    Y_noisy = mat['noisy']


    Y_noisy = torch.from_numpy(Y_noisy).unsqueeze(0).float().permute(0, 3, 1, 2)  # (1, P, H, W)
    Y_clean = torch.from_numpy(Y_clean).unsqueeze(0).float().permute(0, 3, 1, 2)
    Y_noisy = Y_noisy.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = Y_noisy.device
    Y_clean = Y_clean.to(device)
    C, P, H, W = Y_clean.shape


    model = Model(p=P, r=24, base_dim=256).cuda()

    start = time.time()
    Y_denoised, U, V, loss_log, psnr_log, ssim_log, sam_log = train_SSFD(model, Y_noisy, Y_clean, alpha=0.1,
                                                                             lr=1e-3, max_iter=900, tol=1e-4)
    end = time.time()

    elapsed_time = end - start
    print(f"执行时间: {elapsed_time:.4f} 秒")

    psnr_before = psnr(Y_noisy, Y_clean)
    ssim_before = ssim(Y_noisy, Y_clean)
    sam_before = spectral_angle_mapper(Y_noisy, Y_clean)

    psnr_after = psnr(Y_denoised, Y_clean)
    ssim_after = ssim(Y_denoised, Y_clean)
    sam_after = spectral_angle_mapper(Y_denoised, Y_clean)

    print(f"PSNR: {psnr_before:.2f} → {psnr_after:.2f}")
    print(f"SSIM: {ssim_before:.3f} → {ssim_after:.3f}")
    print(f"SAM : {sam_before:.3f}° → {sam_after:.3f}°")


    Y_noisy = _to_numpy_image(Y_noisy)
    Y_clean = _to_numpy_image(Y_clean)
    Y_denoised = _to_numpy_image(Y_denoised)

    # painting(Y_noisy, Y_clean, Y_denoised, 50)
    # plot_spectral_curve1(Y_noisy, Y_clean, Y_denoised, coord=(150, 150))



