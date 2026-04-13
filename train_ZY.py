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




        if (t + 1) % 100 == 0:
            Y_noisy1 = _to_numpy_image(Y_noisy)
            out_prev1 = _to_numpy_image(out_prev)
            painting_rl(Y_noisy1, out_prev1, [89, 40, 1], name=f'{t + 1}')


        if rel_err < tol:
            print(f"Converged early at iter {t + 1} (RelErr={rel_err:.2e})")
            break




    return out_prev, U, V,loss_log, psnr_log, ssim_log, sam_log




if __name__ == "__main__":
    import numpy as np
    from model import Model
    import scipy.io as sio



    import numpy as np
    import scipy.io as sio

    mat_path = "data/gt/ZY1E02D.mat"

    mat = sio.loadmat(mat_path)
    Y_noisy = mat['gt']
    print(Y_noisy.shape)

    Y_noisy = torch.from_numpy(Y_noisy).unsqueeze(0).float().permute(0, 3, 1, 2)  # (1, P, H, W)
    # Y_clean = torch.from_numpy(Y_clean).unsqueeze(0).float().permute(0, 3, 1, 2)

    Y_noisy = Y_noisy.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = Y_noisy.device

    C, P, H, W = Y_noisy.shape

    model = Model(p=P, r=24, base_dim=256).cuda()

    Y_denoised, U, V, loss_log, psnr_log, ssim_log, sam_log = train_SSFD(model, Y_noisy, Y_noisy, alpha=0.1,
                                                                         lr=1e-3, max_iter=300, tol=1e-4)




    Y_noisy = _to_numpy_image(Y_noisy)

    Y_denoised = _to_numpy_image(Y_denoised)

    painting_rl(Y_noisy, Y_denoised, [89, 40, 1], name='ZY')
    plot_spectral_curve(Y_noisy, Y_denoised, coord=(150, 150))



