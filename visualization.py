import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def normalize(data):
    """归一化到0-255（按波段处理）"""
    if data.ndim == 3:
        data_norm = np.zeros_like(data, dtype=np.uint8)
        for b in range(data.shape[-1]):
            band = data[..., b]
            min_val, max_val = np.min(band), np.max(band)
            if max_val > min_val:  # 避免除零
                band_norm = (band - min_val) / (max_val - min_val) * 255
            else:
                band_norm = np.zeros_like(band, dtype=np.uint8)
            data_norm[..., b] = band_norm.astype(np.uint8)
    else:
        min_val, max_val = np.min(data), np.max(data)
        data_norm = ((data - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)
    return data_norm


def hsi_to_rgb(hsi_data, rgb_bands=[79, 36, 1]):
    """将HSI的3个波段合成为RGB图像（124波段专用）"""
    # 确保波段索引在有效范围内（0-123）
    rgb_bands = [b for b in rgb_bands if 0 <= b < hsi_data.shape[-1]]

    r = hsi_data[..., rgb_bands[0]]
    g = hsi_data[..., rgb_bands[1]]
    b = hsi_data[..., rgb_bands[2]]

    r_norm = normalize(r)
    g_norm = normalize(g)
    b_norm = normalize(b)

    return np.stack([r_norm, g_norm, b_norm], axis=-1)

def painting(input_hsi, gt_hsi, hirdiff_hsi,std,rgb_bands=[79, 36, 1]):

    # 设置全局参数
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.edgecolor'] = 'white'  # 隐藏轴边缘线

    # 合成RGB图像（使用推荐的124波段组合）
    input_rgb = hsi_to_rgb(input_hsi, rgb_bands=rgb_bands)
    gt_rgb = hsi_to_rgb(gt_hsi, rgb_bands=rgb_bands)
    hirdiff_rgb = hsi_to_rgb(hirdiff_hsi, rgb_bands=rgb_bands)

    # 定义局部放大区域（ROI）：根据图像尺寸调整，示例为30x30像素
    H, W = input_rgb.shape[:2]
    x1, y1 = W - 80, H - 80  # 右下角区域起点（距右、下边缘80像素）
    x2, y2 = W - 30, H - 30  # 终点（区域大小50x50）

    # 裁剪ROI
    input_roi = input_rgb[y1:y2, x1:x2, :]
    gt_roi = gt_rgb[y1:y2, x1:x2, :]
    hirdiff_roi = hirdiff_rgb[y1:y2, x1:x2, :]

    # 创建布局：主图 + 右下角放大图（使用GridSpec实现不规则布局）
    fig = plt.figure(figsize=(10, 3.5),constrained_layout=True)  # 宽10，高3.5（适合3列布局）
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1, 1, 1],
                  height_ratios=[5, 2],  # 主图占5/7，放大图占2/7
                  wspace=0.05, hspace=0.05)

    # 绘制主图（第一行）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_rgb)
    ax1.set_title(f"Noisy Input (complex)")
    ax1.axis('off')
    # 标记ROI方框
    ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_rgb)
    ax2.set_title("GT")
    ax2.axis('off')
    ax2.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(hirdiff_rgb)
    ax3.set_title("Our")
    ax3.axis('off')
    ax3.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))

    # 绘制右下角放大图（第二行，与主图对应）
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(input_roi)
    # ax4.set_title("Zoom In")
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(gt_roi)
    # ax5.set_title("Zoom In")
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(hirdiff_roi)
    # ax6.set_title("Zoom In")
    ax6.axis('off')

    # 保存图像（论文常用PDF格式，支持无损缩放）
    # plt.savefig(f"visualization/bulb_0822-0909.mat_[0,{std}].pdf",
    #             dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(f"visualization/bulb_0822-0909.mat_[0,{std}].png",
    #             dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()

def painting_rl(input_hsi, hirdiff_hsi,rgb_bands=[154, 95, 42] ,name='gf',flag=False, path='visualization'):
    # 设置全局参数
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.edgecolor'] = 'white'  # 隐藏轴边缘线



    input_rgb = hsi_to_rgb(input_hsi, rgb_bands=rgb_bands)
    hirdiff_rgb = hsi_to_rgb(hirdiff_hsi, rgb_bands=rgb_bands)



    # 定义局部放大区域（ROI）：根据图像尺寸调整，示例为30x30像素
    H, W = input_rgb.shape[:2]
    x1, y1 = W - 130, H - 130  # 右下角区域起点（距右、下边缘80像素）
    x2, y2 = W - 80, H - 80  # 终点（区域大小50x50）

    # 裁剪ROI
    input_roi = input_rgb[y1:y2, x1:x2, :]
    hirdiff_roi = hirdiff_rgb[y1:y2, x1:x2, :]

    # 创建布局：主图 + 右下角放大图（使用GridSpec实现不规则布局）
    fig = plt.figure(figsize=(8, 3.5), constrained_layout=True)  # 宽10，高3.5（适合3列布局）
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1, 1],
                  height_ratios=[5, 2],  # 主图占5/7，放大图占2/7
                  wspace=0.05, hspace=0.05)

    # 绘制主图（第一行）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_rgb)
    ax1.set_title(f"{name}({rgb_bands[2]+1},{rgb_bands[1]+1},{rgb_bands[0]+1})")
    ax1.axis('off')
    # 标记ROI方框
    ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))



    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hirdiff_rgb)
    ax2.set_title(f"Our({rgb_bands[2]+1},{rgb_bands[1]+1},{rgb_bands[0]+1})")
    ax2.axis('off')
    ax2.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))

    # 绘制右下角放大图（第二行，与主图对应）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(input_roi)
    # ax4.set_title("Zoom In")
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(hirdiff_roi)
    # ax5.set_title("Zoom In")
    ax4.axis('off')


    # 保存图像（论文常用PDF格式，支持无损缩放）
    if flag:
        plt.savefig(f"{path}.pdf",
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f"{path}.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches


def painting_single_band(input_hsi, hirdiff_hsi, band_index=95, cmap='gray', name='gf', path='visualization'):
    """
    展示高光谱图像的单个波段，并包含局部放大区域

    参数:
        input_hsi: 输入高光谱图像 (shape: [H, W, B])
        hirdiff_hsi: 对比高光谱图像 (shape: [H, W, B])
        band_index: 要展示的波段索引（从0开始）
        cmap: 配色方案（如'gray'为灰度，'viridis'为伪彩色）
        name: 输入图像的名称标签
        path: 保存路径（当前仅用于展示）
    """
    # 设置全局参数
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.edgecolor'] = 'white'  # 隐藏轴边缘线

    # 提取单个波段（确保波段索引有效）
    input_band = input_hsi[:, :, band_index]  # 形状: [H, W]
    hirdiff_band = hirdiff_hsi[:, :, band_index]

    # 定义局部放大区域（ROI）：右下角区域，大小50x50像素
    H, W = input_band.shape  # 单个波段为2D：(H, W)
    x1, y1 = W - 80, H - 80  # 起点（距右、下边缘80像素）
    x2, y2 = W - 30, H - 30  # 终点（区域大小50x50）

    # 裁剪ROI（单个波段的局部区域）
    input_roi = input_band[y1:y2, x1:x2]
    hirdiff_roi = hirdiff_band[y1:y2, x1:x2]

    # 创建布局：主图 + 下方放大图（保持原GridSpec风格）
    fig = plt.figure(figsize=(8, 3.5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1, 1],
                  height_ratios=[5, 2],  # 主图占比更大
                  wspace=0.05, hspace=0.05)

    # 绘制输入图像的主波段图
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(input_band, cmap=cmap)
    ax1.set_title(f"{name} (Band {band_index + 1})")  # 波段索引+1（符合1-based习惯）
    ax1.axis('off')
    # 标记ROI方框（红色边框）
    ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))

    # 绘制对比图像的主波段图
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(hirdiff_band, cmap=cmap)
    ax2.set_title(f"Our (Band {band_index + 1})")
    ax2.axis('off')
    ax2.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor='red', facecolor='none'))

    # 绘制输入图像的ROI放大图
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(input_roi, cmap=cmap)
    ax3.axis('off')

    # 绘制对比图像的ROI放大图
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(hirdiff_roi, cmap=cmap)
    ax4.axis('off')

    # 可选：添加颜色条（若需要量化参考）
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 右侧颜色条位置
    # fig.colorbar(im1, cax=cbar_ax)

    # 展示或保存
    plt.show()
    plt.close()




def plot_spectral_curve(original_hsi, denoised_hsi, coord=(150, 150)):
    """
    绘制指定像素点的光谱曲线对比
    """
    r, c = coord
    bands = np.arange(original_hsi.shape[2])
    orig_curve = original_hsi[r, c, :]
    denoised_curve = denoised_hsi[r, c, :]

    plt.figure(figsize=(8, 5))
    plt.plot(bands, orig_curve, '--', label="Original (Noisy)")
    plt.plot(bands, denoised_curve, '-', label="Denoised")
    plt.xlabel("Band index")
    plt.ylabel("Reflectance / Intensity")
    plt.title(f"Spectral Curve at {coord}")
    plt.legend()
    plt.grid(True)
    # 保存图像（论文常用PDF格式，支持无损缩放）
    # plt.savefig(f"visualization/spectral_curve.pdf",
    #             dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(f"visualization/spectral_curve.png",
    #             dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt


def plot_spectral_curve1(noisy_hsi, gt_hsi, denoised_hsi, coord=(150, 150), save_path=None):
    """
    绘制指定像素点的光谱曲线对比 (H, W, B)
    """
    assert gt_hsi.ndim == 3 and gt_hsi.shape == noisy_hsi.shape == denoised_hsi.shape, \
        "Input HSIs must all have shape (H, W, B)"
    H, W, B = gt_hsi.shape
    r, c = coord
    assert 0 <= r < H and 0 <= c < W, f"coord {coord} 超出范围 {(H, W)}"


    bands = np.arange(B)
    gt_curve = gt_hsi[r, c, :]
    denoised_curve = denoised_hsi[r, c, :]
    noisy_curve = noisy_hsi[r, c, :]

    plt.figure(figsize=(7, 4.5))
    plt.plot(bands, noisy_curve, 'r:', label="Noisy")
    plt.plot(bands, gt_curve, 'k--', label="Ground Truth")
    plt.plot(bands, denoised_curve, 'b-', linewidth=1.5, label="Denoised")

    plt.xlabel("Spectral Band Index", fontsize=11)
    plt.ylabel("Reflectance / Intensity", fontsize=11)
    plt.title(f"Spectral Curve at pixel {coord}", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()
