import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from my_indexes import *
from visualization import painting


# from utils import sta

#对高光谱图像进行标准化处理，将像素值映射到 [0, 1] 范围，用于统一数据尺度。
#'all'：全局标准化，使用整个图像的最大 / 最小值对所有像素归一化。
#'pb'：按波段（per-band）标准化，每个光谱波段单独使用自身的最大 / 最小值归一化。
def sta(img, mode):
    img = np.float32(img)
    if mode == 'all':
        ma = np.max(img)
        mi = np.min(img)
        #   return (img - mi)/(ma - mi)
        img = (img - mi) / (ma - mi)
        return img
    elif mode == 'pb':
        ma = np.max(img, axis=(0, 1))
        mi = np.min(img, axis=(0, 1))
        img = (img - mi) / (ma - mi)
        return img

    else:
        print('Undefined Mode!')
        return img


# ------------------------- 脉冲噪声 -------------------------
def add_sp(image, prob):
    h, w = image.shape
    output = image.copy()
    num = int(h * w * prob)
    for _ in range(num):
        x, y = np.random.randint(0, h), np.random.randint(0, w)
        output[x, y] = 0 if np.random.rand() < 0.5 else 1
    return output

def add_impulse_noise_fixed(hsi, a):
    """为所有波段添加比例为 a 的脉冲噪声"""
    H, W, B = hsi.shape
    noisy = np.zeros_like(hsi)
    for i in range(B):
        noisy[:, :, i] = add_sp(hsi[:, :, i], a)
    return noisy

def add_impulse_noise_random(hsi, a):
    """为各波段添加随机比例 [0,a] 的脉冲噪声"""
    H, W, B = hsi.shape
    noisy = np.zeros_like(hsi)
    for i in range(B):
        prob = np.random.uniform(0, a)
        noisy[:, :, i] = add_sp(hsi[:, :, i], prob)
    return noisy

# ------------------------- 条带噪声 -------------------------
def add_stripe_noise(hsi, max_ratio=0.45):
    """为随机选取的 0%~45% 波段添加条带噪声"""
    H, W, B = hsi.shape
    noisy = hsi.copy()
    num_bands = int(B * np.random.uniform(0, max_ratio))
    selected = np.random.choice(B, num_bands, replace=False)
    for b in selected:
        num_stripes = np.random.randint(20, 40)
        for _ in range(num_stripes):
            x = np.random.randint(0, W)
            noisy[:, x, b] = np.random.rand(H)  # 条带
    return noisy

# ------------------------- 死线噪声 -------------------------
def add_deadline_noise(hsi, min_ratio=0.3, max_ratio=0.6):
    """为随机选取的 30%~60% 波段添加死线噪声"""
    H, W, B = hsi.shape
    noisy = hsi.copy()
    num_bands = int(B * np.random.uniform(min_ratio, max_ratio))
    selected = np.random.choice(B, num_bands, replace=False)
    for b in selected:
        num_dead = np.random.randint(20, 50)
        for _ in range(num_dead):
            x = np.random.randint(0, W)
            noisy[:, x, b] = 0  # 死线
    return noisy






#为单波段图像添加椒盐噪声（salt-and-pepper noise）
def add_sp(image,prob):
    h = image.shape[0]
    w = image.shape[1]
    output = image.copy()
    sp = h*w   # 计算图像像素点个数
    NP = int(sp*prob)   # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            output[randx, randy] = 0
        else:
            output[randx, randy] = 1
    return output


#为高光谱图像的所有波段添加椒盐噪声
def add_sp_noise(data_path, std_e):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['gt'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    noi_hsi = np.zeros([Hei,Wid,Band])
    print('add sparse noise  (%s)' % std_e)
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_sp(cln_hsi[:, :, ind].copy(),std_e)
    return cln_hsi, noi_hsi


#为单波段图像添加高斯噪声
def add_gaussian(image, sigma):
    # add gaussian noise
    # image in [0,1], sigma in [0,1]
    output = image.copy()
    output = output + np.random.normal(0, sigma,image.shape)
    # output = output + np.random.randn(image.shape[0], image.shape[1])*sigma
    return output


#为高光谱图像的所有波段添加高斯噪声
def add_Gaussian_noise(data_path, std_list):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['DataCube'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    #归一化
    cln_hsi = sta(cln_hsi, 'pb')
    noi_hsi = np.zeros([Hei,Wid,Band])
    for ind in range(Band):
        noi_hsi[:, :, ind] = add_gaussian(cln_hsi[:, :, ind].copy(), std_list[ind])
    #cln_hsi = cln_hsi[0:100, 0:100, 0:90]
    #noi_hsi = noi_hsi[0:100, 0:100, 0:90]
    return cln_hsi, noi_hsi


#为高光谱图像添加混合噪声（高斯噪声 + 椒盐噪声）
def add_Mixture_noise(data_path, std_g_list, std_s_list):
    data = scipy.io.loadmat(data_path)
    cln_hsi = data['DataCube'].astype(np.float32)
    Hei, Wid, Band = cln_hsi.shape
    cln_hsi = sta(cln_hsi, mode='pb')
    noi_hsi = np.zeros([Hei,Wid,Band])

    for ind in range(Band):
        noi_hsi[:, :, ind] = add_sp(cln_hsi[:, :, ind].copy(), std_s_list[ind])
        noi_hsi[:, :, ind] = add_gaussian(noi_hsi[:, :, ind].copy(), std_g_list[ind])
    return cln_hsi, noi_hsi




#根据指定的噪声类型，为高光谱图像添加对应的噪声，是噪声添加的统一入口
def GetNoise(datapath,noise_case,std):
    """
    'complex'：混合噪声（高斯 + 椒盐），高斯噪声标准差范围 [0, std]，椒盐噪声比例范围 [0, 0.1]。
    'n.i.i.d-g'：非独立同分布高斯噪声（每个波段标准差随机，范围 [0, std]）。
    'i.i.d-g'（默认）：独立同分布高斯噪声（所有波段标准差均为 std）
    """
    if noise_case == 'complex':
        print('complex')

        # # c
        # std_g_list = std * np.ones(191)
        # [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
        # noisy = add_impulse_noise_fixed(noi_hsi, 0.2)
        mat = sio.loadmat(datapath)
        cln_hsi, noisy = mat['gt'], mat['noisy']
        #d
        # std_g_list = np.random.uniform(low=0.0, high=std, size=191)
        # [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
        # noisy = add_impulse_noise_random(noi_hsi, 0.2)
        #
        #e
        # noisy = add_stripe_noise(noisy, max_ratio=0.45)
        #
        #f
        noisy = add_deadline_noise(noisy, min_ratio=0.3, max_ratio=0.6)


        return cln_hsi, noisy
    elif noise_case == 'n.i.i.d-g':
        print('n.i.i.d-g')
        std_g_list = np.random.uniform(low=0.0, high=std, size=191)
        [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
        return cln_hsi, noi_hsi
    else:
        print('i.i.d-g')
        std_g_list = std*np.ones(191)
        [cln_hsi, noi_hsi] = add_Gaussian_noise(datapath, std_g_list)
        return cln_hsi, noi_hsi

if __name__ == "__main__":
    datapath = "data/noisy/WDC_e.mat"
    mat = sio.loadmat(datapath)
    print(mat.keys())
    # std_g_list = np.random.uniform(low=0.0, high=0.05, size=31)
    # std_g_list = 0.05 * np.ones(31)
    # std_s_list = np.random.uniform(low=0.0, high=0.2, size=300)
    #[cln_hsi, noi_hsi] = add_Mixture_noise(datapath, std_g_list, std_s_list)
    [cln_hsi, noi_hsi] = GetNoise(datapath,'complex' ,0.4)
    # 保存为.mat文件
    # 计算中心裁剪的起始和结束索引
    # original_size = 512
    # target_size = 256
    # start = (original_size - target_size) // 2  # 128
    # end = start + target_size  # 384
    #
    # # 中心裁剪（保留第三维不变）
    # gt_cropped = cln_hsi[start:end, start:end, :]  # 形状 (256, 256, 31)
    # noisy_cropped = noi_hsi[start:end, start:end, :]  # 形状 (256, 256, 31)
    #
    # # 验证裁剪后形状
    # print("裁剪后干净图像形状：", gt_cropped.shape)  # 输出 (256, 256, 31)
    # print("裁剪后带噪图像形状：", noisy_cropped.shape)  # 输出 (256, 256, 31)
    save_path = "data/noisy/WDC_f_1.mat"
    sio.savemat(
        save_path,
        {
            'gt': cln_hsi,  # 真实图像
            'noisy': noi_hsi,  # 带噪输入
        }
    )

    print(f"去噪结果已保存至：{save_path}")

    painting(noi_hsi, cln_hsi, cln_hsi, 0.4)
