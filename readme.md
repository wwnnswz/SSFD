# Self-Supervised Hyperspectral Image Denoising via Deep Spatial-Spectral-Frequency Decoupled Prior

Weizhen Sun, Qiang Zhang, Wenjing Gao,


<hr />

> **Abstract:**  Hyperspectral image (HSI) is inevitably corrupted by various noises during acquisition, which severely degrade subsequent quantitative 
> analysis and applications. Existing supervised HSI denoising methods heavily rely on large-scale clean datasets, usually suffering from overfitting 
> and limited generalization. To address these limitations, This work proposes a novel self-supervised HSI denoising framework driven by a deep 
> spatial-spectral-frequency decoupled (SSFD) prior. First, inspired by the physical mechanisms of the hyperspectral linear mixing model, a dual-branch 
> parallel architecture is constructed for spatial-spectral decoupling. It transforms high-dimensional image reconstruction into a low-rank factorization process, 
> where the deep prior imposes strong physical constraints on noise modeling. Second, to mitigate the instability of endmember estimation, 
> a frequency-domain adaptive filtering (FAF) module is introduced with learnable weights. By exploiting the spectral discrepancies between signals and noise, 
> the FAF module precisely modulates abnormal frequency components in the complex domain. Finally, a mask-based self-supervised learning strategy is adopted. 
> Leveraging the inherent spatial-spectral-frequency redundancy of HSIs, the proposed method effectively prevents the identity mapping of noise, 
> facilitating blind HSI denoising without clean ground-truth data. Extensive experiments demonstrate that SSFD consistently outperforms state-of-the-art approaches.
<hr />

## Network Architecture

<img src = "figs/model.jpg"> 


