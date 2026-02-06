import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=0.1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.lambda_g = 0.1  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, x, y):
        # 对x, y使用Sobel滤波器求其梯度
        # x, y的shape为[batch_size, 1, height, width]
        # x_grad, y_grad的shape为[batch_size, 2, height, width]
        x_grad_1 = F.conv2d(x, torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], device=self.device, dtype=torch.float), padding=1)
        x_grad_2 = F.conv2d(x, torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], device=self.device, dtype=torch.float), padding=1)
        y_grad_2 = F.conv2d(y, torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], device=self.device, dtype=torch.float), padding=1)
        y_grad_1 = F.conv2d(y, torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], device=self.device, dtype=torch.float), padding=1)
        x_gard = torch.cat((x_grad_1, x_grad_2), dim=1)
        y_gard = torch.cat((y_grad_1, y_grad_2), dim=1)
        loss = x_gard * torch.exp(-self.lambda_g * y_gard)
        # 求loss的L1范数
        loss = torch.sum(torch.abs(loss))
        return loss * self.TVLoss_weight



class SmoothnessLoss(nn.Module):
    """
    计算图像的总变分损失 (Total Variation Loss)，以促进图像平滑。
    惩罚相邻像素之间的梯度绝对值之和。
    """
    def __init__(self, weight=0.1, norm_type='l1'):
        """
        初始化平滑度损失。

        Args:
            weight (float): 此损失项的权重因子。
            norm_type (str): 计算梯度差异的范数类型。
                             'l1': 使用 L1 范数（绝对值），标准的 TV Loss。
                             'l2': 使用 L2 范数（平方和），对大梯度惩罚更重。
                             默认为 'l1'。
        """
        super(SmoothnessLoss, self).__init__()
        if norm_type not in ['l1', 'l2']:
            raise ValueError(f"不支持的 norm_type: {norm_type}. 请选择 'l1' 或 'l2'.")
        self.weight = weight
        self.norm_type = norm_type
        # print(f"SmoothnessLoss 初始化: weight={self.weight}, norm_type='{self.norm_type}'") # 调试信息

    def forward(self, pred_img):
        # --- 计算水平方向梯度 ---
        # (B, C, H, W-1)
        grad_x = pred_img[:, :, :, 1:] - pred_img[:, :, :, :-1]

        # --- 计算垂直方向梯度 ---
        # (B, C, H-1, W)
        grad_y = pred_img[:, :, 1:, :] - pred_img[:, :, :-1, :]

        # --- 计算损失 ---
        if self.norm_type == 'l1':
            # L1 范数：对梯度的绝对值求平均
            loss_x = torch.mean(torch.abs(grad_x))
            loss_y = torch.mean(torch.abs(grad_y))
            loss = loss_x + loss_y
        elif self.norm_type == 'l2':
            # L2 范数：对梯度的平方求平均
            loss_x = torch.mean(torch.pow(grad_x, 2))
            loss_y = torch.mean(torch.pow(grad_y, 2))
            loss = loss_x + loss_y
        else:
            # 理论上在 __init__ 检查后不会到达这里
            raise ValueError(f"内部错误: 无效的 norm_type '{self.norm_type}'")

        # 应用权重
        weighted_loss = self.weight * loss

        return weighted_loss

def gaussian(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1, sigma=1.5):
    """Generates a 2D Gaussian kernel window."""
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, data_range=1.0, size_average=True, C1=None, C2=None):
    """Calculates SSIM index for one scale."""
    # Constants for stabilization
    if C1 is None:
        C1 = (0.01 * data_range) ** 2
    if C2 is None:
        C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # Formula
    # numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    # denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    # ssim_map = numerator / denominator

    # Simpler formula, equivalent for SSIM (alpha=beta=gamma=1)
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3]) # Average across spatial dimensions and channel for each batch element

# --- SSIM Loss Class ---

class SSIMLoss(torch.nn.Module):
    """
    SSIM loss module.

    Args:
        window_size (int): Size of the Gaussian kernel window. Default: 11.
        sigma (float): Standard deviation of the Gaussian kernel. Default: 1.5.
        data_range (float or int): The dynamic range of the input images (usually 1.0 or 255). Default: 1.0.
        size_average (bool): If True, return the mean SSIM for the batch. If False, return SSIM per image in the batch. Default: True.
        channel (int): Number of input channels (should be 1 for grayscale). Default: 1.
    """
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel, sigma=sigma)
        # C1 and C2 calculated dynamically in _ssim based on data_range

    def forward(self, img1, img2):
        """
        Args:
            img1 (torch.Tensor): Predicted images batch (N, 1, H, W), range [0, data_range].
            img2 (torch.Tensor): Target images batch (N, 1, H, W), range [0, data_range].
        Returns:
            torch.Tensor: SSIM loss value (1 - SSIM).
        """
        # Ensure window is on the same device as imgs
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)

        # Make sure channel dimension matches
        (_, channel1, _, _) = img1.size()
        (_, channel2, _, _) = img2.size()
        if channel1 != self.channel or channel2 != self.channel:
             raise ValueError(f"Input images must have {self.channel} channels, but got {channel1} and {channel2}")

        ssim_val = _ssim(img1, img2, window=self.window, window_size=self.window_size, channel=self.channel,
                         data_range=self.data_range, size_average=self.size_average)

        # Loss is 1 - SSIM
        loss = 1.0 - ssim_val
        return loss
    
class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self, lambda_g=10, R_channels=1, reduction='mean'):
        """
        Args:
            lambda_g (float): Weight factor for the reflectance gradient term.
            R_channels (int): Number of channels in the reflectance map R.
                               If R is 3-channel, we might take its average for gradient calculation.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean': output will be summed and
                             divided by the number of elements. 'sum': output will be summed.
        """
        super(IlluminationSmoothnessLoss, self).__init__()
        self.lambda_g = lambda_g
        self.R_channels = R_channels
        self.reduction = reduction

        # Define Sobel filters for gradient calculation (horizontal and vertical)
        # These are fixed, so define them once
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)

        self.register_buffer('sobel_x', kernel_x)
        self.register_buffer('sobel_y', kernel_y)

    def _compute_gradient(self, img_tensor):
        """ Computes gradients using Sobel filters.
            Input img_tensor is expected to be (N, 1, H, W)
        """
        if img_tensor.device != self.sobel_x.device:
            self.sobel_x = self.sobel_x.to(img_tensor.device)
            self.sobel_y = self.sobel_y.to(img_tensor.device)

        grad_x = F.conv2d(img_tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(img_tensor, self.sobel_y, padding=1)
        # You can choose to return |grad_x| + |grad_y| or sqrt(grad_x^2 + grad_y^2)
        # For L1 loss, sum of absolute gradients is common for TV-like losses.
        return torch.abs(grad_x) + torch.abs(grad_y) # (N, 1, H, W)

    def forward(self, I, R):
        """
        Args:
            I (torch.Tensor): Illumination map, shape (N, 1, H, W).
            R (torch.Tensor): Reflectance map, shape (N, C_r, H, W) where C_r is self.R_channels.
        Returns:
            torch.Tensor: The calculated illumination smoothness loss.
        """
        if I.ndim != 4 or I.shape[1] != 1:
            raise ValueError(f"Illumination map I must be (N, 1, H, W), got {I.shape}")
        if R.ndim != 4 or R.shape[1] != self.R_channels:
            raise ValueError(f"Reflectance map R must be (N, {self.R_channels}, H, W), got {R.shape}")

        # 1. Compute gradient of Illumination map I_i: ∇I_i
        grad_I = self._compute_gradient(I) # (N, 1, H, W)

        # 2. Compute gradient of Reflectance map R_i: ∇R_i
        # If R is multi-channel (e.g., 3 for RGB), we need a single-channel representation for its gradient.
        # Common approaches:
        #   a) Average R channels: R_gray = torch.mean(R, dim=1, keepdim=True)
        #   b) Max R channels: R_max, _ = torch.max(R, dim=1, keepdim=True)
        #   c) Luminance (if R is in a color space where luminance is a channel, or convert to YCbCr and use Y)
        # Let's use the average for simplicity here.
        if self.R_channels > 1:
            R_for_grad = torch.mean(R, dim=1, keepdim=True) # (N, 1, H, W)
        else:
            R_for_grad = R # (N, 1, H, W)

        grad_R = self._compute_gradient(R_for_grad) # (N, 1, H, W)

        # 3. Compute the weighting term: exp(-λ_g * |∇R_i|)
        # Note: The formula shows ∇R_i, if ∇R_i can be negative, we should use its magnitude.
        # Our _compute_gradient already returns sum of absolute x and y grads, so it's non-negative.
        weighting_term = torch.exp(-self.lambda_g * grad_R) # (N, 1, H, W)

        # 4. Compute the term inside the L1 norm: ∇I_i ◦ exp(-λ_g * |∇R_i|)
        # The ◦ symbol means element-wise multiplication.
        weighted_grad_I = grad_I * weighting_term # (N, 1, H, W)

        # 5. Compute the L1 norm: || ... ||
        # This is typically the sum of absolute values over all elements (pixels, batch).
        # The formula L_is = Σ_{i=low,normal} || ... || suggests summing over different conditions (low/normal).
        # This forward pass calculates the ||...|| part for ONE condition (i).
        # The sum Σ_{i=low,normal} would be handled outside, by calling this loss twice.

        loss = torch.abs(weighted_grad_I) # Element-wise absolute value

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 256, 256).cuda()
    y = torch.randn(1, 1, 256, 256).cuda()
    tv_loss = TVLoss()
    loss = tv_loss(x, y)
    print(loss)