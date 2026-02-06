import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from skimage import feature, img_as_float
from skimage.metrics import normalized_root_mse
import scipy.stats as stats
def calculate_al_mask(I_light, I_de_light):
    # 计算S_residual
    S_residual = torch.abs(I_light - I_de_light)

    # 计算阈值
    max_value = torch.max(S_residual)
    min_value = torch.min(S_residual)
    threshold = (max_value + min_value) * 0.6

    # 二值化处理得到AL mask
    al_mask = (S_residual > threshold).type(torch.uint8)
    
    return al_mask

######可视化的retinex，但是用这个不能恢复L通道,原因是归一化到[0，255]的范围后，归一化的值会丢失
def physical_retinex_decomposition(image, sigma=1.5):
    """使用LAB空间的L通道进行Retinex分解"""
    # 转换到LAB颜色空间并提取L通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    # 对L通道进行高斯模糊得到光照分量
    illumination = cv2.GaussianBlur(l_channel, (0, 0), sigma)
    
    # 计算反射分量（添加epsilon防止除零）
    epsilon = 1e-6
    reflectance = np.log(l_channel + epsilon) - np.log(illumination + epsilon)
    
    # 将光照分量转换为uint8并限制范围
    illumination = np.clip(illumination, 0, 255).astype(np.uint8)
    
    # 对反射分量进行归一化处理
    reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min() + epsilon) * 255
    reflectance = np.clip(reflectance, 0, 255).astype(np.uint8)
    
    return illumination, reflectance



# def process_dataset(input_path, output_path):
#     """处理指定目录下的所有子文件夹"""
#     subfolders = [d for d in os.listdir(input_path) 
#                  if os.path.isdir(os.path.join(input_path, d))]
    
#     if not subfolders:
#         print(f"警告: {input_path} 目录下没有子文件夹")
#         return
        
#     for subfolder in subfolders:
#         input_dir = os.path.join(input_path, subfolder)
        
#         # 创建输出目录结构
#         illum_dir = os.path.join(output_path, subfolder, 'illumination')
#         refl_dir = os.path.join(output_path, subfolder, 'reflectance')
#         os.makedirs(illum_dir, exist_ok=True)
#         os.makedirs(refl_dir, exist_ok=True)
        
#         # 获取图像文件列表
#         image_files = [f for f in os.listdir(input_dir) 
#                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
#         # 使用进度条处理图像
#         for img_file in tqdm(image_files, desc=f"处理文件夹 {subfolder}"):
#             img_path = os.path.join(input_dir, img_file)
#             img = cv2.imread(img_path)
            
#             if img is None:
#                 print(f"无法读取图像: {img_path}")
#                 continue
            
#             # 进行Retinex分解
#             illumination, reflectance = physical_retinex_decomposition(img)
            
#             # 保存结果（单通道图像）
#             base_name = os.path.splitext(img_file)[0]
#             cv2.imwrite(os.path.join(illum_dir, f"{base_name}_illum.png"), illumination)
#             cv2.imwrite(os.path.join(refl_dir, f"{base_name}_refl.png"), reflectance)
#路径下无子文件夹的处理
# def process_dataset(input_path, output_path):
#     """处理输入目录下的所有图像文件"""
#     # 创建输出目录结构
#     illum_dir = os.path.join(output_path, 'illumination')
#     refl_dir = os.path.join(output_path, 'reflectance')
#     os.makedirs(illum_dir, exist_ok=True)
#     os.makedirs(refl_dir, exist_ok=True)
    
#     # 获取图像文件列表
#     image_files = [f for f in os.listdir(input_path) 
#                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
#     if not image_files:
#         print(f"警告: {input_path} 目录下没有图像文件")
#         return
    
#     # 使用进度条处理图像
#     for img_file in tqdm(image_files, desc="处理图像"):
#         img_path = os.path.join(input_path, img_file)
#         img = cv2.imread(img_path)
        
#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue
#         img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#         # 进行Retinex分解
#         illumination, reflectance = physical_retinex_decomposition(img)
        
#         # 保存结果（单通道图像）
#         base_name = os.path.splitext(img_file)[0]
#         cv2.imwrite(os.path.join(illum_dir, f"{base_name}_illum.png"), illumination)
#         cv2.imwrite(os.path.join(refl_dir, f"{base_name}_refl.png"), reflectance)
    


# def pi(image):
#     """
#     计算PI（Perceptual Index）指标
#     """
#     image = img_as_float(image)
#     edges = feature.canny(image, sigma=3)
#     pi_value = np.sum(edges) / (image.shape[0] * image.shape[1])
#     return pi_value

# def niqe(image):
#     """
#     计算NIQE（Naturalness Image Quality Evaluator）指标
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 需要预训练的模型等，这里简化处理
#     ref = np.ones_like(gray) * np.mean(gray)
#     niqe_value = normalized_root_mse(gray, ref)
#     return niqe_value
# def uicm(image):
#     """
#     计算UIQM中的UICM（Underwater Image Colorfulness Measure）
#     """
#     R = image[:, :, 0].astype(np.float64)
#     G = image[:, :, 1].astype(np.float64)
#     B = image[:, :, 2].astype(np.float64)
#     RG = R - G
#     YB = 0.5 * (R + G) - B
#     mu_rg = np.mean(RG)
#     mu_yb = np.mean(YB)
#     sigma_rg = np.std(RG)
#     sigma_yb = np.std(YB)
#     uicm_value = np.sqrt(mu_rg ** 2 + mu_yb ** 2) + np.sqrt(sigma_rg ** 2 + sigma_yb ** 2)
#     return uicm_value
# def uism(image):
#     """
#     计算UIQM中的UISM（Underwater Image Sharpness Measure）
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 拉普拉斯算子
#     laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#     uism_value = np.std(laplacian)
#     return uism_value
# def uiconm(image):
#     """
#     计算UIQM中的UICONM（Underwater Image Contrast Measure）
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
#     hist = hist / (image.shape[0] * image.shape[1])
#     cdf = np.cumsum(hist)
#     uiconm_value = stats.entropy(cdf)
#     return uiconm_value
# def uiqm(image):
#     """
#     计算UIQM（Underwater Image Quality Measure）指标
#     """
#     uicm_value = uicm(image)
#     uism_value = uism(image)
#     uiconm_value = uiconm(image)
#     # 系数参考相关文献
#     uiqm_value = 0.0282 * uicm_value + 0.2953 * uism_value + 0.6765 * uiconm_value
#     return uiqm_value
# if __name__ == "__main__":
#     # 直接指定路径（可根据需要修改）
#     input_path = "/home/zhw/UIALN/EUVP/images"
#     output_path = "/home/zhw/UIALN_copy/EUVP_retinex"
    
#     print(f"开始处理，输入目录: {input_path}")
#     process_dataset(input_path, output_path)
#     print(f"处理完成！结果保存在: {output_path}")


# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

def physical_retinex_decomposition(image, sigma=1.5):
    """使用LAB空间的L通道进行Retinex分解"""
    # 转换到LAB颜色空间并提取L通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    # 对L通道进行高斯模糊得到光照分量
    illumination = cv2.GaussianBlur(l_channel, (0, 0), sigma)
    
    # 计算反射分量（添加epsilon防止除零）
    epsilon = 1e-6
    reflectance = np.log(l_channel + epsilon) - np.log(illumination + epsilon)
    
    # 将光照分量转换为uint8并限制范围
    illumination = np.clip(illumination, 0, 255).astype(np.uint8)
    
    # 对反射分量进行归一化处理
    reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min() + epsilon) * 255
    reflectance = np.clip(reflectance, 0, 255).astype(np.uint8)
    
    return illumination, reflectance

def process_dataset(input_path, output_path):
    """处理包含train/val的数据集结构"""
    for split in ['train', 'val']:
        split_input = os.path.join(input_path, split)
        split_output = os.path.join(output_path, split)
        
        # 跳过不存在的split目录
        if not os.path.exists(split_input):
            print(f"跳过不存在的split目录: {split_input}")
            continue
        
        # 获取子文件夹列表
        subfolders = [d for d in os.listdir(split_input) 
                     if os.path.isdir(os.path.join(split_input, d))]
        
        for subfolder in subfolders:
            input_dir = os.path.join(split_input, subfolder)
            
            # 准备输出目录
            illum_dir = os.path.join(split_output, subfolder, 'illumination')
            refl_dir = os.path.join(split_output, subfolder, 'reflectance')
            os.makedirs(illum_dir, exist_ok=True)
            os.makedirs(refl_dir, exist_ok=True)
            
            # 处理图像
            image_files = [f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_file in tqdm(image_files, 
                               desc=f"处理 {split}/{subfolder}",
                               leave=False):
                img_path = os.path.join(input_dir, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError("图像读取失败")
                        
                    illum, refl= physical_retinex_decomposition(img)
                    
                    base_name = os.path.splitext(img_file)[0]
                    cv2.imwrite(os.path.join(illum_dir, f"{base_name}_illum.png"), illum)
                    cv2.imwrite(os.path.join(refl_dir, f"{base_name}_refl.png"), refl)
                
                except Exception as e:
                    print(f"处理失败: {img_path} - {str(e)}")
                    continue

if __name__ == "__main__":
    # 配置路径
    input_path = "/home/zhw/UIALN/Synthetic_dataset/synthetic_dataset_with_AL"
    output_path = "/home/zhw/UIALN_copy/Al_retinex"
    
    print("开始处理数据集...")
    print(f"输入目录结构:")
    print(f"{input_path}/[train|val]/[子文件夹]/图片文件")
    
    process_dataset(input_path, output_path)
    
    print("\n处理完成！输出结构:")
    print(f"{output_path}/")
    print("├── train/")
    print("│   └── [子文件夹]/")
    print("│       ├── illumination/")
    print("│       └── reflectance/")
    print("└── val/")
    print("    └── ...相同结构...")
    print(f"共处理 {len(os.listdir(os.path.join(output_path, 'train')))} 个训练子文件夹")
    print(f"共处理 {len(os.listdir(os.path.join(output_path, 'val')))} 个验证子文件夹")