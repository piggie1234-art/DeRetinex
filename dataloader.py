# DataLoader
import cv2
import numpy as np
import torch.random
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
import sys
import glob
import warnings
from PIL import Image

def get_transform_0():
    transform = transforms.Compose([
        # RGB转化为LAB
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
        # 只保留L通道
        transforms.Lambda(lambda x: x[:, :, 0]),
        transforms.ToTensor(),
    ])
    return transform

def get_transform_1():
    transform = transforms.Compose([
        # RGB转化为LAB
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
        # 只保留AB通道
        transforms.Lambda(lambda x: x[:, :, 1:]),
        transforms.ToTensor(),
    ])
    return transform

def get_transform_lab(size=None):
    if size is not None:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])
    else:
        transform = transforms.Compose([
            # RGB转化为LAB
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)),
            transforms.ToTensor(),
        ])
    return transform

# 分解数据集，以及I_delight部分的数据集
class retinex_decomposition_data(Dataset):
    def __init__(self, I_no_light_path, I_light_path):
        # self.I_light_imglist = self.get_path(I_light_path)
        # self.I_no_light_imglist = [os.path.join(I_no_light_path, os.path.basename(img_path)) for img_path in
        #                            self.I_light_imglist]
        # self.transform = get_transform_0()
        self.I_light_path = I_light_path
        self.I_light_imglist = self.get_recursive_path(I_light_path)
        self.I_no_light_imglist = self.generate_corresponding_paths(I_no_light_path)
        self.transform = get_transform_0()  #只保留L通道

    # def get_path(self, path):
    #     img_name_list = sorted(os.listdir(path))
    #     img_list = []
    #     for img_name in img_name_list:
    #         img_list.append(os.path.join(path, img_name))
    #     return img_list

    def get_recursive_path(self, path):
        img_list = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_list.append(os.path.join(root, file))
        return sorted(img_list)
    
    def generate_corresponding_paths(self, no_light_root):
        """生成对应的无光图像路径"""
        corresponding_paths = []
        for light_path in self.I_light_imglist:
            rel_path = os.path.relpath(light_path, self.I_light_path)
            
            #split[0]是train/val，split[1]是子文件夹名
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) < 2:
                raise ValueError(f"无效的路径结构: {light_path}")
            
            # 构建无光图像路径：no_light_root/子文件夹名/文件名
            subfolder = path_parts[-2]  # 第2层是子文件夹名
            filename = path_parts[-1]
            no_light_path = os.path.join(no_light_root, subfolder, filename)
            
            corresponding_paths.append(no_light_path)
        return corresponding_paths

    def __len__(self):
        return len(self.I_no_light_imglist)

    def __getitem__(self, index):
        I_no_AL_img_path = self.I_no_light_imglist[index]
        I_AL_img_path = self.I_light_imglist[index]

        I_no_AL_img = cv2.imread(I_no_AL_img_path, cv2.IMREAD_COLOR)
        I_AL_img = cv2.imread(I_AL_img_path, cv2.IMREAD_COLOR)

        # 检查图片是否读取成功
        if I_no_AL_img is None or I_AL_img is None:
            print(index)
            print(I_AL_img_path)
            print(I_AL_img)
            print("Error: 图片读取失败")
            #sys.exit(0)

        I_no_AL_img = cv2.cvtColor(I_no_AL_img, cv2.COLOR_BGR2RGB)
        I_AL_img = cv2.cvtColor(I_AL_img, cv2.COLOR_BGR2RGB)

        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        I_no_AL_tensor = self.transform(I_no_AL_img)
        torch.random.manual_seed(seed)
        I_AL_tensor = self.transform(I_AL_img)

        return I_no_AL_tensor, I_AL_tensor

# AL区域自导向色彩恢复模块数据集
# class AL_data(Dataset):
#     def __init__(self, ABcc_path, gt_path):
#         self.ABcc_imglist = self.get_path(ABcc_path)
#         # gt_name是basename的_前面的部分
#         self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path).split("_")[0]+'.bmp') for img_path in self.ABcc_imglist]
#         self.transform_1 = get_transform_1()
#         self.transform_0 = get_transform_0()

#     def get_path(self, path):
#         img_name_list = sorted(os.listdir(path))
#         img_list = []
#         for img_name in img_name_list:
#             img_list.append(os.path.join(path, img_name))
#         return img_list

#     def __len__(self):
#         return len(self.ABcc_imglist)

#     def __getitem__(self, index):
#         ABcc_img_path = self.ABcc_imglist[index]
#         gt_img_path = self.gt_imglist[index]

#         ABcc_img = cv2.imread(ABcc_img_path, cv2.IMREAD_COLOR)
#         gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

#         # 检查图片是否读取成功
#         if ABcc_img is None or gt_img is None:
#             print(index)
#             print(ABcc_img_path)
#             print(gt_img_path)
#             print("Error: 图片读取失败")
#             #sys.exit(0)

#         ABcc_img = cv2.cvtColor(ABcc_img, cv2.COLOR_BGR2RGB)
#         gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

#         seed = torch.random.seed()

#         torch.random.manual_seed(seed)
#         ABcc_tensor = self.transform_1(ABcc_img)
#         torch.random.manual_seed(seed)
#         gt_tensor = self.transform_1(gt_img)
#         torch.random.manual_seed(seed)
#         L_tensor = self.transform_0(ABcc_img)

#         return ABcc_tensor, gt_tensor, L_tensor
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
class AL_data(Dataset):
    """
    加载图像四元组 (quadruplet)。

    - 图片1: 来自 dir1/<parent>/illumination/<img_name> (transform1: ToTensor)
    - 图片2: 来自 dir2/<parent>/<img_name_mod.bmp>       (transform)
    - 图片3: 来自 dir3/<label_num>.<any_image_ext>     (transform)
    - 图片4: 来自 dir1/<parent>/reflectance/<img_name>  (transform4: ToTensor)

    label_num 从 img1_name 提取。img2, img3 使用 transform。
    """
    def __init__(self, dir1_base, dir2_base, dir3_base): # 保持您提供的构造函数签名
        super().__init__()
        self.dir1_base = dir1_base
        self.dir2_base = dir2_base
        self.dir3_base = dir3_base
        self.transform = get_transform_1() # 用于 img2 和 img3
        self.transform14 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]) # 用于 img1 (illumination)

        self.img3_map = self._scan_dir3_and_build_map()

        # --- 修改: 从 triplets 到 quadruplets ---
        self.image_quadruplets = [] # 存储找到的 (img1_path, img2_path, img3_path, img4_path)
        if self.img3_map is not None:
            self._find_quadruplets() # 重命名查找函数
        else:
            warnings.warn("数据集初始化失败：无法扫描或构建 img3 路径映射。请检查 dir3 路径和权限。")
            return

        # 更新警告和成功信息
        if not self.image_quadruplets and self.img3_map is not None:
            warnings.warn(f"数据集初始化警告：虽然成功扫描了 dir3，但在指定的路径结构下没有找到任何完整的图像四元组。\n"
                          f"请检查：\n"
                          f"  - Dir1: {self.dir1_base}/<parent>/illumination/<img_name> 是否存在文件？\n"
                          f"  - Dir1: {self.dir1_base}/<parent>/reflectance/<img_name> 是否存在对应文件？\n"
                          f"  - Dir2: {self.dir2_base}/<parent>/<img_name_mod.bmp> 是否存在对应文件？\n"
                          f"  - 文件名是否能正确解析出数字前缀？\n"
                          f"  - 解析出的数字是否存在于 dir3 ({len(self.img3_map)} 个标签已映射)？")
        elif self.image_quadruplets:
            print(f"数据集初始化成功。共找到 {len(self.image_quadruplets)} 个图像四元组。")
            print(f"  (基于在 dir3 中成功映射的 {len(self.img3_map)} 个标签图像)")

    def _scan_dir3_and_build_map(self):
        # 这个函数逻辑不变
        print(f"正在扫描第三个目录 (dir3: {self.dir3_base}) 以构建标签到路径的映射...")
        img3_map = {}
        valid_img_count = 0
        skipped_non_img = 0
        skipped_naming = 0
        try:
            if not os.path.isdir(self.dir3_base):
                 print(f"错误：第三个目录 (dir3: {self.dir3_base}) 不存在或不是一个目录。")
                 return None
            for filename in os.listdir(self.dir3_base):
                file_path = os.path.join(self.dir3_base, filename)
                if os.path.isfile(file_path):
                    base_name, ext = os.path.splitext(filename)
                    if ext.lower() in IMG_EXTENSIONS:
                        if base_name.isdigit():
                            if base_name in img3_map:
                                print(f"警告：在 dir3 中发现重复的标签数字 '{base_name}'。将使用新发现的路径覆盖: {file_path}")
                            img3_map[base_name] = file_path
                            valid_img_count += 1
                        else: skipped_naming += 1
                    else: skipped_non_img += 1
            print(f"dir3 扫描完成。共找到 {valid_img_count} 个有效的标签图像文件。")
            if skipped_non_img > 0: print(f"  - 跳过了 {skipped_non_img} 个非图像文件。")
            if skipped_naming > 0: print(f"  - 跳过了 {skipped_naming} 个文件名不是纯数字的图像文件。")
            if not img3_map: print("警告：在 dir3 中没有找到任何有效的标签图像文件。")
            return img3_map
        except Exception as e:
            print(f"错误：扫描第三个目录时发生意外错误: {e}")
            return None

    # --- 修改: 重命名函数并处理 quadruplets ---
    def _find_quadruplets(self):
        """
        扫描 dir1 和 dir2，并使用预构建的 img3_map 查找四元组。
        """
        print("正在扫描 dir1 和 dir2 以查找图像四元组...")
        found_count = 0
        skipped_due_to_missing_img1 = 0
        skipped_due_to_missing_img2 = 0
        skipped_due_to_missing_img3_in_map = 0
        skipped_due_to_missing_img4 = 0 # img4 缺失计数器 (已存在于您提供的代码中)
        skipped_due_to_naming = 0
        skipped_non_image = 0
        processed_img1_candidates = 0

        try:
            parent_folders_dir1 = sorted([d for d in os.listdir(self.dir1_base) if os.path.isdir(os.path.join(self.dir1_base, d))])
            if not parent_folders_dir1:
                 print(f"警告：在 {self.dir1_base} 中未找到父文件夹。")
                 return
        except Exception as e:
            print(f"错误：访问目录 {self.dir1_base} 时出错: {e}")
            return

        for p_folder_name in parent_folders_dir1:
            dir1_illum_path = os.path.join(self.dir1_base, p_folder_name, 'illumination')
            dir1_reflect_path = os.path.join(self.dir1_base, p_folder_name, 'reflectance') # img4 所在目录

            if not os.path.isdir(dir1_illum_path): continue # illumination 必须存在

            try:
                image_files_in_illum = os.listdir(dir1_illum_path)
            except Exception as e: continue

            for img1_filename_ext in image_files_in_illum:
                processed_img1_candidates += 1
                img1_name, img1_ext = os.path.splitext(img1_filename_ext)

                if img1_ext.lower() not in IMG_EXTENSIONS:
                    skipped_non_image += 1; continue

                # --- 查找 img1 (Illumination) ---
                img1_path = os.path.join(dir1_illum_path, img1_filename_ext)
                if not os.path.exists(img1_path):
                    skipped_due_to_missing_img1 += 1; continue

                # --- 解析标签号 ---
                try:
                    label_num_str = img1_name.split('_')[0]
                    if not label_num_str.isdigit(): raise ValueError
                except (IndexError, ValueError):
                    skipped_due_to_naming += 1; continue

                # --- 查找 img3 (Label) ---
                if label_num_str in self.img3_map:
                    img3_path = self.img3_map[label_num_str]
                else:
                    skipped_due_to_missing_img3_in_map += 1; continue

                # --- 查找 img2 (Synthetic) ---
                # 这里的路径构建逻辑是您提供的，它修改了文件名和扩展名
                img2_name_base = os.path.splitext(img1_filename_ext.replace("_illum", ""))[0]
                img2_filename = f"{img2_name_base}.bmp"
                img2_path = os.path.join(self.dir2_base, p_folder_name, img2_filename)
                if not os.path.exists(img2_path):
                    skipped_due_to_missing_img2 += 1; continue

                # --- 查找 img4 (Reflectance) ---
                img4_path = os.path.join(dir1_reflect_path, img1_filename_ext.replace("_illum", "_refl")) # 文件名与 img1 相同
                if not os.path.exists(img4_path):
                    skipped_due_to_missing_img4 += 1; continue

                # --- 所有四个文件都找到 ---
                # --- 修改: 添加到 quadruplets 列表 ---
                self.image_quadruplets.append((img1_path, img2_path, img3_path, img4_path))
                found_count += 1

        # 更新打印的统计信息
        print(f"扫描完成。共处理 {processed_img1_candidates} 个来自 dir1/illumination 的候选文件。")
        print(f"成功找到 {found_count} 个有效的图像四元组。") #<-- 更新消息
        print(f"跳过统计：")
        if skipped_non_image > 0: print(f"  - {skipped_non_image} 个非图像文件 (img1)")
        if skipped_due_to_naming > 0: print(f"  - {skipped_due_to_naming} 个因 img1 文件名无法解析数字标签而被跳过")
        if skipped_due_to_missing_img1 > 0: print(f"  - {skipped_due_to_missing_img1} 个因 img1 文件实际不存在而被跳过")
        if skipped_due_to_missing_img2 > 0: print(f"  - {skipped_due_to_missing_img2} 个因对应的 img2 文件不存在而被跳过")
        if skipped_due_to_missing_img3_in_map > 0: print(f"  - {skipped_due_to_missing_img3_in_map} 个因对应的数字标签在 dir3 映射中未找到而被跳过")
        if skipped_due_to_missing_img4 > 0: print(f"  - {skipped_due_to_missing_img4} 个因对应的 img4 (reflectance) 文件不存在而被跳过") #<-- 添加 img4 统计


    # --- 修改: 返回 quadruplets 列表的长度 ---
    def __len__(self):
        return len(self.image_quadruplets)

    # --- 修改: 处理并返回四个图像 ---
    def __getitem__(self, index):
        # --- 修改: 从 quadruplets 获取路径 ---
        img1_path, img2_path, img3_path, img4_path = self.image_quadruplets[index]
        try:
            # 使用 cv2 读取图像
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
            img3 = cv2.imread(img3_path, cv2.IMREAD_COLOR)
            img4 = cv2.imread(img4_path, cv2.IMREAD_GRAYSCALE) # 读取第四张图


            # 检查图像是否成功加载 (cv2 在失败时返回 None)
            if img1 is None: raise IOError(f"无法读取图像文件 (img1): {img1_path}")
            if img2 is None: raise IOError(f"无法读取图像文件 (img2): {img2_path}")
            if img3 is None: raise IOError(f"无法读取图像文件 (img3): {img3_path}")
            if img4 is None: raise IOError(f"无法读取图像文件 (img4): {img4_path}")

            # BGR -> RGB (cv2 默认 BGR, PyTorch 通常期望 RGB)
            #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            #img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB) # 转换第四张图

            # 应用转换
            I_light = self.transform14(img1) # img1 使用 transform1
            ABcc = self.transform(img2)     # img2 使用 transform
            GT = self.transform(img3)       # img3 使用 transform
            R_light = self.transform14(img4) # img4 使用 transform4
            

            # --- 修改: 返回四个张量 ---
            return I_light, ABcc, GT, R_light

        except FileNotFoundError as e: # 这个通常不应触发，因为路径已在 init 验证
            print(f"严重错误：加载图像失败 (索引 {index})，文件在getitem时未找到: {e}")
            print(f"  涉及路径: img1={img1_path}, img2={img2_path}, img3={img3_path}, img4={img4_path}")
            raise RuntimeError(f"在索引 {index} 处加载图像失败: {e}") from e
        except IOError as e: # 处理 cv2.imread 失败
            print(f"错误：读取图像文件失败 (索引 {index}): {e}")
            raise RuntimeError(f"在索引 {index} 处读取图像时出错") from e
        except Exception as e:
            print(f"错误：处理图像时发生意外错误 (索引 {index})")
            print(f"  涉及路径: img1={img1_path}, img2={img2_path}, img3={img3_path}, img4={img4_path}")
            print(f"  错误详情: {e}")
            raise RuntimeError(f"在索引 {index} 处处理图像时出错") from e
  

class IlluminationDataset(Dataset):
    def __init__(self, 
                 noal_root="/home/zhw/UIALN_copy/NoAl_retinex",
                 al_root="/home/zhw/UIALN_copy/Al_retinex/train",
                 transform=None):
        """
        参数说明:
        - 每个样本返回四张图：[NoAl_illum, Al_illum, NoAl_refl, Al_refl]
        - 文件名要求：XXX_illum.ext 和 XXX_refl.ext 的基名相同
        """
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 构建四图路径列表
        self.quads = self._validate_and_pair_paths(noal_root, al_root)

    def _parse_basename(self, filename):
        """解析文件名，返回基名和类型（illum/refl）"""
        base = os.path.splitext(filename)[0]
        if '_illum' in base:
            return base.replace('_illum', ''), 'illum'
        elif '_refl' in base:
            return base.replace('_refl', ''), 'refl'
        return None, None

    def _validate_and_pair_paths(self, noal_root, al_root):
        # 验证子文件夹结构
        subs = self._validate_subfolders(noal_root, al_root)
        
        quads = []
        for sub in subs:
            # 定义四个目录路径
            dirs = {
                'noal_illum': os.path.join(noal_root, sub, 'illumination'),
                'noal_refl': os.path.join(noal_root, sub, 'reflectance'),
                'al_illum': os.path.join(al_root, sub, 'illumination'),
                'al_refl': os.path.join(al_root, sub, 'reflectance')
            }
            
            # 验证所有目录存在
            for d in dirs.values():
                if not os.path.exists(d):
                    raise FileNotFoundError(f"缺失目录: {d}")
            
            # 收集所有文件的基名映射
            base_map = {}
            for key, path in dirs.items():
                for f in os.listdir(path):
                    base, type_ = self._parse_basename(f)
                    if not base:
                        continue
                        
                    if base not in base_map:
                        base_map[base] = {'noal_illum': None, 'al_illum': None,
                                         'noal_refl': None, 'al_refl': None}
                        
                    # 更新对应路径
                    full_path = os.path.join(path, f)
                    if key.startswith('noal'):
                        if type_ == 'illum':
                            base_map[base]['noal_illum'] = full_path
                        else:
                            base_map[base]['noal_refl'] = full_path
                    else:
                        if type_ == 'illum':
                            base_map[base]['al_illum'] = full_path
                        else:
                            base_map[base]['al_refl'] = full_path
            
            # 验证四图完整性
            for base, paths in base_map.items():
                if all(paths.values()):
                    quads.append((
                        paths['noal_illum'],
                        paths['al_illum'],
                        paths['noal_refl'],
                        paths['al_refl']
                    ))
                # else:
                #     missing = [k for k, v in paths.items() if not v]
                #     print(f"跳过 {base}，缺失: {', '.join(missing)}")
        
        if not quads:
            raise RuntimeError("未找到任何有效四图组合")
        return quads

    def _validate_subfolders(self, noal_root, al_root):
        # 验证10个同名子文件夹
        noal_subs = sorted([d for d in os.listdir(noal_root) 
                          if os.path.isdir(os.path.join(noal_root, d))])
        al_subs = sorted([d for d in os.listdir(al_root) 
                        if os.path.isdir(os.path.join(al_root, d))])
        
        if len(noal_subs) != 10 or noal_subs != al_subs:
            raise ValueError("子文件夹结构不匹配")
        return noal_subs

    def __len__(self):
        return len(self.quads)

    def __getitem__(self, idx):
        noal_illum_path, al_illum_path, noal_refl_path, al_refl_path = self.quads[idx]
        
        # 加载图像函数
        def load_image(path):
            img = Image.open(path).convert('L')
            if self.transform:
                img = self.transform(img)
            return img
        
        return {
            'noal_illum': load_image(noal_illum_path),
            'al_illum': load_image(al_illum_path),
            'noal_refl': load_image(noal_refl_path),
            'al_refl': load_image(al_refl_path),
            'base_name': os.path.basename(noal_illum_path).split('_illum')[0]
        }

# class Detail_Enhancement_data(Dataset):
#     def __init__(self, ABcc_path, gt_path, size=256):
#         self.size = size
#         self.ABcc_imglist = self.get_path(ABcc_path)
#         # gt_name是basename的_前面的部分
#         # self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path).split("_")[0]+'.bmp') for img_path in self.ABcc_imglist]
#         self.gt_imglist = [os.path.join(gt_path, os.path.basename(img_path)) for img_path in self.ABcc_imglist]
#         self.transform_1 = get_transform_1(self.size)
#         self.transform_0 = get_transform_0(self.size)
#         self.transform_lab = get_transform_lab(self.size)

#     def get_path(self, path):
#         img_name_list = sorted(os.listdir(path))
#         img_list = []
#         for img_name in img_name_list:
#             img_list.append(os.path.join(path, img_name))
#         return img_list

#     def __len__(self):
#         return len(self.ABcc_imglist)

#     def __getitem__(self, index):
#         ABcc_img_path = self.ABcc_imglist[index]
#         gt_img_path = self.gt_imglist[index]

#         ABcc_img = cv2.imread(ABcc_img_path, cv2.IMREAD_COLOR)
#         gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)

#         # 检查图片是否读取成功
#         if ABcc_img is None or gt_img is None:
#             print(index)
#             print(ABcc_img_path)
#             print(gt_img_path)
#             print("Error: 图片读取失败")
#             exit(0)

#         ABcc_img = cv2.cvtColor(ABcc_img, cv2.COLOR_BGR2RGB)
#         gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

#         seed = torch.random.seed()

#         torch.random.manual_seed(seed)
#         ABcc_tensor = self.transform_1(ABcc_img)
#         torch.random.manual_seed(seed)
#         gt_L_tensor = self.transform_0(gt_img)
#         torch.random.manual_seed(seed)
#         L_tensor = self.transform_0(ABcc_img)
#         torch.random.manual_seed(seed)
#         gt = self.transform_lab(gt_img)

#         return ABcc_tensor, L_tensor, gt_L_tensor, gt
class Detail_Enhancement_data(Dataset):
    """
    加载图像四元组 (quadruplet)。

    - 图片1: 来自 dir1/<parent>/illumination/<img_name> (transform1: ToTensor)
    - 图片2: 来自 dir2/<parent>/<img_name_mod.bmp>       (transform)
    - 图片3: 来自 dir3/<label_num>.<any_image_ext>     (transform)
    - 图片4: 来自 dir1/<parent>/reflectance/<img_name>  (transform4: ToTensor)

    label_num 从 img1_name 提取。img2, img3 使用 transform。
    """
    def __init__(self, dir1_base, dir2_base, dir3_base): # 保持您提供的构造函数签名
        super().__init__()
        self.dir1_base = dir1_base
        self.dir2_base = dir2_base
        self.dir3_base = dir3_base
        self.transform0 = get_transform_0() # 用于 img2 和 img3
        self.transform = get_transform_1() # 用于 img2 和 img3
        self.transformLab = get_transform_lab() # 用于 img2 和 img3
        self.transform14 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]) # 用于 img1 (illumination)

        self.img3_map = self._scan_dir3_and_build_map()

        # --- 修改: 从 triplets 到 quadruplets ---
        self.image_quadruplets = [] # 存储找到的 (img1_path, img2_path, img3_path, img4_path)
        if self.img3_map is not None:
            self._find_quadruplets() # 重命名查找函数
        else:
            warnings.warn("数据集初始化失败：无法扫描或构建 img3 路径映射。请检查 dir3 路径和权限。")
            return

        # 更新警告和成功信息
        if not self.image_quadruplets and self.img3_map is not None:
            warnings.warn(f"数据集初始化警告：虽然成功扫描了 dir3，但在指定的路径结构下没有找到任何完整的图像四元组。\n"
                          f"请检查：\n"
                          f"  - Dir1: {self.dir1_base}/<parent>/illumination/<img_name> 是否存在文件？\n"
                          f"  - Dir1: {self.dir1_base}/<parent>/reflectance/<img_name> 是否存在对应文件？\n"
                          f"  - Dir2: {self.dir2_base}/<parent>/<img_name_mod.bmp> 是否存在对应文件？\n"
                          f"  - 文件名是否能正确解析出数字前缀？\n"
                          f"  - 解析出的数字是否存在于 dir3 ({len(self.img3_map)} 个标签已映射)？")
        elif self.image_quadruplets:
            print(f"数据集初始化成功。共找到 {len(self.image_quadruplets)} 个图像四元组。")
            print(f"  (基于在 dir3 中成功映射的 {len(self.img3_map)} 个标签图像)")

    def _scan_dir3_and_build_map(self):
        # 这个函数逻辑不变
        print(f"正在扫描第三个目录 (dir3: {self.dir3_base}) 以构建标签到路径的映射...")
        img3_map = {}
        valid_img_count = 0
        skipped_non_img = 0
        skipped_naming = 0
        try:
            if not os.path.isdir(self.dir3_base):
                 print(f"错误：第三个目录 (dir3: {self.dir3_base}) 不存在或不是一个目录。")
                 return None
            for filename in os.listdir(self.dir3_base):
                file_path = os.path.join(self.dir3_base, filename)
                if os.path.isfile(file_path):
                    print("找到label图片:",os.path.join(self.dir3_base, filename))
                    base_name, ext = os.path.splitext(filename)
                    if ext.lower() in IMG_EXTENSIONS:
                        if base_name.isdigit():
                            if base_name in img3_map:
                                print(f"警告：在 dir3 中发现重复的标签数字 '{base_name}'。将使用新发现的路径覆盖: {file_path}")
                            img3_map[base_name] = file_path
                            valid_img_count += 1
                        else: skipped_naming += 1
                    else: skipped_non_img += 1
            print(f"dir3 扫描完成。共找到 {valid_img_count} 个有效的标签图像文件。")
            if skipped_non_img > 0: print(f"  - 跳过了 {skipped_non_img} 个非图像文件。")
            if skipped_naming > 0: print(f"  - 跳过了 {skipped_naming} 个文件名不是纯数字的图像文件。")
            if not img3_map: print("警告：在 dir3 中没有找到任何有效的标签图像文件。")
            return img3_map
        except Exception as e:
            print(f"错误：扫描第三个目录时发生意外错误: {e}")
            return None

    # --- 修改: 重命名函数并处理 quadruplets ---
    def _find_quadruplets(self):
        """
        扫描 dir1 和 dir2，并使用预构建的 img3_map 查找四元组。
        """
        print("正在扫描 dir1 和 dir2 以查找图像四元组...")
        found_count = 0
        skipped_due_to_missing_img1 = 0
        skipped_due_to_missing_img2 = 0
        skipped_due_to_missing_img3_in_map = 0
        skipped_due_to_missing_img4 = 0 # img4 缺失计数器 (已存在于您提供的代码中)
        skipped_due_to_naming = 0
        skipped_non_image = 0
        processed_img1_candidates = 0

        try:
            parent_folders_dir1 = sorted([d for d in os.listdir(self.dir1_base) if os.path.isdir(os.path.join(self.dir1_base, d))])
            if not parent_folders_dir1:
                 print(f"警告：在 {self.dir1_base} 中未找到父文件夹。")
                 return
        except Exception as e:
            print(f"错误：访问目录 {self.dir1_base} 时出错: {e}")
            return

        for p_folder_name in parent_folders_dir1:
            dir1_illum_path = os.path.join(self.dir1_base, p_folder_name, 'illumination')
            dir1_reflect_path = os.path.join(self.dir1_base, p_folder_name, 'reflectance') # img4 所在目录

            if not os.path.isdir(dir1_illum_path): continue # illumination 必须存在

            try:
                image_files_in_illum = os.listdir(dir1_illum_path)
            except Exception as e: continue

            for img1_filename_ext in image_files_in_illum:
                processed_img1_candidates += 1
                img1_name, img1_ext = os.path.splitext(img1_filename_ext)

                if img1_ext.lower() not in IMG_EXTENSIONS:
                    skipped_non_image += 1; continue

                # --- 查找 img1 (Illumination) ---
                img1_path = os.path.join(dir1_illum_path, img1_filename_ext)
                if not os.path.exists(img1_path):
                    skipped_due_to_missing_img1 += 1; continue

                # --- 解析标签号 ---
                try:
                    label_num_str = img1_name.split('_')[0]
                    if not label_num_str.isdigit(): raise ValueError
                except (IndexError, ValueError):
                    skipped_due_to_naming += 1; continue

                # --- 查找 img3 (Label) ---
                if label_num_str in self.img3_map:
                    img3_path = self.img3_map[label_num_str]
                else:
                    skipped_due_to_missing_img3_in_map += 1; continue

                # --- 查找 img2 (Synthetic) ---
                # 这里的路径构建逻辑是您提供的，它修改了文件名和扩展名
                img2_name_base = os.path.splitext(img1_filename_ext.replace("_illum", ""))[0]
                img2_filename = f"{img2_name_base}.bmp"
                img2_path = os.path.join(self.dir2_base, p_folder_name, img2_filename)
                if not os.path.exists(img2_path):
                    skipped_due_to_missing_img2 += 1; continue

                # --- 查找 img4 (Reflectance) ---
                img4_path = os.path.join(dir1_reflect_path, img1_filename_ext.replace("_illum", "_refl")) # 文件名与 img1 相同
                if not os.path.exists(img4_path):
                    skipped_due_to_missing_img4 += 1; continue

                # --- 所有四个文件都找到 ---
                # --- 修改: 添加到 quadruplets 列表 ---
                self.image_quadruplets.append((img1_path, img2_path, img3_path, img4_path))
                found_count += 1

        # 更新打印的统计信息
        print(f"扫描完成。共处理 {processed_img1_candidates} 个来自 dir1/illumination 的候选文件。")
        print(f"成功找到 {found_count} 个有效的图像四元组。") #<-- 更新消息
        print(f"跳过统计：")
        if skipped_non_image > 0: print(f"  - {skipped_non_image} 个非图像文件 (img1)")
        if skipped_due_to_naming > 0: print(f"  - {skipped_due_to_naming} 个因 img1 文件名无法解析数字标签而被跳过")
        if skipped_due_to_missing_img1 > 0: print(f"  - {skipped_due_to_missing_img1} 个因 img1 文件实际不存在而被跳过")
        if skipped_due_to_missing_img2 > 0: print(f"  - {skipped_due_to_missing_img2} 个因对应的 img2 文件不存在而被跳过")
        if skipped_due_to_missing_img3_in_map > 0: print(f"  - {skipped_due_to_missing_img3_in_map} 个因对应的数字标签在 dir3 映射中未找到而被跳过")
        if skipped_due_to_missing_img4 > 0: print(f"  - {skipped_due_to_missing_img4} 个因对应的 img4 (reflectance) 文件不存在而被跳过") #<-- 添加 img4 统计


    # --- 修改: 返回 quadruplets 列表的长度 ---
    def __len__(self):
        return len(self.image_quadruplets)

    # --- 修改: 处理并返回四个图像 ---
    def __getitem__(self, index):
        # --- 修改: 从 quadruplets 获取路径 ---
        img1_path, img2_path, img3_path, img4_path = self.image_quadruplets[index]
        try:
            # 使用 cv2 读取图像
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
            img3 = cv2.imread(img3_path, cv2.IMREAD_COLOR)
            img4 = cv2.imread(img4_path, cv2.IMREAD_GRAYSCALE) # 读取第四张图


            # 检查图像是否成功加载 (cv2 在失败时返回 None)
            if img1 is None: raise IOError(f"无法读取图像文件 (img1): {img1_path}")
            if img2 is None: raise IOError(f"无法读取图像文件 (img2): {img2_path}")
            if img3 is None: raise IOError(f"无法读取图像文件 (img3): {img3_path}")
            if img4 is None: raise IOError(f"无法读取图像文件 (img4): {img4_path}")

            # BGR -> RGB (cv2 默认 BGR, PyTorch 通常期望 RGB)
            #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            #img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB) # 转换第四张图

            # 应用转换
            I_light = self.transform14(img1) 
            ABcc = self.transform(img2)     
            GT = self.transformLab(img3)       
            R_light = self.transform14(img4) 
            GT_l = self.transform0(img3) 

            # --- 修改: 返回四个张量 ---
            return I_light, ABcc, GT, R_light,GT_l

        except FileNotFoundError as e: # 这个通常不应触发，因为路径已在 init 验证
            print(f"严重错误：加载图像失败 (索引 {index})，文件在getitem时未找到: {e}")
            print(f"  涉及路径: img1={img1_path}, img2={img2_path}, img3={img3_path}, img4={img4_path}")
            raise RuntimeError(f"在索引 {index} 处加载图像失败: {e}") from e
        except IOError as e: # 处理 cv2.imread 失败
            print(f"错误：读取图像文件失败 (索引 {index}): {e}")
            raise RuntimeError(f"在索引 {index} 处读取图像时出错") from e
        except Exception as e:
            print(f"错误：处理图像时发生意外错误 (索引 {index})")
            print(f"  涉及路径: img1={img1_path}, img2={img2_path}, img3={img3_path}, img4={img4_path}")
            print(f"  错误详情: {e}")
            raise RuntimeError(f"在索引 {index} 处处理图像时出错") from e



import random

class UnpairedUnderwaterDataset(Dataset):
    """
    用于加载非配对水下图像的 PyTorch Dataset 类。

    参数:
        hr_folder (str): 存放高分辨率 (HR) 图像的文件夹路径。
        lr_folder (str): 存放真实低分辨率 (LR) 图像的文件夹路径。
        transform (callable, optional): 应用于图像的 torchvision 变换。
    """
    def __init__(self, hr_folder, lr_folder, transform=None):
        super(UnpairedUnderwaterDataset, self).__init__()
        
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.transform = transform
        
        # 1. 加载 HR 和 LR 文件夹中所有图像文件的路径
        # 同时过滤掉非图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.hr_image_files = sorted([
            os.path.join(hr_folder, f) for f in os.listdir(hr_folder) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        self.lr_image_files = sorted([
            os.path.join(lr_folder, f) for f in os.listdir(lr_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])

        if not self.hr_image_files or not self.lr_image_files:
            raise ValueError("HR or LR folder is empty or contains no valid images.")

        self.hr_len = len(self.hr_image_files)
        self.lr_len = len(self.lr_image_files)

    def __len__(self):
        """
        返回数据集中样本的总数。
        我们取两个文件夹中较大者的大小，以确保所有数据都被利用。
        """
        return max(self.hr_len, self.lr_len)

    def __getitem__(self, index):
        """
        获取一个数据样本，包含一张 HR 图像和一张非配对的 LR 图像。
        """
        # 2. 获取 HR 图像
        # 使用取模运算确保索引不会越界
        hr_path = self.hr_image_files[index % self.hr_len]
        
        # 3. 随机获取一张 LR 图像，实现非配对
        # 为了可复现性和效率，也可以使用取模，
        # 真正的随机性由 DataLoader 的 shuffle=True 提供。
        lr_index = random.randint(0, self.lr_len - 1)
        # 或者使用取模，更稳定: lr_index = index % self.lr_len
        lr_path = self.lr_image_files[lr_index]
        
        # 使用 PIL 加载图像
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        # 4. 应用图像变换
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        # 以字典形式返回，方便在训练循环中通过键名访问
        return {"hr": hr_image, "lr": lr_image}

if __name__ == '__main__':
    hr_dir = "/home/zhw/hr1x3" # 替换为你的路径
    lr_dir = "/home/zhw/LR" # 替换为你的路径
    #dir3 = "/home/zhw/UIALN/Synthetic_dataset/labels/raw" # 替换为你的路径
    
    try:
        SR_dataset = UnpairedUnderwaterDataset(hr_folder=hr_dir,lr_folder=lr_dir)
        dataloader = DataLoader(dataset=SR_dataset,batch_size=4,shuffle=False)
        print("✅ 数据集实例化成功")
        print(f"HR图像数量: {SR_dataset.hr_len}")
        print(f"LR图像数量: {SR_dataset.lr_len}")
        print(f"数据集总长度: {len(SR_dataset)}")
    except Exception as e:
        print(f"❌ 数据集实例化失败: {e}")
        exit()
    
    # 2. 测试获取单张HR和LR图像
    print("\n🔍 测试单样本获取:")
    try:
        sample = SR_dataset[0]  # 获取第一个样本
        img1_sample, img2_sample= SR_dataset[0]
        print(f"  样本 {0}:")
        print(f"    图片1 (Tensor) - 形状: {img1_sample.shape}, 类型: {img1_sample.dtype}")
        print(f"    图片2 (Tensor) - 形状: {img2_sample.shape}, 类型: {img2_sample.dtype}")
     
        print(f"样本类型: {type(sample)}")
        print(f"包含的键: {list(sample.keys())}")
        print(f"HR图像形状: {sample['hr'].shape}")  # 应为 torch.Size([3, 高, 宽])
        print(f"LR图像形状: {sample['lr'].shape}")
        print(f"数据类型: {sample['hr'].dtype}")
        print(f"数值范围: HR[{sample['hr'].min():.3f}, {sample['hr'].max():.3f}] "
              f"LR[{sample['lr'].min():.3f}, {sample['lr'].max():.3f}]")
        
        # 检查是否为有效图像张量
        assert sample['hr'].shape[0] == 3, "HR图像通道数应为3"
        assert sample['lr'].shape[0] == 3, "LR图像通道数应为3"
        assert 0 <= sample['hr'].min() <= sample['hr'].max() <= 1.0, "HR图像值超出[0,1]范围"
        
        print("✅ 单样本测试通过")
        
    except Exception as e:
        print(f"❌ 单样本测试失败: {e}")
        exit()
    
    # 3. 测试非配对特性
    print("\n🔄 测试非配对特性:")
    try:
        sample1 = SR_dataset[0]  # 第一个样本
        sample2 = SR_dataset[1]  # 第二个样本
        sample3 = SR_dataset[SR_dataset.hr_len + 1]  # 测试索引超出HR数量时的行为
        
        # 简单的非配对验证：不同样本的LR图像路径应该不同（通过形状或内容判断）
        # 更准确的验证需要比较图像内容或路径
        print("✅ 非配对数据获取正常")
        
    except Exception as e:
        print(f"❌ 非配对测试异常: {e}")

    # print("开始创建 AL_data 数据集实例...")
    # paths_ok = True
    # for p in [dir1, dir2, dir3]:
    #     if not os.path.isdir(p):
    #         print(f"错误：基础目录 '{p}' 不存在！")
    #         paths_ok = False

    # if paths_ok:
    #     try:
    #         # 注意：构造函数现在不接受 transform 参数了
    #         al_dataset = AL_data(dir1_base=dir1, dir2_base=dir2, dir3_base=dir3)

    #         print(f"\n数据集的总大小: {len(al_dataset)}")

    #         if len(al_dataset) > 0:
    #             print("\n获取第一个样本数据 (索引 0):")
    #             sample_index = 0
    #             try:
    #                 # --- 修改: 接收四个返回值 ---
    #                 img1_sample, img2_sample, img3_sample, img4_sample = al_dataset[sample_index]
    #                 print(f"  样本 {sample_index}:")
    #                 print(f"    图片1 (Tensor) - 形状: {img1_sample.shape}, 类型: {img1_sample.dtype}")
    #                 print(f"    图片2 (Tensor) - 形状: {img2_sample.shape}, 类型: {img2_sample.dtype}")
    #                 print(f"    图片3 (Tensor) - 形状: {img3_sample.shape}, 类型: {img3_sample.dtype}")
    #                 print(f"    图片4 (Tensor) - 形状: {img4_sample.shape}, 类型: {img4_sample.dtype}") #<-- 显示 img4 信息

    #             except Exception as e:
    #                 print(f"  获取样本 {sample_index} 时出错: {e}")
    #                 import traceback
    #                 traceback.print_exc()


    #             # 测试 DataLoader
    #             from torch.utils.data import DataLoader
    #             print("\n测试 DataLoader:")
    #             try:
    #                 data_loader = DataLoader(al_dataset, batch_size=4, shuffle=True, num_workers=0)
    #                 # --- 修改: 接收四个批次 ---
    #                 first_batch = next(iter(data_loader))
    #                 img1_batch, img2_batch, img3_batch, img4_batch = first_batch
    #                 print(f"  成功获取第一个批次数据:")
    #                 print(f"    图片1 批次形状: {img1_batch.shape}")
    #                 print(f"    图片2 批次形状: {img2_batch.shape}")
    #                 print(f"    图片3 批次形状: {img3_batch.shape}")
    #                 print(f"    图片4 批次形状: {img4_batch.shape}") #<-- 显示 img4 批次信息
    #             except StopIteration:
    #                  print("  DataLoader 为空或已迭代完毕。")
    #             except Exception as e:
    #                  print(f"  使用 DataLoader 获取批次数据时出错: {e}")
    #                  import traceback
    #                  traceback.print_exc()

    #         else:
    #              # ... (之前的空数据集提示信息) ...
    #              print("\n数据集为空或未找到任何有效的四元组。")
    #              if al_dataset.img3_map is None: print("  主要问题：无法扫描或读取第三个目录 (dir3)。")
    #              elif len(al_dataset.img3_map) == 0: print("  主要问题：第三个目录 (dir3) 中没有找到有效的标签图像文件。")
    #              else:
    #                  print(f"  已成功映射 {len(al_dataset.img3_map)} 个 dir3 标签图像，但未能匹配成完整四元组。")
    #                  print(f"  请检查 dir1/illumination, dir1/reflectance, dir2 的文件是否存在及命名规则。")
    #                  print(f"  查看上面 _find_quadruplets 方法的 '跳过统计' 以获取线索。")

    #     except Exception as e:
    #         print(f"\n创建或使用 AL_data 数据集时发生顶层错误: {e}")
    #         import traceback
    #         traceback.print_exc()
    # else:
    #     print("\n由于一个或多个基础目录不存在，无法创建数据集。")