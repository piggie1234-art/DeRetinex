import torch
from torch import nn
from torch.nn import ModuleList
from torchsummary import summary
from SGFMT import TransformerModel
# import sys
# print(sys.path)
from net.IntmdSequential import *
from CMSFFT import ChannelTransformer
import scipy
import numpy as np


# class Retinex_Decomposition_net(nn.Module):
#     def __init__(self, in_channels=1, out_channels=2):
#         super(Retinex_Decomposition_net, self).__init__()
#         #self.relu = nn.LeakyReLU(inplace=True)
#         self.relu = nn.ELU(alpha=1.0,inplace=True)
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         # relu激活
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))
#         x = self.relu(self.conv5(x))
#         return x

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True): # nf: num_features (输入和输出通道数), gc: growth_channel (内部增长通道数)
        super(ResidualDenseBlock, self).__init__()
        # RDB 包含多个卷积层，每个卷积层的输入是原始输入 x 和之前所有卷积层输出的拼接
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        # 最后一个卷积层将所有拼接的特征映射回 nf 通道
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.ELU(inplace=True) # 使用 LeakyReLU 是 RDB 中常见的做法

        # 权重初始化 (可选)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1_ = self.conv1(x)
        x1 = self.lrelu(x1_)

        x2_ = self.conv2(torch.cat((x, x1), 1))
        x2 = self.lrelu(x2_)

        x3_ = self.conv3(torch.cat((x, x1, x2), 1))
        x3 = self.lrelu(x3_)

        x4_ = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = self.lrelu(x4_)

        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # 局部残差学习: 输出是 x5 + x (输入特征直接加到最后一个卷积的输出上)
        # 有时会用一个缩放因子，如 x5 * 0.2 + x，这里我们先用直接相加
        return x5 + x

# --- DecomNet_LAB_L_RDB 网络定义 ---
class DecomNet_DecompositionNet_L(nn.Module):
    def __init__(self, in_channels=1, out_channels_R=1, out_channels_I=1,
                 num_features=64, rdb_growth_channel=32, num_rdbs=3,
                 initial_conv_kernel_size=9, recon_conv_kernel_size=3):
        """
        Args:
            in_channels (int): 输入图像的通道数 (对于L通道是1)
            out_channels_R (int): 输出反射率图的通道数 (对于R_L是1)
            out_channels_I (int): 输出光照图的通道数 (对于I_L是1)
            num_features (int): 网络中主要的特征通道数 (nf)
            rdb_growth_channel (int): RDB内部的增长通道数 (gc)
            num_rdbs (int): 使用的RDB块的数量
            initial_conv_kernel_size (int): 第一个卷积层的kernel size 
            recon_conv_kernel_size (int): 最后一个重建卷积层的kernel size
        """
        super().__init__()

        # 1. 初始特征提取层 (Shallow Feature Extraction)
        #    padding = (kernel_size - 1) // 2 保持尺寸不变
        padding_initial = (initial_conv_kernel_size - 1) // 2
        self.conv_first = nn.Conv2d(in_channels, num_features, initial_conv_kernel_size,
                                    padding=padding_initial, padding_mode='replicate')
        self.lrelu_first = nn.ELU(inplace=True) # 匹配RDB的激活函数

        # 2. 多个串联的 RDB 块 (Deep Feature Extraction using RDBs)
        rdb_blocks_list = [ResidualDenseBlock(nf=num_features, gc=rdb_growth_channel) for _ in range(num_rdbs)]
        self.rdbs = nn.Sequential(*rdb_blocks_list)

        # 3. 特征整合层 (Feature Aggregation/Fusion after RDBs)
        #    在所有RDB之后，通常会有一个卷积层（例如1x1或3x3）来进一步整合RDB提取的特征。
        self.conv_gff = nn.Conv2d(num_features * num_rdbs, num_features, kernel_size=1, padding=0) 

        # 如果RDBs是简单串联，最后一个RDB的输出通道数是num_features
        self.conv_before_recon = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.lrelu_before_recon = nn.ELU(inplace=True)


        # 4. 重建层 (Reconstruction Layer)
        #    输出两个单通道图: R_L 和 I_L，所以总输出通道是 out_channels_R + out_channels_I
        padding_recon = (recon_conv_kernel_size - 1) // 2
        self.conv_recon = nn.Conv2d(num_features, out_channels_R + out_channels_I, recon_conv_kernel_size,
                                    padding=padding_recon, padding_mode='replicate')

        self._initialize_weights()

    def _initialize_weights(self):
        # 初始化 conv_first, conv_before_recon, conv_recon
        # RDB 内部已经有初始化了
        for m_name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'rdbs' not in m_name: # 避免重复初始化RDB内部的
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_L_channel):
        # 1. 初始特征提取
        shallow_features = self.conv_first(input_L_channel)
        x = self.lrelu_first(shallow_features) # 激活

        # 2. 通过 RDB 块
        #    x_rdb_input = x # 保存RDBs的输入，用于可能的全局残差连接
        x = self.rdbs(x)    # RDBs简单串联

        # (可选的全局残差连接，模仿RDN的 Long Skip Connection (LSC))
        # x = x + shallow_features # 如果想加这个，需要确保通道数和尺寸匹配

        # 3. RDBs之后进一步处理/整合
        x = self.conv_before_recon(x)
        x = self.lrelu_before_recon(x)

        # 4. 重建输出
        outs = self.conv_recon(x)

        # 分解为 R_L 和 I_L
        R_L = torch.sigmoid(outs[:, 0:1, :, :]) # 第一个通道作为 R_L
        I_L = torch.sigmoid(outs[:, 1:2, :, :]) # 第二个通道作为 I_L

        return I_L,R_L


class Illumination_Correction(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(Illumination_Correction, self).__init__()
        self.down_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.down_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.down_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.up_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.up_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.up_3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1)
        # 相当于两次反卷积
        self.up_4_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)  ################# 存疑
        self.up_4_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1)
        self.up_5 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1)
        self.conv1 = nn.Conv2d(32 * 3, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x1 = self.up_1(x)
        #print(x1.shape)
        x2 = self.up_2(x1)
        #print(x2.shape)
        x = self.up_3(x2)
        x1 = self.up_4_2(self.up_4_1(x1))
        x2 = self.up_5(x2)
        #print(x.shape, x1.shape, x2.shape)
        x = torch.cat((x, x1, x2), dim=1)
        x = self.conv1(x)
        return x
    
class PositionEncoding(nn.Module):
    """
    可学习的绝对位置编码 (类似于 ViT)
    需要根据预期的 H_small, W_small 和 transformer_dim 初始化
    """
    def __init__(self, seq_len, dim):
        super().__init__()
        # seq_len = H_small * W_small
        # dim = transformer_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        # 可以使用 xavier / kaiming / trunc_normal 等初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        # x shape: (B, seq_len, dim)
        if x.shape[1] != self.pos_embed.shape[1]:
            # 处理输入序列长度与位置编码不匹配的问题 (例如，通过插值)
            # 这里为了简单起见，先假设它们匹配或报错
            raise ValueError(f"Input sequence length {x.shape[1]} doesn't match position encoding length {self.pos_embed.shape[1]}")
        return x + self.pos_embed

class Illumination_Correction(nn.Module):
    def __init__(self, in_channels=2, out_channels=1,
                 transformer_dim=128,   # Transformer 输入/输出维度 (瓶颈处通道数)
                 transformer_depth=1,   # Transformer Block 数量
                 transformer_heads=8,   # 注意力头数
                 transformer_mlp_dim=512,# Transformer FFN 隐藏维度
                 dropout=0.1,           # 通用 Dropout 率
                 attn_dropout=0.1,      # Attention Dropout 率
                 pe_dropout=0.1,        # 位置编码后的 Dropout 率
                 expected_input_height=256, # 假设一个预期的输入图像高度
                 expected_input_width=256): 
        """
        初始化带有 Transformer 瓶颈的光照校正网络。
        Args:
            in_channels (int): 输入图像通道数。
            out_channels (int): 输出图像通道数。
            transformer_dim (int): 瓶颈处 Transformer 的特征维度，需与编码器最后一层输出通道数匹配。
            transformer_depth (int): Transformer 内部 Block 的层数。
            transformer_heads (int): Transformer 自注意力头数。
            transformer_mlp_dim (int): Transformer 前馈网络隐藏层维度。
            dropout (float): Transformer 中的通用 Dropout 概率。
            attn_dropout (float): Transformer 中 Attention Map 的 Dropout 概率。
        """
        super(Illumination_Correction, self).__init__()

        # self.rgb_to_feature=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])
        # self.feature_to_rgb=ModuleList([to_rgb(32),to_rgb(64),to_rgb(128),to_rgb(256)])

        # --- 编码器 (Encoder) ---
        # 注意：添加 padding=1 使得 stride=2 时，空间尺寸减半
        self.down_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ELU(inplace=True)
        self.down_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ELU(inplace=True)
        encoder_out_channels = 128
        # 编码器最后一层，输出通道数等于 transformer_dim
        self.down_3 = nn.Conv2d(64, transformer_dim, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ELU(inplace=True)
        self.pre_trans_bn = nn.BatchNorm2d(encoder_out_channels) # 或 LayerNorm? 片段只写了bn
        self.pre_trans_relu = nn.ELU(inplace=True) # 使用 ELU 保持一致性
        # 这个 Conv_x 将编码器输出通道映射到 Transformer 维度
        self.pre_trans_conv = nn.Conv2d(encoder_out_channels, transformer_dim, kernel_size=1)

        h_small = expected_input_height // 8
        w_small = expected_input_width // 8
        expected_seq_len = h_small * w_small
        self.position_encoding = PositionEncoding(expected_seq_len, transformer_dim)
        self.pe_dropout = nn.Dropout(pe_dropout)

        # --- Transformer 瓶颈 (Bottleneck) ---
        # Transformer 的输入维度 (dim) 必须与 down_3 的输出通道数 (transformer_dim) 一致
        self.transformer = TransformerModel(
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
            dropout_rate=dropout,
            attn_dropout_rate=attn_dropout
        )
        #transformer的输出维度也必须与 transformer_dim 一致
        self.post_trans_ln = nn.LayerNorm(transformer_dim)
        self.post_trans_conv = nn.Conv2d(transformer_dim, encoder_out_channels, kernel_size=1)
        # --- 解码器 (Decoder) ---
        # 注意：ConvTranspose2d 使用 kernel=3, stride=2, padding=1, output_padding=1 来大致加倍空间尺寸
        # up_1 的输入通道数必须与 transformer 输出维度 (transformer_dim) 一致
        self.up_1 = nn.ConvTranspose2d(transformer_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ELU(inplace=True)
        self.up_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.ELU(inplace=True)
        self.up_3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6 = nn.ELU(inplace=True)

        # --- 旁路/多尺度上采样路径 (遵循原始逻辑) ---
        # 这些路径操作解码器中间层的输出
        # up_4_1 输入通道数应为 up_1 输出通道数 (64)
        self.up_4_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # 第一个反卷积，输入来自 up_1 的输出
        self.relu7 = nn.ELU(inplace=True)
        # up_4_2 输入通道数应为 up_4_1 输出通道数 (64)
        self.up_4_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 第二个反卷积
        self.relu8 = nn.ELU(inplace=True)

        # up_5 输入通道数应为 up_2 输出通道数 (32)
        self.up_5 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # 输入来自 up_2 的输出
        self.relu9 = nn.ELU(inplace=True)

        # --- 最终卷积层 ---
        # 输入通道数 = up_3 输出 (32) + up_4_2 输出 (32) + up_5 输出 (32) = 96
        self.conv1 = nn.Conv2d(32 * 3, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """ 网络的前向传播 """
        x_down1 = self.relu1(self.down_1(x))     # (B, 32, H/2, W/2)
        x_down2 = self.relu2(self.down_2(x_down1)) # (B, 64, H/4, W/4)
        encoded_features = self.relu3(self.down_3(x_down2)) # (B, encoder_out_channels, H/8, W/8)

        # --- Transformer 瓶颈 (带前后处理和残差连接) ---
        B, C_enc, H_small, W_small = encoded_features.shape
        N = H_small * W_small

        # 1. 保存残差
        residual = encoded_features

        # 2. 预处理
        t = self.pre_trans_bn(residual)
        t = self.pre_trans_relu(t)
        t = self.pre_trans_conv(t) # (B, transformer_dim, H_small, W_small)

        # 3. Reshape 为序列
        # (B, C_trans, H_small, W_small) -> (B, N, C_trans)
        t_flat = t.flatten(2).transpose(1, 2)

        # 4. 添加位置编码和 Dropout
        t_pe = self.position_encoding(t_flat) # 需要确保 PositionEncoding 能处理可变长度或插值
        t_pe = self.pe_dropout(t_pe)

        # 5. 应用 Transformer
        t_transformed = self.transformer(t_pe) # (B, N, C_trans)

        # 6. 后处理 LN
        t_post_ln = self.post_trans_ln(t_transformed)

        # 7. Reshape 回特征图
        # (B, N, C_trans) -> (B, C_trans, N) -> (B, C_trans, H_small, W_small)
        t_reshaped = t_post_ln.transpose(1, 2).reshape(B, -1, H_small, W_small)

        # 8. 后处理 Conv
        t_post_conv = self.post_trans_conv(t_reshaped) # (B, encoder_out_channels, H_small, W_small)

        # 9. 添加残差连接
        x_bottleneck = t_post_conv + residual # (B, encoder_out_channels, H_small, W_small)

        # --- 解码器 ---
        # 将瓶颈层的输出送入解码器
        d1 = self.relu4(self.up_1(x_bottleneck)) # (B, 64, H/4, W/4)
        d2 = self.relu5(self.up_2(d1))           # (B, 32, H/2, W/2)
        d3 = self.relu6(self.up_3(d2))           # (B, 32, H, W)

        # --- 旁路/多尺度上采样 ---
        d1_side = self.relu8(self.up_4_2(self.relu7(self.up_4_1(d1)))) # (B, 32, H, W)
        d2_side = self.relu9(self.up_5(d2))                            # (B, 32, H, W)

        # --- 特征融合与最终输出 ---
        final_features = torch.cat((d3, d1_side, d2_side), dim=1) # (B, 96, H, W)
        out = self.conv1(final_features) # (B, out_channels, H, W)

        return out


#Residual Dense Block
# class Dense_Block_IN(nn.Module):
#     def __init__(self, block_num, inter_channel, channel, with_residual=True):
#         super(Dense_Block_IN, self).__init__()
#         concat_channels = channel + block_num * inter_channel
#         channels_now = channel

#         self.group_list = nn.ModuleList([])
#         for i in range(block_num):
#             group = nn.Sequential(
#                 nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
#                           stride=1, padding=1),
#                 nn.InstanceNorm2d(inter_channel, affine=True),
#                 nn.ReLU(),
#             )
#             self.add_module(name='group_%d' % i, module=group)
#             self.group_list.append(group)
#             channels_now += inter_channel
#         assert channels_now == concat_channels
#         self.fusion = nn.Sequential(
#             nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
#             nn.InstanceNorm2d(channel, affine=True),
#             nn.ReLU(),
#         )
#         self.with_residual = with_residual

#     def forward(self, x):
#         feature_list = [x]
#         for group in self.group_list:
#             inputs = torch.cat(feature_list, dim=1)
#             outputs = group(inputs)
#             feature_list.append(outputs)
#         inputs = torch.cat(feature_list, dim=1)
#         fusion_outputs = self.fusion(inputs)
#         if self.with_residual:
#             block_outputs = fusion_outputs + x
#         else:
#             block_outputs = fusion_outputs

#         return block_outputs


# class AL_Area_Selfguidance_Color_Correction(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2):
#         super(AL_Area_Selfguidance_Color_Correction, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.RDB1 = Dense_Block_IN(4, 32, 64)
#         self.Down_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
#         self.RDB2 = Dense_Block_IN(4, 32, 128)
#         self.Down_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
#         self.RDB3 = Dense_Block_IN(4, 32, 256)
#         self.Up_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
#         self.RDB4 = Dense_Block_IN(4, 32, 128 + 128)
#         self.Up_2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, output_padding=1)
#         self.RDB5 = Dense_Block_IN(4, 32, 64 + 64)
#         self.conv3 = nn.Conv2d(64 + 64, out_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x, y):
#         x = x * y
#         x = self.conv1(x)
#         y = self.conv2(y)
#         x = torch.cat((x, y), dim=1)
#         x1 = self.RDB1(x)
#         x2 = self.RDB2(self.Down_1(x1))
#         x = self.RDB3(self.Down_2(x2))
#         x = self.Up_1(x)
#         x = torch.cat((x, x2), dim=1)
#         x = self.Up_2(self.RDB4(x))
#         x = torch.cat((x, x1), dim=1)
#         x = self.RDB5(x)
#         x = self.conv3(x)
#         return x


class Dense_Block_IN(nn.Module):
    def __init__(self, block_num, inter_channel, channel, with_residual=True):
        super(Dense_Block_IN, self).__init__()
        concat_channels = channel + block_num * inter_channel
        channels_now = channel

        self.group_list = nn.ModuleList([])
        for i in range(block_num):
            group = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
                          stride=1, padding=1),
                nn.InstanceNorm2d(inter_channel, affine=True),
                nn.ELU(),
            )
            self.group_list.append(group)
            channels_now += inter_channel
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ELU(),
        )
        self.with_residual = with_residual

    def forward(self, x):
        feature_list = [x]
        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)
        inputs_fusion = torch.cat(feature_list, dim=1)
        fusion_outputs = self.fusion(inputs_fusion)
        if self.with_residual:
            block_outputs = fusion_outputs + x
        else:
            block_outputs = fusion_outputs
        return block_outputs

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # 沿通道维度平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 沿通道维度最大池化
        x_pool = torch.cat([avg_out, max_out], dim=1) # 拼接
        x_att = self.conv1(x_pool) # 卷积降维到1个通道
        return self.sigmoid(x_att)

class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.ca(x) * x # 通道注意力加权
        x_out = self.sa(x_out) * x_out # 空间注意力加权
        return x_out


# --- Begin: Integrated Network ---

class AL_Area_Selfguidance_Color_Correction(nn.Module):
    # 移除 in_channels 参数，因为 x 固定为 1 通道, y 固定为 2 通道
    def __init__(self, out_channels=2, img_size=256,
                 transformer_layers=4, transformer_heads=4, transformer_vis=False):
        super(AL_Area_Selfguidance_Color_Correction, self).__init__()

        # --- U-Net Encoder Path ---
        # conv1 处理 x * y (广播后为 2 通道)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        # conv2 处理 y (2 通道)
        self.conv2 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)

        # RDB1 输入通道数为 conv1输出(32) + conv2输出(32) = 64
        self.RDB1 = Dense_Block_IN(4, 32, 64) # Out: 64 channels, H x W
        self.Down_1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Out: H/2 x W/2
        self.RDB2 = Dense_Block_IN(4, 32, 128) # Out: 128 channels, H/2 x W/2
        self.Down_2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # Out: H/4 x W/4
        self.RDB3 = Dense_Block_IN(4, 32, 256) # Out: 256 channels, H/4 x W/4

        # --- Channel Transformer Integration ---
        self.transformer_channel_num = [64, 128, 256, 0] # Channels of RDB1, RDB2, RDB3 outputs
        self.transformer_kv_size = sum(c for c in self.transformer_channel_num if c > 0) # 448
        # 根据 img_size 动态计算 patch sizes 或者保持固定，这里假设 img_size=256
        patch_size_base = img_size // 8 # e.g., 32 for 256x256
        self.transformer_patch_size = [patch_size_base, patch_size_base // 2, patch_size_base // 4, patch_size_base // 8] # [32, 16, 8, 4]

        self.transformer = ChannelTransformer(
            vis=transformer_vis,
            img_size=img_size,
            channel_num=self.transformer_channel_num,
            patchSize=self.transformer_patch_size,
            kv_size=self.transformer_kv_size,
            num_layers=transformer_layers,
            num_heads=transformer_heads
        )

        # --- U-Net Decoder Path ---
        self.Up_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # RDB4 input = Up_1(128) + trans_enc2(128) = 256
        self.RDB4 = Dense_Block_IN(4, 32, 256) # Input 256, Output 256
        self.Up_2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        # RDB5 input = Up_2(64) + trans_enc1(64) = 128
        self.RDB5 = Dense_Block_IN(4, 32, 128) # Input 128, Output 128
        # Final Convolution, input from RDB5 (128 channels)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        # x: (B, 1, H, W), y: (B, 2, H, W)
        # input_guidance = x * y 使用广播 -> (B, 2, H, W)
        input_guidance = x * y
        x_feat = self.conv1(input_guidance) # (B, 32, H, W)
        y_feat = self.conv2(y)          # (B, 32, H, W)

        # --- Encoder ---
        enc1_in = torch.cat((x_feat, y_feat), dim=1) # (B, 64, H, W)
        enc1 = self.RDB1(enc1_in)                 # (B, 64, H, W)

        enc2_in = self.Down_1(enc1)               # (B, 128, H/2, W/2)
        enc2 = self.RDB2(enc2_in)                 # (B, 128, H/2, W/2)

        enc3_in = self.Down_2(enc2)               # (B, 256, H/4, W/4)
        enc3 = self.RDB3(enc3_in)                 # (B, 256, H/4, W/4)

        # --- Transformer ---
        trans_enc1, trans_enc2, trans_enc3, _, attn_weights = self.transformer(enc1, enc2, enc3, None)

        # --- Decoder ---
        dec1_up = self.Up_1(trans_enc3) # (B, 128, H/2, W/2)
        dec1_cat = torch.cat((dec1_up, trans_enc2), dim=1) # (B, 256, H/2, W/2)
        dec1 = self.RDB4(dec1_cat)      # (B, 256, H/2, W/2)

        dec2_up = self.Up_2(dec1)       # (B, 64, H, W)
        dec2_cat = torch.cat((dec2_up, trans_enc1), dim=1) # (B, 128, H, W)
        dec2 = self.RDB5(dec2_cat)      # (B, 128, H, W)

        # --- Final Output ---
        output = self.conv3(dec2)       # (B, out_channels, H, W)

        return output



class Detail_Enhancement(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Detail_Enhancement, self).__init__()
        self.Down_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.DB_1 = Dense_Block_IN(4, 32, 32, with_residual=False)
        self.Down_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.DB_2 = Dense_Block_IN(4, 32, 64, with_residual=False)
        self.Down_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.DB_3 = Dense_Block_IN(4, 32, 128, with_residual=False)
        self.Down_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.DB_4 = Dense_Block_IN(4, 32, 256, with_residual=False)
        self.Up_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        
        self.DB_5 = Dense_Block_IN(4, 32, 128 + 128, with_residual=False)
        self.Up_2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2)
        self.DB_6 = Dense_Block_IN(4, 32, 64 + 64, with_residual=False)
        self.Up_3 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2)
        self.DB_7 = Dense_Block_IN(4, 32, 32 + 32, with_residual=False)
        self.Up_4 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=3, stride=2, output_padding=1)
        self.DB_8 = Dense_Block_IN(4, 32, 16 + in_channels, with_residual=False)
        self.conv1 = nn.Conv2d(16 + in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = x
        x1 = self.DB_1(self.Down_1(x))
        x2 = self.DB_2(self.Down_2(x1))
        x3 = self.DB_3(self.Down_3(x2))
        x = self.DB_4(self.Down_4(x3))
        x = self.Up_1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.Up_2(self.DB_5(x))
        x = torch.cat((x, x2), dim=1)
        x = self.Up_3(self.DB_6(x))
        x = torch.cat((x, x1), dim=1)
        x = self.Up_4(self.DB_7(x))
        x = torch.cat((x, x0), dim=1)
        x = self.DB_8(x)
        x = self.conv1(x)
        return x
# class Detail_Enhancement(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super(Detail_Enhancement, self).__init__()

#         # --- 编码器部分 ---
#         self.Down_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
#         self.DB_1 = Dense_Block_IN(4, 32, 32, with_residual=False)
#         self.cbam1 = CBAM(32) # CBAM after DB_1

#         self.Down_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.DB_2 = Dense_Block_IN(4, 32, 64, with_residual=False)
#         self.cbam2 = CBAM(64) # CBAM after DB_2

#         self.Down_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.DB_3 = Dense_Block_IN(4, 32, 128, with_residual=False)
#         self.cbam3 = CBAM(128) # CBAM after DB_3

#         self.Down_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.DB_4 = Dense_Block_IN(4, 32, 256, with_residual=False)
#         self.cbam4 = CBAM(256) # CBAM after DB_4 (bottleneck)

#         # --- 解码器部分 ---
#         # 上采样层，同样使用 k=4,s=2,p=1 保证尺寸
#         self.Up_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.DB_5 = Dense_Block_IN(4, 32, 128 + 128, with_residual=False) # 输入是 cat(x3_att, up1)
#         self.cbam5 = CBAM(128 + 128) # CBAM after DB_5

#         self.Up_2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
#         self.DB_6 = Dense_Block_IN(4, 32, 64 + 64, with_residual=False) # 输入是 cat(x2_att, up2)
#         self.cbam6 = CBAM(64 + 64) # CBAM after DB_6

#         self.Up_3 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1)
#         self.DB_7 = Dense_Block_IN(4, 32, 32 + 32, with_residual=False) # 输入是 cat(x1_att, up3)
#         self.cbam7 = CBAM(32 + 32) # CBAM after DB_7

#         self.Up_4 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)
#         self.DB_8 = Dense_Block_IN(4, 32, 16 + in_channels, with_residual=False) # 输入是 cat(x0, up4)
#         self.cbam8 = CBAM(16 + in_channels) # CBAM after DB_8

#         # --- 输出层 ---
#         self.conv1 = nn.Conv2d(16 + in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # --- 编码器 ---
#         x0 = x
#         d1 = self.Down_1(x0)
#         db1_out = self.DB_1(d1)
#         x1 = self.cbam1(db1_out) # 应用 CBAM

#         d2 = self.Down_2(x1) # 注意输入是 x1 (CBAM 处理后)
#         db2_out = self.DB_2(d2)
#         x2 = self.cbam2(db2_out) # 应用 CBAM

#         d3 = self.Down_3(x2)
#         db3_out = self.DB_3(d3)
#         x3 = self.cbam3(db3_out) # 应用 CBAM

#         d4 = self.Down_4(x3)
#         db4_out = self.DB_4(d4)
#         bottle_neck = self.cbam4(db4_out) # 应用 CBAM (瓶颈层)

#         # --- 解码器 ---
#         up1 = self.Up_1(bottle_neck)
#         # 跳跃连接过来的特征 (x3) 已经是经过 CBAM 处理的
#         cat1 = torch.cat((x3, up1), dim=1) # 使用 CBAM 处理过的 x3
#         db5_out = self.DB_5(cat1)
#         cbam5_out = self.cbam5(db5_out) # 应用 CBAM

#         up2 = self.Up_2(cbam5_out) # 输入是上一个 CBAM 处理过的结果
#         cat2 = torch.cat((x2, up2), dim=1) # 使用 CBAM 处理过的 x2
#         db6_out = self.DB_6(cat2)
#         cbam6_out = self.cbam6(db6_out) # 应用 CBAM

#         up3 = self.Up_3(cbam6_out)
#         cat3 = torch.cat((x1, up3), dim=1) # 使用 CBAM 处理过的 x1
#         db7_out = self.DB_7(cat3)
#         cbam7_out = self.cbam7(db7_out) # 应用 CBAM

#         up4 = self.Up_4(cbam7_out)
#         # x0 是原始输入，没有经过 CBAM，如果需要也可以加
#         cat4 = torch.cat((x0, up4), dim=1)
#         db8_out = self.DB_8(cat4)
#         cbam8_out = self.cbam8(db8_out) # 应用 CBAM

#         # --- 输出层 ---
#         # 使用最后一个 CBAM 模块的输出
#         out = self.conv1(cbam8_out)
#         return out    


class FCALayer(nn.Module):
    """
    FcaNet Channel Attention Layer
    使用 SciPy 的 DCT 选择频率分量来计算通道注意力权重。
    """
    def __init__(self, channel, reduction=16):
        super(FCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化 (Squeeze)
        # 确保 k 至少为 1
        self.k = max(1, int(channel * 1.0 / reduction)) # 确保 k 是整数，并且至少为1
        # print(f"FCALayer for {channel} channels: keeping k={self.k} frequency components.")
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # 获取原始设备（CPU 或 GPU）
        original_device = x.device

        y = self.avg_pool(x)
        y_squeezed = y.squeeze(-1).squeeze(-1) # Shape: (b, c)

        # --- 使用 SciPy 进行 DCT ---
        # 1. 将 PyTorch 张量转换为 NumPy 数组 (需要移到 CPU 并 detach)
        y_squeezed_np = y_squeezed.detach().cpu().numpy()

        # 2. 应用 SciPy DCT (使用 axis=1 对应 PyTorch 的 dim=1)
        #    注意：scipy.fft.dct 默认 type=2, 与 torch.fft.dct 兼容
        y_dct_np = scipy.fft.dct(y_squeezed_np, norm='ortho', axis=1)

        # 3. 将 NumPy 结果转换回 PyTorch 张量，并移回原始设备
        #    确保数据类型匹配（通常是 float32）
        y_dct = torch.from_numpy(y_dct_np.astype(np.float32)).to(original_device)
        # --- SciPy DCT 结束 ---

        # 创建一个零张量来存储选定的频率分量
        y_weights_dct = torch.zeros_like(y_dct)
        # 只保留前 k 个频率分量
        y_weights_dct[:, :self.k] = y_dct[:, :self.k]

        # --- 使用 SciPy 进行 IDCT ---
        # 1. 将包含选定权重的 PyTorch 张量转换为 NumPy 数组
        y_weights_dct_np = y_weights_dct.detach().cpu().numpy()

        # 2. 应用 SciPy IDCT
        y_weights_idct_np = scipy.fft.idct(y_weights_dct_np, norm='ortho', axis=1)

        # 3. 将 NumPy 结果转换回 PyTorch 张量，并移回原始设备
        y_weights_idct = torch.from_numpy(y_weights_idct_np.astype(np.float32)).to(original_device)
        # --- SciPy IDCT 结束 ---

        # 恢复维度以匹配卷积层输入 (b, c, 1, 1)
        y_weights_idct = y_weights_idct.unsqueeze(-1).unsqueeze(-1)

        # 通过全连接层（用 1x1 卷积实现）和 Sigmoid 得到最终注意力权重
        attn_weights = self.sigmoid(self.fc(y_weights_idct))

        # 将注意力权重应用到原始输入特征图上
        output = x * attn_weights.expand_as(x)
        return output

class Channels_Fusion_with_FCANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, reduction=16): # 添加 reduction 参数
        super(Channels_Fusion_with_FCANet, self).__init__()
        self.relu = nn.ELU(inplace=True) # 或者使用 nn.ReLU(inplace=True)

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # 在第二个卷积层和激活函数之后应用 FCANet 注意力
        # FCANet 需要知道输入的通道数，这里是 32
        self.fca = FCALayer(channel=32, reduction=reduction) # 将 reduction 传递给 FCALayer

        # 第三个卷积层 (最终输出)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
     

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # 应用 FCANet 注意力
        x = self.fca(x) # 在激活之后应用注意力

        x = self.conv3(x)
        return x
    
class Channels_Fusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Channels_Fusion, self).__init__()
        self.relu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AL_Area_Selfguidance_Color_Correction().to(device)
    summary(model, [(1,256, 256),(2, 256, 256)])
    print('Well Done!')