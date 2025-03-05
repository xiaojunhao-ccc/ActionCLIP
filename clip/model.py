from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
from einops import rearrange

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QuickGELU: 通过 x * sigmoid(1.702 * x) 近似 GELU，计算更快，适用于大规模 Transformer 任务"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    在 batch 级别随机丢弃部分样本的残差路径
    "一条高速公路上有多个收费站 (残差路径)，每辆车 (样本) 要随机决定是否通过某些收费站 (是否进入残差路径)，但最终在推理时所有收费站都会打开。"
    ----------
    x : torch.Tensor
        输入张量 (Tensor)
    drop_prob : float, 默认 0.
        丢弃的概率，取值范围 [0,1]，若为 0，则不进行丢弃
    training : bool, 默认 False
        是否处于训练模式，仅在训练时启用 Drop Path
    ----------
    torch.Tensor
        经过 Drop Path 处理后的张量
    """
    # 若 drop_prob 为 0 或当前不处于训练模式，则直接返回输入 x，不进行 Drop Path 操作
    if drop_prob == 0. or not training:
        return x
    # 计算保留的概率
    keep_prob = 1 - drop_prob 
    
    # 生成一个随机张量，与 x 具有相同 batch 维度的形状，其他维度均为 1，每个样本单独应用随机深度
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 例如：x (8, 3, 224, 224) -> shape (8, 1, 1, 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 随机在一个 batch 中丢弃部分样本
    # 使用 floor_() 进行二值化，使其变为 0 或 1，即以 drop_prob 的概率丢弃路径
    mask.floor_()
    
    # 由于部分路径被丢弃，为保持整体期望不变，需要用 keep_prob 进行归一化 (放大)，以确保在 training=True 时，模型的输出分布与 training=False 时保持一致
    output = x.div(keep_prob) * mask  
    
    return output

class DropPath(nn.Module):
    """
    Drop Path (随机深度) 模块，用于神经网络的残差路径中进行随机丢弃，起到正则化的作用。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob  # 存储丢弃概率

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ResidualAttentionBlock(nn.Module):
    """残差注意力模块"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([  # 有序字典，会记住键值对的插入顺序
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

"""残差注意力网络"""
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisualTransformer(nn.Module):  
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, dropout=None, emb_dropout=0., joint=False, T=None):
        """
        视觉 Transformer (ViT) 结构，适用于图像分类等任务。

        参数:
        - input_resolution (int): 输入图像的分辨率 (例如 224 表示 224x224)。
        - patch_size (int): 每个 patch 的大小 (例如 16 表示 16x16)。
        - width (int): Transformer 的隐藏层维度 (即 embedding 维度)。
        - layers (int): Transformer 的层数 (encoder 块的数量)。
        - heads (int): 多头注意力机制中的头数。
        - output_dim (int): 最终输出的特征维度。
        - dropout (float, 可选): Transformer 中的 Dropout 概率，默认为 None。
        - emb_dropout (float, 可选): 是否在 embedding 之后应用 Dropout(用于正则化)。
        - joint (bool, 可选): 是否启用时空联合嵌入 (用于视频数据)。
        - T (int, 可选): 视频片段数
        """
        super().__init__()
        
        # 保存输入分辨率和输出维度
        self.input_resolution = input_resolution  # 输入图像的分辨率，例如 224×224
        self.output_dim = output_dim  
        
        # **卷积层：将输入图像转换为 patch embedding**
        # 这里使用一个 2D 卷积层，类似于 ViT 直接将图像分割为 patch，并进行 embedding
        self.conv1 = nn.Conv2d(
            in_channels=3,      # 输入通道数，RGB 图像为 3
            out_channels=width, # 输出通道数，即 embedding 维度
            kernel_size=patch_size, # patch 大小，例如 16×16
            stride=patch_size,   # 步长等于 patch 大小，相当于划分成不重叠的 patch
            bias=False          # 不使用偏置
        )

        # **初始化 class token 和位置编码**
        # 初始化时控制输出方差至 1 附近：如果每一层的输出方差 增大 (>1)，那么经过多层传播后，信号会不断放大，最终溢出；反之，如果方差 减小 (<1)，那么信号会不断衰减，导致梯度消失
        scale = width ** -0.5  # 1/sqrt(n) 归一化，控制神经元的输出方差 | 根据方差乘法法则，输入维度增大时，输出方差也会增大
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # cls 分类 token (可训练)
        patch_num = (input_resolution//patch_size)**2  # 被打成多少个 patch
        self.positional_embedding = nn.Parameter(  # 位置编码，包括 class token 位置
            scale * torch.randn(patch_num+1, width)  # +1 为 额外的 CLS Token
        )  
        
        # Embedding 层的 Dropout(如果设置了)
        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout  # dropout 率
         # 输出 embedding Dropout 信息
        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        # 层归一化
        self.ln_pre = LayerNorm(width)  

        # **Transformer 编码器**
        self.transformer = Transformer(width, layers, heads, dropout=dropout)  

        # **输出层**
        self.ln_post = LayerNorm(width)  # 层归一化
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # 最终投影到输出维度

        # 是否启用时空联合嵌入 (通常用于视频) --- BY: ActionClip
        self.joint = joint  
        # **时空联合嵌入 (仅在 joint 模式下启用)**
        if joint:
            print('=====using joint space-time====')   # 时间编码 | 仿位置编码引入位置信息，时间编码引入时间信息
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))  # T:视频片段数 (时间步)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        参数:
        - x (Tensor): 输入的图像张量，形状为 (batch_size, 3, input_resolution, input_resolution)

        返回:
        - x (Tensor): 处理后的特征向量，形状为 (batch_size, output_dim)
        """
        
        # **1. Patch Embedding：将输入图像转换为 patch 级别的 token**
        x = self.conv1(x)  # 形状变为 (batch_size, width, grid, grid)，其中 grid = input_resolution // patch_size
        
        # **2. 调整形状**
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 变为 (batch_size, width, patch_num), patch 的数量 patch_num = grid*grid
        x = x.permute(0, 2, 1)  # 变为 (batch_size, patch_num, width)，符合 Transformer 输入格式

        # **3. 拼接 x 和 class token**
        # 利用自动广播扩展为 batchsize 个 class token
        batch_class_tokens = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                            dtype=x.dtype, device=x.device)
        x = torch.cat([batch_class_tokens, x], dim=1)  # 形状变为 (batch_size, patch_num + 1, width)
        
        # **4. 加入位置编码**
        x = x + self.positional_embedding.to(x.dtype)

        # **5. 处理时空联合嵌入 (仅适用于视频)**
        if self.joint:
            B = x.shape[0] // self.T  # 计算 batch 维度的样本数量
            cls_tokens = x[:B, 0, :].unsqueeze(1)  # 获取 class token
            x = x[:, 1:]  # 移除 class token，因为时间编码不加在 class token 上 |(class token 只加了位置编码)
            
            # **对时序维度进行调整**
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=self.T)  # 重新排列时间维度 | n: patch_num | m: width-每个 patch 的特征维度 
            x = x + self.time_embedding.to(x.dtype)  # 加入时间嵌入
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=self.T)  # 变回原格式
            x = torch.cat((cls_tokens, x), dim=1)  # 重新添加 class token -> (batch_size, patch_num+1, width) 即 (batch_size, seq_len, width) 
        
        # **6. Embedding Dropout(如果启用)**
        if self.emb_dropout > 0:
            x = self.dropout(x)

        # **7. 归一化**
        x = self.ln_pre(x)

        # **8. 进入 Transformer 编码器**
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, width) -> (seq_len, batch_size, width)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, width) -> (batch_size, seq_len, width)

        # **9. 取出 CLS token 表示整个视频序列的特征**
        CLS = self.ln_post(x[:, 0, :])  # (batch_size, 1, width)

        # **10. 进行投影 (如果启用)**
        if self.proj is not None:
            image_features = CLS @ self.proj  # 矩阵乘法，将特征投影到 output_dim 维度
        
        return image_features  # 返回最终特征

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # 视觉 (Vision) 部分
                 image_resolution: int, # 输入图像的分辨率 (如 224)
                 vision_layers: Union[Tuple[int, int, int, int], int], # 视觉 Transformer 的层数
                 vision_width: int,  # 视觉 Transformer 的隐藏层宽度
                 vision_patch_size: int,  # Patch 的大小 (ViT 模型)
                 # 文本 (Text) 部分
                 context_length: int,  # 最大文本长度 (序列长度)
                 vocab_size: int, # 词表大小 (Embedding 层的输入维度)
                 transformer_width: int, # Transformer 的隐藏层宽度
                 transformer_heads: int, # Transformer 的注意力头数
                 transformer_layers: int, # Transformer 的层数
                 # 其他参数
                 joint=False, # 是否使用时间信息编码，和位置编码进行联合训练
                 tsm=False, # 是否使用时间偏移模块 (Temporal Shift Module
                 T=8,  # 视频片段数
                 dropout=0., # Transformer 中的 Dropout 概率
                 emb_dropout = 0. # 是否在 embedding 之后应用 Dropout(用于正则化)。
                 ):
        super().__init__()

        self.context_length = context_length

        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # 从 0 逐步递增到 dropout 的 渐进式的 Dropout 衰减
        else:
            dpr = None

        # 创建视觉 Transformer
        vision_heads = vision_width // 64  # 计算 Transformer 头数，通常设为 width/64
        self.visualTransformer = VisualTransformer(
            input_resolution=image_resolution,  # 图像输入分辨率
            patch_size=vision_patch_size,  # Patch 的大小
            width=vision_width,  # Transformer 的隐藏层宽度
            layers=vision_layers,  # 视觉 Transformer 的层数
            heads=vision_heads,  # 计算得出的多头注意力头数
            output_dim=embed_dim,  # 输出的特征维度
            dropout=dpr, # Transformer 中的 Dropout 概率，默认为 None。
            emb_dropout=emb_dropout, # 是否在 embedding 之后应用 Dropout(用于正则化)。
            joint=joint, # 是否使用时间信息编码，和位置编码进行联合训练
            T=T # 视频片段数
        )

        # 时间偏移模块，用于视频任务中的时间建模。
        if tsm:
            print('=========using TSM==========')
            from modules.temporal_shift import make_temporal_shift_vit
            make_temporal_shift_vit(self.visualTransformer, T)

        # 构建文本 Transformer
        self.transformer = Transformer(
            width=transformer_width,  # Transformer 隐藏层维度
            layers=transformer_layers,  # Transformer 层数
            heads=transformer_heads,  # Transformer 注意力头数
            attn_mask=self.build_attention_mask(),  # 注意力掩码 (mask)
            dropout=dpr  # Dropout
        )

        self.vocab_size = vocab_size  # 词表大小
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 词嵌入层
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 位置编码
        self.ln_final = LayerNorm(transformer_width)  # LayerNorm 归一化层
        
        self.dropout = nn.Dropout(emb_dropout) # embedding 层的 dropout
        self.emb_dropout = emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # project head 将文本特征投影到 embed_dim 维度
        
        τ = 0.07 # CLIP 训练时的 softmax 温度参数 τ，较小的 τ(较大的 logit_scale)：分布更加陡峭，相似度高的样本更加突出
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / τ)) # 控制 logits 在 softmax 计算中的放缩 (CLIP 训练时的 softmax 温度)

        self.initialize_parameters()  # 初始化权重

    # 权重初始化
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # 构建 因果注意力掩码 (causal attention mask) - 上三角矩阵 - 仅允许 Transformer 看到 当前 Token 及其之前的 Token，防止未来信息泄露。
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 仅保留上三角矩阵
        return mask

    @property  # 将 类方法 转换为 只读属性，从而允许像访问普通属性一样调用方法，而无需加括号
    def dtype(self):
        return self.visualTransformer.conv1.weight.dtype

    # 视觉编码器 (ViT)
    def encode_image(self, image):
        return self.visualTransformer(image.type(self.dtype))

    # 文本编码器
    def encode_text(self, text):
        # 将输入文本转换为 token 嵌入，形状为 [batch_size, n_ctx(上下文长度), transformer_width]
        x = self.token_embedding(text).type(self.dtype)  
        # 加上可训练的位置编码，保留序列位置信息
        x = x + self.positional_embedding.type(self.dtype)
        # 若 emb_dropout > 0，则应用 Dropout 进行正则化
        if self.emb_dropout > 0:
            x = self.dropout(x)
        
        # 通过 Transformer 进行文本编码
        x = x.permute(1, 0, 2)  # 调整维度为 [n_ctx, batch_size, transformer_width] 以适配 Transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 还原维度为 [batch_size, n_ctx, transformer_width]

        # 通过 layerNorm 层归一化数据
        x = self.ln_final(x).type(self.dtype)

        # 使用 EOT (End-of-Text) token 对应的特征作为整个文本序列的表示 (类似 Bert 用 [cls] token)
        EOT = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  

        # 通过 `text_projection` 进行线性变换，得到最终的文本特征
        text_features  = EOT @ self.text_projection  

        # 返回文本的编码向量
        return text_features 

    def forward(self, image, text):
        # 提取文本、图像特征
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 归一化特征向量：沿着特征维度计算特征向量的模，并用特征向量除以模
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # xxx_features.norm: 计算每个特征向量的模 (L2 范数)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算 图像 和 文本 间的 相似度
        logit_scale = self.logit_scale.exp() # 温度参数 τ 的倒数
        logits_per_image = logit_scale * image_features @ text_features.t() # image->text 相似度
        logits_per_text = logit_scale * text_features @ image_features.t() # text->image 相似度

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """将模型中的适用参数转换为 fp16 (半精度浮点数)"""
    
    def _convert_weights_to_fp16(l):
        
        # 如果层是 1D 或 2D 卷积层，或者是全连接层 (线性层)
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 将权重转换为半精度浮点数 (fp16)
            l.weight.data = l.weight.data.half()
            # 如果该层有偏置项，也将其转换为 fp16
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
       
        # 如果层是 MultiheadAttention(多头注意力机制)
        if isinstance(l, nn.MultiheadAttention):
            """
            遍历需要转换的权重参数，包括：
            "in_proj_weight"(输入投影权重)
            "q_proj_weight"(Q 投影权重)
            "k_proj_weight"(K 投影权重)
            "v_proj_weight"(V 投影权重)
            "in_proj_bias"(输入投影偏置)
            "bias_k"(K 偏置)
            "bias_v"(V 偏置)"""
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)  # 获取相应名称的参数
                # 如果该权重存在，则转换为 fp16
                if tensor is not None:
                    tensor.data = tensor.data.half()
        
        # 处理 `text_projection` 和 `proj` 这两个可能的额外参数
        for name in ["text_projection", "proj"]:
            # 如果该层有这些属性
            if hasattr(l, name):
                attr = getattr(l, name) # 获取相应名称的参数
                # 如果该参数存在，则转换为 fp16
                if attr is not None:
                    attr.data = attr.data.half()
    
    # 对 model 的所有子模块递归应用 `_convert_weights_to_fp16`
    model.apply(_convert_weights_to_fp16)


# 构建 CLIP 模型
def build_model(state_dict:dict, tsm=False,T=8, dropout=0., joint=False, emb_dropout=0., pretrain=True):
    """
    根据给定的 `state_dict`(预训练权重字典) 构建 CLIP 模型，并根据 `tsm`、`pretrain` 等参数进行调整。
        参数：
        - state_dict (dict): 预训练模型的权重字典。
        - tsm (bool): 是否使用 Temporal Shift Module(时间偏移模块)。
        - T (int): 视频片段数，时序维度的长度 (如果 `tsm=True`)。
        - dropout (float): Transformer 层的 dropout 率。
        - joint (bool): 是否使用时间信息编码，和位置编码进行联合训练。
        - emb_dropout (float): 词向量层的 dropout 率。
        - pretrain (bool): 是否加载完整的预训练模型。
        返回：
        - model (nn.Module): 经过初始化的 CLIP 模型，默认处于 `eval()` 模式。
    """
    # 判断 `state_dict` 是否是 ViT 结构
    assert "visual.proj" in state_dict, 'THIS IS NOT VIT MODEL > "visual.proj" not in state_dict'

    # 从权重文件中提取模型信息
    vision_width = state_dict["visual.conv1.weight"].shape[0]  # conv1.weight.shape=(Cout,Cin,PatchSize,PatchSize)
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    # 构建 CLIP 模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  tsm=tsm, T=T, joint=joint,
        dropout=dropout, emb_dropout=emb_dropout
    )

    # 删除 `state_dict` 中无关的键
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # 如果启用了 `tsm`(时间偏移)，需要调整 `state_dict` 的键名
    if tsm:
        for k in list(state_dict.keys()):
            
            # 处理 `conv1` 层的参数，修改键名以匹配 `TSM` 结构
            if k.find("conv1")>-1 and k.find("layer")>-1: 
                n_k = k.split('conv1.')[0]+'conv1.net.'+k.split('conv1.')[1]
                state_dict[n_k] = state_dict.pop(k)
            
            # 处理 `resblocks` 层的参数，修改键名以匹配 `TSM` 结构
            if k.find("resblocks")>-1 and k.find("visual")>-1: 
                tmp = ''
                for i, t_ in enumerate(k.split('resblocks.')[1].split('.')):
                    if i>=1:
                        tmp += '.' + t_ 
                n_k = k.split('resblocks.')[0]+'resblocks.' + k.split('resblocks.')[1].split('.')[0]+'.net'+ tmp
                state_dict[n_k] = state_dict.pop(k)

    # 转换模型参数为 `fp16`(半精度)
    convert_weights(model)
    
    # 加载预训练权重
    if pretrain:
        print('loading clip pretrained model!')
        if joint:  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict,strict=False)
        else:
            model.load_state_dict(state_dict)
    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

    # 默认返回 `eval()` 模式的模型
    return model.eval()
