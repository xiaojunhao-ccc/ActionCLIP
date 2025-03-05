import torch
from torch import nn
from collections import OrderedDict

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# QuickGELU: 通过 x * sigmoid(1.702 * x) 近似 GELU，计算更快，适用于大规模 Transformer 任务
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# 残差注意力模块
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_head:int, attn_mask:torch.Tensor=None):  # d_model: 表示输入特征的维度
        super().__init__()

        self.multi_head_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 线性变换，维度扩展到 4 倍
            ("gelu", QuickGELU()), # QuickGELU 激活函数，比 GELU 更快，适用于大规模 Transformer 任务
            ("c_proj", nn.Linear(d_model * 4, d_model)) # 线性变换，将维度恢复
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask # 注意力掩码

    # 计算多头自注意力
    def multi_attention(self, x: torch.Tensor):
        if self.attn_mask is not None: self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.multi_head_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.multi_attention(self.ln_1(x)) # 计算注意力，并加入残差连接，缓解梯度消失问题。
        x = x + self.mlp(self.ln_2(x)) # 通过 MLP 计算非线性变换，并加入残差连接
        return x
    
# 时间 Transformer（Temporal Transformer），用于处理时间序列数据，增强 Transformer 在时间维度上的建模能力 
class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # 创建多个残差注意力块（ResidualAttentionBlock），形成完整的 Transformer 结构
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]) 

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


# 基于 Transformer 的时序特征聚合模块
class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length  # 视频帧序列的长度
        drop_rate = 0.
        # Transformer
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)  # Transformer 编码层，d_model 输入的特征维度，nhead 多头自注意力的头数
        self.transformer_enc = nn.TransformerEncoder(enc_layer,  # 组合多个 Transformer 编码层形成完整的 Transformer 编码器
                                                     num_layers=n_layers, 
                                                     norm=nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 可训练的分类 token（CLS），用于全局特征表示
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length+1, embed_dim)) # 位置编码，用于注入时间位置信息
        self.pos_drop = nn.Dropout(p=drop_rate)  # 用于正则化，此处 p=0 表示未启用

        # 用正态分布初始化位置编码和分类 token
        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)  # 位置编码初始化
            trunc_normal_(self.cls_token, std=.02)  # 分类 token 初始化
        self.apply(self._init_weights)  # 初始化所有子模块的权重

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]  # 视频数量（batchsize）
        cls_tokens = self.cls_token.expand(nvids, -1, -1) # 扩展 CLS token 为 (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # 在时间维度上拼接 CLS token (batch_size, clip_length+1, embed_dim)
        x = x + self.pos_embed # 添加位置编码，保持时序信息
        x.transpose_(1, 0) # 交换 batch 维度和时间维度，变为 (clip_length+1, batch_size, embed_dim)，符合 Transformer 需要的输入格式
        o = self.transformer_enc(x) # 送入 Transformer 编码器，提取时序特征
        return o[0] # 返回 CLS token 作为最终的全局视频表示

# Post-network Prompt
class visual_prompt(nn.Module):

    def __init__(self, sim_head, clip_state_dict, num_segments):
        super().__init__()
        self.sim_header = sim_head
        self.num_segments = num_segments
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]  # Post-network Prompt 类型
        if self.sim_header == "meanP":  # 效果对于小数据集来说已经不错了
            return
        else:
            if self.sim_header == "LSTM" or self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D" :
                embed_dim = clip_state_dict["text_projection"].shape[1]
                context_length = clip_state_dict["positional_embedding"].shape[0]
                vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
                transformer_width = clip_state_dict["ln_final.weight"].shape[0]
                transformer_heads = transformer_width // 64
                transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))
                self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
            if self.sim_header == "Transf" :
                self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            if self.sim_header == "LSTM":
                self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True, bidirectional=False, num_layers=1)

            # 初始化所有子模块的权重
            self.apply(self._init_weights) # self.apply(self.init_weights) 会递归地将 self.init_weights 应用于模型的所有子模块

            if self.sim_header == "Transf_cls": # 基于 Transformer 的时序特征聚合模块
                self.transformer = TAggregate(clip_length=self.num_segments, embed_dim=embed_dim, n_layers=6)

            if self.sim_header == 'Conv_1D' :  # 基于 1D 卷积的时序特征平移（Temporal Shift Module, TSM）
                """ 通过 通道分组的 1D 卷积来对特征图进行时间维度上的偏移操作，有效建模了时序关系。"""
                # 创建 1D 卷积层，作用于时间维度 (T)，用来执行特征平移操作
                self.shift = nn.Conv1d(
                    embed_dim,         
                    embed_dim,        
                    3,                
                    padding=1,         
                    groups=embed_dim,  # 使用分组卷积（Depthwise Convolution），每个通道独立卷积
                    bias=False         # 不使用偏置
                )
                weight = torch.zeros(embed_dim, 1, 3)
                # 设置权重实现特征平移
                weight[:embed_dim//4, 0, 0] = 1.0  # 前 1/4 通道特征向前平移 (t-1) （表示过去的特征）
                weight[embed_dim//4 : embed_dim//4+embed_dim//2, 0, 1] = 1.0  # 中间 1/2 通道保持不变 (t)
                weight[-embed_dim//4:, 0, 2] = 1.0  # 后 1/4 通道特征向后平移 (t+1) （表示未来的特征）
                self.shift.weight = nn.Parameter(weight) # 将手工初始化的权重转换为可训练的参数，允许模型通过梯度下降学习更好的特征平移模式。

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == "meanP":  # 效果对于小数据集来说已经不错了
            return x.mean(dim=1, keepdim=False)
        else:
            if self.sim_header == 'Conv_1D':
                x_original = x
                x = x.view(-1, c, t)
                x = self.shift(x.float())
                x = x.permute(0, 2, 1)
                x = x.type(x_original.dtype) + x_original
            elif self.sim_header == "Transf":
                x_original = x
                seq_length = t
                position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
                position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
                frame_position_embeddings = self.frame_position_embeddings(position_ids)
                x = x + frame_position_embeddings
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = x.type(x_original.dtype) + x_original
            elif self.sim_header == "LSTM":
                x_original = x
                x, _ = self.lstm_visual(x.float())
                self.lstm_visual.flatten_parameters()
                x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
                x = x.type(x_original.dtype) + x_original
            elif self.sim_header == "Transf_cls":
                x_original = x
                return self.transformer(x).type(x_original.dtype)
            else:
                raise ValueError('Unknown optimizer: {}'.format(self.sim_header))
        
        
