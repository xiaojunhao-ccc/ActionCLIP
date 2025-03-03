# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import numpy

# 生成用于训练的标签矩阵，表示类别相同的样本对。
def gen_label(labels):
    """
    生成用于训练的标签矩阵，表示类别相同的样本对，用于计算对比损失。
    参数：
        labels: 样本的类别标签列表，长度为 num。
    返回：
        gt: 形状为 (num, num) 的二值矩阵，若两个样本属于同一类别，则对应位置为 1 否则为 0。
            示例：gt = numpy.array([
                [1, 0, 1, 0],  # 样本 0(jumping) 与自己和样本 2(jumping) 相同
                [0, 1, 0, 0],  # 样本 1(running) 仅与自己相同
                [1, 0, 1, 0],  # 样本 2(jumping) 与自己和样本 0(jumping) 相同
                [0, 0, 0, 1]   # 样本 3(walking) 仅与自己相同
            ])
    """
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))  # 初始化全 0 的标签矩阵
    
    for i, label in enumerate(labels):  
        for k in range(num):
            if labels[k] == label:  # 如果两个样本的类别相同
                gt[i,k] = 1
    return gt

# 计算输入特征之间的相似度分数 (logits) -> 归一化 + 余弦相似度
def calc_similarity(x1, x2, logit_scale):
    """
    计算输入特征之间的相似度分数 (logits) -> 归一化 + 余弦相似度。

    参数:
        x1 (torch.Tensor): 第一组嵌入向量，形状为 [batch_size, feature_dim]。
        x2 (torch.Tensor): 第二组嵌入向量，形状为 [batch_size, feature_dim]。
        logit_scale (torch.Tensor): 训练中可学习的缩放因子，通常用于放大相似度值。

    返回:
        logits_per_x1 (torch.Tensor): x1 到 x2 的相似度矩阵，形状为 [batch_size, batch_size]。
        logits_per_x2 (torch.Tensor): x2 到 x1 的相似度矩阵，形状为 [batch_size, batch_size]。
    """
    # 对 x1 和 x2 进行 L2 归一化，使其模长为 1
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # 计算余弦相似度，并乘以 logit_scale 进行缩放
    logits_per_x1 = logit_scale * x1 @ x2.t() # x1 与 x2 之间的相似度
    logits_per_x2 = logit_scale * x2 @ x1.t() # x2 与 x1 之间的相似度

    # 返回计算出的相似度矩阵
    return logits_per_x1, logits_per_x2

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()