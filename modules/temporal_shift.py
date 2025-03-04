# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)  # 让父目录模块优先导入
from clip.model import VisualTransformer
import numpy as np
    
# 时间偏移 VIT
class TemporalShift_VIT(nn.Module):
    """ 在 ViT 前加上 时间偏移 """
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift_VIT, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace: print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # 时间偏移
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        # ViT
        x = self.net(x)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        hw, nt, c = x.size()
        cls_ = x[0,:,:].unsqueeze(0)
        x = x[1:,:,:]
        x = x.permute(1,2,0)  # nt,c,hw
        n_batch = nt // n_segment
        h = int(np.sqrt(hw-1))
        w = h
        x = x.contiguous().view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        out = out.contiguous().view(nt, c, h*w)
        out = out.permute(2,0,1) #hw, nt, c
        out = torch.cat((cls_,out),dim=0)
        return out
    

# 在视觉变换器 (ViT) 模型中添加时间位移模块 (Temporal Shift Module)       
def make_temporal_shift_vit(net, n_segment, n_div=8, place='block'):
    """
    参数解释：
    net: 视觉变换器模型 (VisualTransformer)
    n_segment: 视频片段数 | 例如：视频的总帧数是 80, n_segment=8, 那么模型将视频分成 8 个段，每个段有 10 帧
    n_div: 将每个块分成的部分数 (默认为 8)
    place: 插入时间位移的位置 ('block' 表示在每个块上添加)
    temporal_pool: 是否使用时间池化来调整每个层级的段数
    """
    assert isinstance(net, VisualTransformer), "The net should be 'VisualTransformer'"
    assert place == 'block', "The place should be 'block', The extra place is unimplemented"

    n_segment_list = [n_segment] * 4 

    assert n_segment_list[-1] > 0  # 确保最后一个层级的片段数大于零
    print('=> n_segment per stage: {}'.format(n_segment_list))
    
    def _make_block_temporal(stage, this_segment):
        blocks = list(stage.children())
        print('=> Processing stage with {} blocks'.format(len(blocks)))
        # 遍历指定层级中的所有块，将这些块替换为有时间位移模块的`TemporalShift_VIT`
        for i, b in enumerate(blocks):
            blocks[i] = TemporalShift_VIT(b, n_segment=this_segment, n_div=n_div)
        return nn.Sequential(*(blocks))

    # 给 net.transformer.resblocks 添加 时间位移模块 (Temporal Shift Module)
    net.transformer.resblocks = _make_block_temporal(stage=net.transformer.resblocks, 
                                                     this_segment=n_segment_list[0])
