import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
from PIL import Image
import torch

# 定义一个用于批量图像转换的类
class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform # 图像转换器

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group] # 对图像组中的每一张图像进行转换

# 定义一个将图像转换为 Torch 张量的类
class ToTorchFormatTensor(object):
    """ 将 PIL.Image (RGB) 或 numpy.ndarray (H x W x C) 范围 [0, 255]
        转换为 torch.FloatTensor 形状 (C x H x W) 范围 [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div # div 为 True 时，将图像像素值除以 255

    def __call__(self, pic):
        if isinstance(pic, np.ndarray): # 如果输入图像是 numpy 数组
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else: # 如果输入图像是 PIL.Image 对象
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes())) # 将图像转换为字节张量
            img = img.view(pic.size[1], pic.size[0], len(pic.mode)) # 将字节张量转换为图像的维度
            img = img.transpose(0, 1).transpose(0, 2).contiguous() # (H x W x C) -> (C x H x W)
        
        if self.div:
            return img.float().div(255)
        else:
            return img.float()

# 定义一个将图像堆叠的类 > 将多张图像按照通道维度进行堆叠
class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll # roll 为 True 时，将图像的通道维度转换为 (C x H x W)

    def __call__(self, img_group): # img_group 是一个图像组，其中包含多张图像
        if img_group[0].mode == 'L': # 如果图像是灰度图
            # np.expand_dims(x, 2) 在 第 2 维（索引为 2）添加一个新维度，从 (H, W) 变成 (H, W, 1)
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2) # 将 img_group 中的图像堆叠 (H, W, 1)->(H, W, N)
        elif img_group[0].mode == 'RGB': # 如果图像是 RGB 图
            if self.roll:
                img_group = [np.array(x)[:, :, ::-1] for x in img_group] # [:, :, ::-1] 表示将通道维度进行翻转（RGB ↔ BGR），其他维度不变。
                return np.concatenate(img_group, axis=2) # 在通道维度进行拼接
            else:
                rst = np.concatenate(img_group, axis=2)
                return rst

# 存储视频的元数据，包括 路径 (path)、帧数 (num_frames) 和 标签 (label)
class VideoRecord(object):
    def __init__(self, row):
        self._data = row
    
    @property # 通过 @property 方式提供访问接口
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Action_DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.list_file = list_file  # 视频列表文件路径
        self.num_segments = num_segments  # 采样的时间段数
        self.seg_length = new_length  # 每个时间段采样的帧数
        self.image_tmpl = image_tmpl  # 读取图片的文件名模板
        self.transform = transform  # 图像转换操作
        self.random_shift = random_shift  # 是否进行随机偏移
        self.test_mode = test_mode  # 是否是测试模式
        self.loop = False  # 是否循环采样
        self.index_bias = index_bias  # 帧索引偏移
        self.labels_file = labels_file  # 类别标签文件路径

        # 自动根据文件名模板设置索引偏移
        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1

        self._parse_list()  # 解析视频列表
        self.initialized = False  # 记录是否已初始化

    # 加载指定索引的图像
    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')] # [img]
    
    # 计算视频总的帧数
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    # 读取类别标签
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    # 解析视频列表文件，并将文件中每一行数据都转换为 VideoRecord 对象
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    # 训练模式 - 随机采样
    def _sample_indices(self, record):
        """用于在训练阶段随机采样视频帧，确保训练时的帧序列具有一定的随机性，增强模型的泛化能力。"""
        
        """视频帧数小于所需采样数 - (随机) 补全帧数"""
        if record.num_frames <= self.total_length:
            if self.loop: # (随机) 循环采样补全帧数 - 通过 np.mod() 进行循环填充，并加上一个随机偏移量，保证帧索引随机性。
                return np.mod(np.arange(self.total_length) + randint(record.num_frames//2),  # [0,...,total_length-1]->[0+offset,...,total_length-1+offset]（添加一个随机偏移）
                            record.num_frames) + self.index_bias  # 取模，确保索引不会超出视频帧范围 (循环采样)。加上索引偏移量，确保索引从 1（或 0）开始
            else: # 非循环模式 - (随机) 重复填充索引 - 用 randint() 随机重复已有帧索引，填充到 total_length。
                fill_size = self.total_length-record.num_frames # 还需要填充 fill_size 帧
                offsets = np.concatenate((
                    np.arange(record.num_frames), # 先添加已有的视频帧索引
                    randint(record.num_frames, size=fill_size))) # 随机重复采样已有帧索引，额外填充不足的部分
                return np.sort(offsets) + self.index_bias  # 对索引排序，并加上偏移量

        """视频帧数足够 - (随机) 采样帧 (等间隔划分视频帧索引 + 随机偏移采样)"""
        offsets = list() # offsets 列表中存放采样到的帧索引
        # 计算每个片段的起始位置
        ticks = [i * record.num_frames // self.num_segments  # 按照段数，等间隔划分视频帧索引
                for i in range(self.num_segments + 1)]
        # 遍历每个片段，随机采样帧
        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i] # 片段长度
            tick = ticks[i]  # 起始点
            # 如果当前段的帧数大于等于所需采样帧数，则在可选范围内随机选择一个起始点（原起始点 + 随机偏移量）
            if tick_len >= self.seg_length: 
                tick += randint(tick_len-self.seg_length+1)  # 随机偏移量
            offsets.extend([j for j in range(tick, tick+self.seg_length)]) # offsets 列表中存放采样到的帧索引
        return np.array(offsets) + self.index_bias # 加上索引偏移量

    # 验证模式 - 均匀取样
    def _get_val_indices(self, record):
        """用于在验证 (Val) 或测试阶段均匀采样视频帧，保证测试数据的稳定性，使不同批次的输入一致。"""
        
        """单片段情况 - 直接返回视频的 中间帧索引"""
        if self.num_segments == 1:
            return np.array([record.num_frames//2], dtype=np.int) + self.index_bias

        """视频帧数小于所需采样数 - 补全帧数"""
        # 如果视频的总帧数小于等于所需的总帧数
        if record.num_frames <= self.total_length:
            if self.loop: # 循环采样模式 - 使用 np.mod() 进行循环填充，保证索引范围合法。
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias  # 取模循环采样帧索引
            else: # 非循环模式 - 按等间隔填充索引，不引入随机性。
                return np.array([i * record.num_frames // self.total_length
                                for i in range(self.total_length)], 
                                dtype=np.int) + self.index_bias
        
        """视频帧数足够 - 均匀采样帧"""
        # 计算偏移量，使得采样的片段尽可能处于各片段的中间位置
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        # 按照固定的间隔从视频中选取帧索引
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments) # 遍历所有片段
                         for j in range(self.seg_length) # 在每个片段中取 `seg_length` 个帧
                        ], dtype=np.int) + self.index_bias # 返回最终的索引并加上索引偏移量

    # 返回相应索引对应的数据 (预处理后的图像数据 及 对应的标签)
    def __getitem__(self, index):
        """
        根据索引 `index` 获取对应的视频样本，并返回处理后的 帧数据 及 标签。

        参数：
        - index: 数据集中的索引，用于获取 `self.video_list` 中的相应视频信息。

        返回：
        - 经过 `_get` 处理后的图像数据（即采样并预处理的视频帧）。
        - 该视频样本的类别标签。
        """
        record = self.video_list[index]  # 获取索引对应的视频记录对象（包含视频路径、标签等信息）
        
        # 根据是否 `random_shift` 选择采样策略：
        # - 训练阶段 (random_shift=True) -> 调用 `_sample_indices` 进行随机采样
        # - 验证/测试阶段 (random_shift=False) -> 调用 `_get_val_indices` 进行均匀采样
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        
        # 返回 变换处理后的视频帧图像的列表 和 标签
        return self._get(record, segment_indices)

    def _get(self, record, indices):
        """
        根据提供的索引列表 `indices` 加载视频帧，并进行预处理。

        参数：
        - record: 存储视频信息的对象，包含视频的路径 (path)、帧数 (num_frames) 和 标签 (label)。
        - indices: 需要采样的帧索引列表。

        返回：
        - process_data: 经过变换（`self.transform`）后的图像数据。
        - record.label: 该视频样本的类别标签。
        """
        images = list()   # 用于存储加载的所有帧图像
        for _, seg_ind in enumerate(indices): # 遍历所有采样到的帧索引
            p = int(seg_ind)  # 将索引转换为整数（确保索引为有效帧索引）
            try:
                seg_imgs = self._load_image(record.path, p)  # 读取索引对应的视频帧
            except OSError:  # 无法读取图像
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs) # 将加载的帧图像添加到 `images` 列表
        process_data = self.transform(images) # 对所有图像进行变换（如归一化、数据增强等）
        return process_data, record.label  # 返回 预处理后的图像数据 及 对应的标签
    
    # 对 img_group 中的图片进行 transform 转换
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    # 返回数据集中有多少个视频
    def __len__(self):
        return len(self.video_list)