import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

# 测试
def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug):
    """
    在验证集上评估模型性能，计算 Top-1 和 Top-5 准确率，并记录到 WandB。

    参数：
    - epoch: 当前训练的轮次。
    - val_loader: 验证集的数据加载器，提供输入图像和对应的类别 ID。
    - classes: 预先处理好的文本类别信息 (文本 prompt)。
    - device: 计算设备。
    - model: 训练好的模型。
    - fusion_model: 视频多帧特征池化模块。
    - config: 配置对象，包含数据和训练的超参数。
    - num_text_aug: 文本增强的次数 (用于处理多个文本描述)。

    返回：
    - top1: Top-1 准确率 (单个最高相似度的类别匹配准确率)。
    """
    # 设置模型为评估模式，防止 BatchNorm 和 Dropout 影响结果
    model.eval()
    fusion_model.eval()
    
    # 初始化计数器
    num = 0        # 统计样本总数
    corr_top_1 = 0     # 统计 Top-1 预测正确的样本数
    corr_top_5 = 0     # 统计 Top-5 预测正确的样本数

    with torch.no_grad():  # 关闭梯度计算，节省显存

        # 计算所有类别的文本特征
        text_inputs = classes.to(device) 
        text_features = model.encode_text(text_inputs)

        # 遍历验证集
        for _, (image, class_id) in enumerate(tqdm(val_loader)):

            # 真实类别 class_id 
            class_id = class_id.to(device)
            
            # 输入 image：(batch_size, num_segments * 3, H, W) -> (batch_size, num_segments, 3, H, W)
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            # 将所有帧展平为 (batch_size * num_segments, 3, H, W)
            image_input = image.to(device).view(-1, c, h, w)  
            
            # 图像编码器得到 当前图像特征
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化
            
            # 所有类别的文本特征
            text_features /= text_features.norm(dim=-1, keepdim=True)  # 归一化
            
            # 计算 当前图像特征 和 所有类别的文本特征 的 余弦相似度 (batch_size, num_classes)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1) # 计算 softmax 归一化，得到每个类别的置信度
            similarity = similarity.mean(dim=1, keepdim=False)  # 在多种文本 prompts 下的置信度均值
            
            # 计算 Top-1 和 Top-5 预测结果
            values_1, top_1 = similarity.topk(1, dim=-1)
            values_5, top_5 = similarity.topk(5, dim=-1)

            # 统计 Top-1, Top-5 预测正确的样本数
            for i in range(b):
                if top_1[i] == class_id[i]:
                    corr_top_1 += 1
                if class_id[i] in top_5[i]:
                    corr_top_5 += 1

            # 更新总样本数
            num += b
    
    # 计算最终的 Top-1 和 Top-5 准确率
    top1 = float(corr_top_1) / num * 100
    top5 = float(corr_top_5) / num * 100
    
    # 记录到 WandB 进行可视化
    wandb.log({"top1": top1})
    wandb.log({"top5": top5})

    # 打印当前 Epoch 的验证结果
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    
    return top1, top5

def main():
    # 初始化
    global args
    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    """解析用户输入参数 (实验配置，当前时间)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                         config['data']['dataset']))
    
    # 打印当前训练配置，并创建相关的文件夹，用于记录
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config) # 将 config 转换为 DotMap 对象，使其可以使用点符号 (config.key) 访问字典中的键值。

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)
    
    """数据增强策略"""
    transform_val = get_augmentation(False, config)
    # 打印数据增强策略配置
    print('val transforms: {}'.format(transform_val.transforms))

    """模型定义"""
    """加载 clip 预训练模型"""
    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    """视频多帧特征池化模块"""
    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    """文本 encoder"""
    model_text = TextCLIP(model)
    """图像 encoder"""
    model_image = ImageCLIP(model)

    # 以支持多 GPU 并行计算
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    
    # 浮点精度
    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    # 使用 wandb.watch 监视 model 和 fusion_model，以跟踪其梯度和参数变化，方便可视化和调试。
    wandb.watch(model)
    wandb.watch(fusion_model)

    """加载数据"""
    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    
    """开始测试"""
    # 从配置中读取 开始 epoch
    start_epoch = config.solver.start_epoch
    # 加载预训练模型
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
    
    # 生成文本 prompt (dance -> a photo of action {{dance}})
    classes, num_text_aug, text_dict = text_prompt(val_data)

    """开始测试"""
    top1, top5 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)
    print(f"测试结束，top1 结果为{top1}，top5 结果为{top5}")

if __name__ == '__main__':
    main()
