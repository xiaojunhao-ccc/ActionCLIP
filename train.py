import os
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
from utils.Text_Prompt import *
from utils.KLLoss import KLLoss
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.saving import  *

""" 文本编码器 和 图像编码器 均复用 Clip 原来的，
    只是在图像编码器上加了个 fusion_model(图像 prompt) """
class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def main():
    # 初始化
    global args, best_prec1
    global global_step
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    
    """解析用户输入参数（实验配置，当前时间）"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    wandb.init(
        project=config['network']['type'],
        name=f'{args.log_time}_{config['network']['type']}_{config['network']['arch']}_{config['data']['dataset']}'
    )
    config = DotMap(config) # 将 config 转换为 DotMap 对象，使其可以使用点符号 (config.key) 访问字典中的键值。 

    # 打印当前训练配置，并创建相关的文件夹，用于记录
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

    """数据增强策略"""
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)
    # 打印数据增强策略配置
    print('train transforms: {}'.format(transform_train.transforms))  
    print('val transforms: {}'.format(transform_val.transforms))

    """模型定义"""
    """clip 预训练模型"""
    model, clip_state_dict = clip.load(
        config.network.arch,  # 模型架构名称，例如 'ViT-B/16' 
        device=device,
        jit=False, # 禁用 TorchScript，即不使用 JIT 编译，确保模型可训练 - Must set jit=False for training  ViT-B/32 
        dropout=config.network.drop_out, # Transformer 中的 Dropout 概率，默认为 None。
        emb_dropout=config.network.emb_dropout, # Embedding 层的 Dropout 概率
        pretrain=config.network.init, # 预训练模型的路径或标识符
        tsm=config.network.tsm, # 是否使用时间偏移模块（Temporal Shift Module）
        T=config.data.num_segments,  # 视频片段数 (时间步数)
        joint = config.network.joint) # 是否使用时间信息编码，和位置编码进行联合训练
    """视频多帧特征池化模块"""  # (降维时间维度，以匹配文本特征维度) (Post-network prompt: MeanP, Conv1D, LSTM and Transf)
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
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)
    
    """损失函数"""
    loss_img = KLLoss()
    loss_txt = KLLoss()

    # 使用 wandb.watch 监视 model 和 fusion_model，以跟踪其梯度和参数变化，方便可视化和调试。
    wandb.watch(model)
    wandb.watch(fusion_model)

    """加载数据"""
    train_data = Action_DATASETS(config.data.train_list, config.data.label_list, num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
                       transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True, pin_memory=False, drop_last=True)
    val_data = Action_DATASETS(config.data.val_list,config.data.label_list, random_shift=False, num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                       transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False, pin_memory=False, drop_last=True)

    """开始训练"""
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
            print(("=> pretrain: no checkpoint found at '{}'".format(config.resume)))
    # 模型继续训练
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> resume: no checkpoint found at '{}'".format(config.pretrain)))
    
    # 根据配置文件，初始化优化器和学习率调度器
    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    # 生成文本 prompt，用于构造文本模态的训练数据 (dance -> a photo of action {{dance}})
    classes, num_text_aug, text_dict = text_prompt(train_data)

    # 如果设置了只进行评估，则直接在验证集上评估模型，并返回
    if config.solver.evaluate: 
        prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)
        return
    
    # 打印模型的所有参数以及它们是否可训练
    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    
    """开始训练循环"""
    best_prec1 = 0.0  # 初始化最佳准确率
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        # 遍历训练数据
        for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
            
            # 如果优化器类型不是 'monitor'，则按一定间隔更新学习率
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad() # 清空梯度

            # 调整 images 形状，使其符合 (batch, time, channels, height, width) 格式
            images = images.view((-1,config.data.num_segments,3) + images.size()[-2:])
            b,t,c,h,w = images.size()
            # 随机选择文本数据的增强版本
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id, text_id)]) # prompt 增强后的的文本标签
            # 将 images 和 texts 迁移到计算设备
            images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)

            """
            (model_image+fusion_model) 提取出 图像特征，
            (text_embedding) 提取出 文本特征，
            计算 图像特征 和 文本特征的 相似度分数，并与 GT 比较，计算损失
            """
            # 提取图像特征
            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b,t,-1)
            image_embedding = fusion_model(image_embedding)
            # 提取文本特征
            text_embedding = model_text(texts)

            # 如果固定文本编码器（不训练），则冻结梯度
            if config.network.fix_text:
                text_embedding.detach_()

            # 计算对比学习的 logit scale
            logit_scale = model.logit_scale.exp()
            # 计算图像和文本的对比学习相似度分数
            logits_per_image, logits_per_text = calc_similarity(image_embedding, text_embedding, logit_scale)
            # 读取 GT
            ground_truth = torch.tensor(gen_label(list_id), dtype=image_embedding.dtype, device=device)
            
            # 损失 = (图像->文本) 匹配损失 + (文本->图像) 匹配损失
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts)/2
            
            # 记录训练损失到 WandB（用于可视化）
            wandb.log({"train_total_loss": total_loss})
            wandb.log({"train_loss_imgs": loss_imgs})
            wandb.log({"train_loss_texts": loss_texts})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            
            # 反向传播
            total_loss.backward()

            # 根据设备类型进行优化器更新
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)  # 将模型转换为 FP32 精度，防止梯度计算因精度问题导致的错误
                optimizer.step()
                clip.model.convert_weights(model)  # 将模型恢复到为 FP16 精度

        # 每隔 `eval_freq` 轮进行一次 eval
        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug)

        # 记录当前准确率
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))
        
        # 保存当前训练的模型权重
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)
        epoch_saving(epoch, model, fusion_model, optimizer, filename)
        # 额外保存最佳模型
        if is_best:
            best_saving(working_dir, epoch, model, fusion_model, optimizer)

if __name__ == '__main__':
    main()
