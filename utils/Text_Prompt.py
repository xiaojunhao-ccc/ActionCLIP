# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import clip

def text_prompt(data):
    # 定义一组文本 prompt 模板，这些模板用于构造描述动作（action）的自然语言句子
    text_aug = [
        f"a photo of action {{}}",  # "一张关于 {} 动作的照片"
        f"a picture of action {{}}",  # "一张关于 {} 动作的图片"
        f"Human action of {{}}",  # "{} 的人类动作"
        f"{{}}, an action",  # "{}，一个动作"
        f"{{}} this is an action",  # "{}，这是一个动作"
        f"{{}}, a video of action",  # "{}，一个关于动作的视频"
        f"Playing action of {{}}",  # "正在进行 {} 动作"
        f"{{}}",  # 直接使用动作名称
        f"Playing a kind of action, {{}}",  # "进行某种动作，{}"
        f"Doing a kind of action, {{}}",  # "做某种动作，{}"
        f"Look, the human is {{}}",  # "看，这个人正在 {}"
        f"Can you recognize the action of {{}}?",  # "你能认出 {} 这个动作吗？"
        f"Video classification of {{}}",  # "关于 {} 的视频分类"
        f"A video of {{}}",  # "一个关于 {} 的视频"
        f"The man is {{}}",  # "这个男人正在 {}"
        f"The woman is {{}}",  # "这个女人正在 {}"
    ]

    text_dict = {} # 初始化文本字典，用于存储不同的文本提示对应的 token 序列
    num_text_aug = len(text_aug) # 获取文本增强的数量，即 text_aug 中有多少种不同的文本模板

    # 遍历所有文本模板，并针对数据集中的每个类别生成 token
    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    # 将所有类别对应的文本 token 拼接成一个大张量
    classes = torch.cat([v for k, v in text_dict.items()])

    # 返回处理后的类别 token、文本增强的数量，以及文本字典
    return classes, num_text_aug, text_dict


"""
假设 data.classes = ["jumping", "running"]，则 text_prompt(data) 的输出示例为：

    num_text_aug = 16  # 16 个文本模板

    text_dict = {
        0: tensor([tokenized("a photo of action jumping"), tokenized("a photo of action running")]),
        1: tensor([tokenized("a picture of action jumping"), tokenized("a picture of action running")]),
        ...
        15: tensor([tokenized("The woman is jumping"), tokenized("The woman is running")])
    }

    classes = tensor([
        tokenized("a photo of action jumping"), tokenized("a photo of action running"),
        tokenized("a picture of action jumping"), tokenized("a picture of action running"),
        ...
        tokenized("The woman is jumping"), tokenized("The woman is running")
    ])  # 共 16×2=32 行

"""