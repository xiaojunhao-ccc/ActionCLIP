import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
}

# 预训练 clip 模型下载函数
def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):  # ~ 代表当前用户的 Home 目录 (win: C:\Users\xiaoj linux: /root)
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

# 数据转换处理
def _transform(n_px):
    return Compose([
        # 将图像调整到 n_px 大小，并使用 双三次（BICUBIC）插值进行缩放
        Resize(n_px, interpolation=Image.BICUBIC), # 比 BILINEAR（双线性插值）更清晰锐利，减少模糊效果，比 NEAREST（最近邻插值）平滑，不会产生马赛克块状 
        CenterCrop(n_px), # Resize(n_px) + CenterCrop(n_px) 先等比例缩放，再中心裁剪，避免变形
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# 返回可用的 CLIP 模型名称
def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

# 加载 clip 模型
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True, tsm=False, joint=False, T=8, dropout=0., emb_dropout=0.,pretrain=True):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 从权重文件中载入模型
    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval() # 尝试加载 JIT 编译后的模型 | JIT 会优化计算图，减少 Python 解释开销，提高运行速度。
        state_dict = None
    except RuntimeError:  # 如果加载 JIT 模型失败，说明可能是一个普通的 state_dict 而不是 JIT 存档
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False  # 关闭 JIT 选项，改为加载普通模型权重
        state_dict = torch.load(model_path, map_location="cpu") # 加载普通的 state_dict（即模型参数）

    # 如果 JIT 未启用（即模型是普通的 PyTorch 模型）
    if not jit: 
        model = build_model(state_dict or model.state_dict(),  # 通过 state_dict 或从 JIT 模型提取的 state_dict 构建模型
                            joint=joint,  # 是否进行联合训练
                            tsm=tsm,  # 是否启用 Temporal Shift Module（时间偏移模块）
                            T=T,  # 视频片段数 (时间步)
                            dropout=dropout,  # Transformer 中的 Dropout 概率，默认为 None。
                            emb_dropout=emb_dropout,  # Embedding 层的 Dropout 概率
                            pretrain=pretrain  # 预训练权重的来源
                        ).to(device)  # 将模型移动到指定设备
        # 如果设备是 CPU，则强制转换模型为 float 类型（避免可能的精度问题）
        if str(device) == "cpu":
            model.float()
        # 返回模型以及模型的 state_dict（用于后续可能的保存或调试）
        return model, model.state_dict()
    
    # 如果 JIT 启用（即模型是 JIT 模型） ---- 下面的代码用于修正 JIT 模型的设备映射 ----
    else:
        # 创建一个“设备占位符”张量，并通过 JIT 追踪，目的是在 JIT 计算图中找到设备相关的节点
        device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
        # 在 JIT 计算图中查找所有 prim::Constant 类型的节点，并找到包含 "Device" 的最后一个节点
        device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

        # 遍历 JIT 计算图，将所有涉及设备的常量节点替换为目标设备。
        def patch_device(module):
            """
            遍历 JIT 计算图，将所有涉及设备的常量节点替换为目标设备。
            这样可以确保模型在不同设备（CPU/GPU）上运行时不会出错。
            """
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                        node.copyAttributes(device_node)

        model.apply(patch_device)

        if str(device) == "cpu":
            float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
            float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
            float_node = float_input.node()

            def patch_float(module):
                graphs = [module.graph] if hasattr(module, "graph") else []
                if hasattr(module, "forward1"):
                    graphs.append(module.forward1.graph)

                for graph in graphs:
                    for node in graph.findAllNodes("aten::to"):
                        inputs = list(node.inputs())
                        for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                            if inputs[i].node()["value"] == 5:
                                inputs[i].node().copyAttributes(float_node)

            model.apply(patch_float)
            patch_float(model.encode_image)
            patch_float(model.encode_text)

            model.float()

        return model, _transform(model.input_resolution.item())

# 对输入的字符串或字符串列表进行分词（tokenize），并返回其 token 表示。
def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    对输入的字符串或字符串列表进行分词（tokenize），并返回其 token 表示。

    参数：
    ----------
    texts : Union[str, List[str]]
        需要进行分词的文本，可以是单个字符串或字符串列表。

    context_length : int
        设定的上下文长度（context length），所有 CLIP 模型都使用 77 作为默认上下文长度。

    返回：
    -------
    torch.LongTensor
        一个二维张量，包含文本的 token 结果，形状为 (文本数量，context_length)。
    """
    # 如果输入是单个字符串，则转换为列表，确保后续处理统一
    if isinstance(texts, str):
        texts = [texts]
    
    # 获取特殊 token：<|startoftext|>（SOT，文本起始标记） <|endoftext|>（EOT，文本结束标记）
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # 对每个文本进行编码，并添加起始和结束标记 | 文本 "Hello world" -> [sot_token, tokenized("Hello"), tokenized("world"), eot_token]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # 创建一个形状为 (文本数量，context_length) 的张量，初始值为 0（填充用）
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    # 遍历所有 token 序列，将其填充到 `result` 张量中
    for i, tokens in enumerate(all_tokens):
        # 如果 token 数量超过 context_length，则抛出错误
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        # 将 tokens 填充到 result[i]，超出部分自动填充 0
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
