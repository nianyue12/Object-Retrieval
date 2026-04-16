from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cocoop_prompt_learner import ConditionalPromptLearner
from models.prompt_learner import PromptLearner, TextEncoder
from utils.clip_utils import load_clip_model


# 默认文本模板：
# 对同一类别使用多种自然语言描述，再把文本特征取平均，
# 通常会比只用一条模板更稳。
PROMPT_TEMPLATES = [
    "a 3D model of a {}",
    "a {} 3D object",
    "a CAD model of a {}",
    "a rendered view of a {}",
    "a depth rendering of a {}",
]


def _encode_fixed_text_prototypes(
    class_names: Iterable[str],
    clip_module,
    model,
    device: torch.device,
    templates: List[str],
) -> np.ndarray:
    """
    功能：用固定 prompt 模板为每个类别生成文本原型。

    用途：
        当没有训练好的 prompt checkpoint 时，
        直接用手写模板把类别名映射到 CLIP 文本特征空间。

    参数：
        class_names: 类别名列表
        clip_module: CLIP 的 tokenize 模块
        model: CLIP 模型
        device: 推理设备
        templates: 文本模板列表

    返回：
        protos: shape = (num_classes, feat_dim) 的归一化文本原型矩阵
    """
    # 先转成 list，避免传入的是只能遍历一次的对象
    class_names = list(class_names)
    protos = []

    with torch.no_grad():
        for cls in class_names:
            # 把类别名中的下划线替换为空格，让文本更自然
            name = cls.replace("_", " ")

            # 为当前类别构造多条 prompt
            prompts = [template.format(name) for template in templates]
            tokens = clip_module.tokenize(prompts).to(device)

            # 编码文本，得到每条 prompt 的文本特征
            text_feat = model.encode_text(tokens)

            # 先逐条归一化，方便后续做平均
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # 多条 prompt 的特征取平均，作为该类别的 prototype
            proto = text_feat.mean(dim=0)

            # 再归一化一次，保证类别原型也在单位球面上
            proto = proto / proto.norm()
            protos.append(proto.cpu().numpy())

    # 堆叠成二维矩阵，并再做一次 numpy 侧归一化
    protos = np.stack(protos).astype(np.float32)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    return protos


def _safe_torch_load(path: str, device):
    """
    功能：兼容不同 PyTorch 版本加载 checkpoint。
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_prompt_checkpoint(path: str, device) -> dict:
    """
    功能：加载 prompt checkpoint，并做基础合法性检查。

    检查内容：
        1. prompt_mode 必须是 'coop' 或 'cocoop'
        2. checkpoint 中必须包含 'ctx'
    """
    checkpoint = _safe_torch_load(path, device)
    prompt_mode = checkpoint.get("prompt_mode")
    if prompt_mode not in {"coop", "cocoop"}:
        raise ValueError(f"Unsupported prompt checkpoint mode: {prompt_mode}")
    if "ctx" not in checkpoint:
        raise KeyError("Prompt checkpoint is missing 'ctx'.")
    return checkpoint


def _encode_learned_prompt_prototypes(
    class_names: Iterable[str],
    model,
    device: torch.device,
    checkpoint: dict,
) -> np.ndarray:
    """
    功能：使用 CoOp 学到的上下文，为每个类别生成文本原型。

    用途：
        和固定模板不同，这里不是手写 prompt，
        而是使用训练得到的上下文 token 来构造类别文本表示。
    """
    # 根据 checkpoint 配置构造 PromptLearner
    prompt_learner = PromptLearner(
        class_names=class_names,
        clip_model=model,
        n_ctx=int(checkpoint["n_ctx"]),
        ctx_init=checkpoint.get("ctx_init", ""),
        class_token_position=checkpoint.get("class_token_position", "end"),
    ).to(device)

    # 恢复训练得到的上下文参数
    prompt_learner.load_context(checkpoint["ctx"])

    # TextEncoder 负责把 prompt 编码成文本特征
    text_encoder = TextEncoder(model).to(device)

    with torch.no_grad():
        # 生成 learned prompts
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts

        # 编码并归一化
        text_feat = text_encoder(prompts, tokenized_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    return text_feat.cpu().numpy().astype(np.float32)


def build_text_prototypes(
    class_names: Iterable[str],
    clip_model: str,
    device: torch.device,
    templates: Optional[List[str]] = None,
    prompt_checkpoint: Optional[str] = None,
) -> np.ndarray:
    """
    功能：构建一组类别对应的文本原型。

    两种模式：
        1. 不传 prompt_checkpoint：使用固定模板
        2. 传入 CoOp checkpoint：使用学习型 prompt
    """
    templates = templates or PROMPT_TEMPLATES

    # 加载 CLIP 模型
    clip_module, model, _ = load_clip_model(
        clip_model,
        device=device,
        force_float=True,
    )
    model.eval()

    if prompt_checkpoint:
        checkpoint = load_prompt_checkpoint(prompt_checkpoint, device=device)

        # 这里只支持 CoOp，不支持 CoCoOp
        if checkpoint.get("prompt_mode") != "coop":
            raise ValueError(
                f"build_text_prototypes only supports CoOp checkpoints, "
                f"but got {checkpoint.get('prompt_mode')}."
            )

        # 检查 checkpoint 对应的 CLIP 型号是否一致
        checkpoint_model = checkpoint.get("clip_model")
        if checkpoint_model and checkpoint_model != clip_model:
            raise ValueError(
                f"Prompt checkpoint expects clip_model={checkpoint_model}, "
                f"but received {clip_model}."
            )
        return _encode_learned_prompt_prototypes(
            class_names,
            model,
            device,
            checkpoint,
        )

    # 默认走固定模板路线
    return _encode_fixed_text_prototypes(
        class_names,
        clip_module,
        model,
        device,
        templates,
    )


def build_text_prototypes_for_sets(
    class_name_sets: Sequence[Iterable[str]],
    clip_model: str,
    device: torch.device,
    templates: Optional[List[str]] = None,
    prompt_checkpoint: Optional[str] = None,
) -> List[np.ndarray]:
    """
    功能：为多组类别集合分别构建文本原型。

    用途：
        当不同任务/数据划分有不同类别表时，
        可以一次性为每组类别生成对应的 prototype。
    """
    templates = templates or PROMPT_TEMPLATES

    # 只加载一次 CLIP，减少重复开销
    clip_module, model, _ = load_clip_model(
        clip_model,
        device=device,
        force_float=True,
    )
    model.eval()

    if prompt_checkpoint:
        checkpoint = load_prompt_checkpoint(prompt_checkpoint, device=device)

        if checkpoint.get("prompt_mode") != "coop":
            raise ValueError(
                f"build_text_prototypes_for_sets only supports CoOp checkpoints, "
                f"but got {checkpoint.get('prompt_mode')}."
            )
        checkpoint_model = checkpoint.get("clip_model")
        if checkpoint_model and checkpoint_model != clip_model:
            raise ValueError(
                f"Prompt checkpoint expects clip_model={checkpoint_model}, "
                f"but received {clip_model}."
            )

        # 对每组类别分别生成 learned prototypes
        return [
            _encode_learned_prompt_prototypes(class_names, model, device, checkpoint)
            for class_names in class_name_sets
        ]

    # 否则对每组类别使用固定模板
    return [
        _encode_fixed_text_prototypes(
            class_names,
            clip_module,
            model,
            device,
            templates,
        )
        for class_names in class_name_sets
    ]


def _entropy_confidence_torch(probs: torch.Tensor) -> torch.Tensor:
    """
    功能：根据类别概率分布的熵计算置信度。

    说明：
        分布越尖锐，说明模型越确定，置信度越高；
        分布越平坦，说明模型越不确定，置信度越低。
    """
    # 只有一个类别时，不存在不确定性
    if probs.shape[1] <= 1:
        return torch.ones(probs.shape[0], dtype=probs.dtype, device=probs.device)

    # 计算熵，再转成 [0, 1] 附近的置信度
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)
    return 1.0 - entropy / np.log(probs.shape[1])


def load_cocoop_prompt_components(
    class_names: Iterable[str],
    clip_model: str,
    device: torch.device,
    prompt_checkpoint: str,
) -> Tuple[torch.nn.Module, ConditionalPromptLearner, TextEncoder, dict]:
    """
    功能：加载 CoCoOp 推理所需的全部组件。

    返回：
        model: CLIP 模型
        prompt_learner: 条件式 PromptLearner
        text_encoder: 文本编码器
        checkpoint: 原始 checkpoint
    """
    checkpoint = load_prompt_checkpoint(prompt_checkpoint, device=device)
    if checkpoint.get("prompt_mode") != "cocoop":
        raise ValueError(
            f"Expected a CoCoOp checkpoint, but got {checkpoint.get('prompt_mode')}."
        )

    checkpoint_model = checkpoint.get("clip_model")
    if checkpoint_model and checkpoint_model != clip_model:
        raise ValueError(
            f"Prompt checkpoint expects clip_model={checkpoint_model}, "
            f"but received {clip_model}."
        )

    # 加载 CLIP 主模型
    _, model, _ = load_clip_model(
        clip_model,
        device=device,
        force_float=True,
    )
    model.eval()

    # 推理阶段冻结 CLIP 参数
    for param in model.parameters():
        param.requires_grad_(False)

    # 构造条件式 prompt learner
    prompt_learner = ConditionalPromptLearner(
        class_names=class_names,
        clip_model=model,
        image_feature_dim=int(checkpoint.get("feature_dim", model.text_projection.shape[1])),
        n_ctx=int(checkpoint["n_ctx"]),
        ctx_init=checkpoint.get("ctx_init", ""),
        meta_hidden_dim=int(checkpoint.get("meta_hidden_dim", 64)),
        class_token_position=checkpoint.get("class_token_position", "end"),
    ).to(device)

    # 恢复训练好的上下文和 meta-net 参数
    prompt_learner.load_context(checkpoint["ctx"])
    prompt_learner.meta_net.load_state_dict(checkpoint["meta_net_state_dict"])
    prompt_learner.eval()

    text_encoder = TextEncoder(model).to(device)
    text_encoder.eval()
    return model, prompt_learner, text_encoder, checkpoint


def build_conditional_semantic_branch(
    feats: np.ndarray,
    model,
    prompt_learner: ConditionalPromptLearner,
    text_encoder: TextEncoder,
    temperature: float,
    batch_size: int,
    prompt_chunk_size: int,
    desc: str,
):
    """
    功能：基于 CoCoOp 的条件文本特征，构建 semantic branch 输出。

    输出包含：
        logits: 每个样本对每个类别的打分
        probs: 类别分布
        embed: 语义嵌入
        confidence: 置信度
    """
    feats = np.asarray(feats, dtype=np.float32)
    num_items = feats.shape[0]
    num_classes = prompt_learner.n_cls

    # 预分配输出数组，避免循环里频繁申请内存
    logits_out = np.empty((num_items, num_classes), dtype=np.float32)
    probs_out = np.empty((num_items, num_classes), dtype=np.float32)
    embed_out = np.empty((num_items, feats.shape[1]), dtype=np.float32)
    confidence_out = np.empty((num_items,), dtype=np.float32)

    device = prompt_learner.ctx.device

    # 避免 temperature 太小导致数值不稳定
    temperature = float(max(temperature, 1e-6))

    with torch.no_grad():
        # 分 batch 处理，控制显存占用
        for start in tqdm(
            range(0, num_items, batch_size),
            desc=desc,
        ):
            end = min(start + batch_size, num_items)

            # 当前 batch 的输入特征，并做 L2 归一化
            batch_feats = torch.from_numpy(feats[start:end]).to(device)
            batch_feats = F.normalize(batch_feats, dim=-1)

            # 根据当前样本特征动态生成类别文本特征
            text_features = prompt_learner.get_text_features(
                batch_feats,
                text_encoder,
                prompt_chunk_size=prompt_chunk_size,
            )

            # 计算样本与各类别文本特征的相似度
            logits = torch.einsum("bd,bcd->bc", batch_feats, text_features) / temperature

            # softmax 得到类别分布
            raw_probs = F.softmax(logits, dim=-1)

            # 再做一次 normalize，作为输出用的 probs 表示
            probs = F.normalize(raw_probs, dim=-1)

            # 用类别概率对文本特征加权，得到语义嵌入
            semantic_embed = torch.sum(raw_probs.unsqueeze(-1) * text_features, dim=1)
            semantic_embed = F.normalize(semantic_embed, dim=-1)

            # 计算每个样本的预测置信度
            confidence = _entropy_confidence_torch(raw_probs)

            # 写回当前 batch 的结果
            logits_out[start:end] = logits.cpu().numpy().astype(np.float32)
            probs_out[start:end] = probs.cpu().numpy().astype(np.float32)
            embed_out[start:end] = semantic_embed.cpu().numpy().astype(np.float32)
            confidence_out[start:end] = confidence.cpu().numpy().astype(np.float32)

    return {
        "logits": logits_out,
        "probs": probs_out,
        "embed": embed_out,
        "confidence": confidence_out,
    }
