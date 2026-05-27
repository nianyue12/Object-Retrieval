import math
from typing import List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """
    功能：给线性层包一层轻量级 LoRA 适配器。

    说明：
        原始权重冻结，只训练低秩分解出的增量参数。
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if not hasattr(base_layer, "weight"):
            raise TypeError("base_layer must expose a weight tensor.")

        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = int(base_layer.in_features)
        out_features = int(base_layer.out_features)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        # LoRA 不直接训练完整权重 W，而是训练低秩增量 B @ A。
        # A 负责把输入降到 rank 维，B 再把 rank 维映射回输出维度。
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 基础线性层被冻结，训练时只更新 lora_A / lora_B 两组小参数。
        for param in self.base.parameters():
            param.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def merged_weight(self) -> torch.Tensor:
        """返回基础权重和 LoRA 增量合并后的视图。"""
        # 低秩增量的形状会恢复成和 base.weight 一样，才能与原权重相加。
        delta = self.lora_B @ self.lora_A
        delta = delta.to(
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        return self.base.weight + delta * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        # MultiheadAttention 会直接访问 out_proj.weight，因此这里暴露合并后的权重视图。
        return self.merged_weight()

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向输出 = 原层输出 + LoRA 增量。"""
        # base_out 是冻结 CLIP 的原始输出；lora_update 是当前任务学到的补偿项。
        base_out = self.base(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_update = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_out + lora_update.to(dtype=base_out.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:.4f}"
        )


def _get_child_module(root: nn.Module, dotted_name: str):
    """根据点分路径找到父模块和子模块名。"""
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _extract_visual_block_index(module_name: str) -> Optional[int]:
    """从 CLIP 视觉分支模块名中解析 transformer block 编号。"""
    prefix = "visual.transformer.resblocks."
    if not module_name.startswith(prefix):
        return None

    suffix = module_name[len(prefix) :]
    block_name = suffix.split(".", 1)[0]
    if not block_name.isdigit():
        return None
    return int(block_name)


def _should_replace_module(
    module_name: str,
    module: nn.Module,
    block_indices: Optional[Sequence[int]],
    module_suffixes: Sequence[str],
) -> bool:
    """
    功能：判断某个模块是否应该被 LoRA 替换。
    """
    if not isinstance(module, nn.Linear):
        return False

    block_index = _extract_visual_block_index(module_name)
    if block_index is None:
        return False

    if block_indices is not None and block_index not in set(block_indices):
        return False

    # 只替换指定后缀的线性层，例如 MLP 投影层或注意力输出层。
    return any(module_name.endswith(suffix) for suffix in module_suffixes)


def apply_lora_to_clip(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    block_indices: Optional[Sequence[int]] = None,
    module_suffixes: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    功能：把 LoRA 注入到 CLIP 视觉分支指定模块上。

    返回：
        被替换掉的模块名列表
    """
    if module_suffixes is None:
        module_suffixes = ("mlp.c_fc", "mlp.c_proj", "attn.out_proj")

    named_modules = list(model.named_modules())
    replaced = []

    for module_name, module in named_modules:
        if not _should_replace_module(
            module_name,
            module,
            block_indices=block_indices,
            module_suffixes=module_suffixes,
        ):
            continue

        # 找到目标 Linear 所属的父模块，然后用 LoRALinear 原地包住它。
        parent, child_name = _get_child_module(model, module_name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced.append(module_name)

    if not replaced:
        raise ValueError(
            "No modules were replaced by LoRA. Check block_indices/module_suffixes."
        )

    return replaced


def mark_only_lora_trainable(model: nn.Module) -> List[str]:
    """
    功能：冻结全部参数，只保留 LoRA 参数可训练。
    """
    # 先全局冻结，避免误训练 CLIP 原始参数。
    for param in model.parameters():
        param.requires_grad_(False)

    trainable = []
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        # 再只打开每个 LoRALinear 里的 A / B 矩阵。
        module.lora_A.requires_grad_(True)
        module.lora_B.requires_grad_(True)
        trainable.append(f"{module_name}.lora_A")
        trainable.append(f"{module_name}.lora_B")

    if not trainable:
        raise ValueError("No LoRA parameters found to mark trainable.")

    return trainable


def extract_lora_state_dict(model: nn.Module) -> dict:
    """导出当前模型里的 LoRA 参数。"""
    state = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        state[f"{module_name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{module_name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict, strict: bool = True) -> dict:
    """
    功能：把保存的 LoRA 参数加载回模型。

    返回：
        缺失项和多余项信息
    """
    module_map = dict(model.named_modules())
    loaded = set()
    unexpected = []

    for key, value in state_dict.items():
        if not (key.endswith(".lora_A") or key.endswith(".lora_B")):
            unexpected.append(key)
            continue

        module_name, param_name = key.rsplit(".", 1)
        module = module_map.get(module_name)
        if not isinstance(module, LoRALinear):
            unexpected.append(key)
            continue

        target = getattr(module, param_name)
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(
                f"LoRA tensor shape mismatch for {key}: "
                f"expected {tuple(target.shape)}, got {tuple(value.shape)}."
            )

        # checkpoint 中只保存 LoRA 增量参数，这里把它们拷回对应模块。
        with torch.no_grad():
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        loaded.add(key)

    missing = []
    if strict:
        for module_name, module in module_map.items():
            if not isinstance(module, LoRALinear):
                continue
            for param_name in ("lora_A", "lora_B"):
                key = f"{module_name}.{param_name}"
                if key not in loaded:
                    missing.append(key)
        if missing or unexpected:
            raise KeyError(
                f"LoRA state dict mismatch. Missing: {missing}; Unexpected: {unexpected}"
            )

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """统计当前可训练参数量。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """统计模型总参数量。"""
    return sum(param.numel() for param in model.parameters())


def parse_int_list(raw_value: str) -> List[int]:
    """把逗号分隔的字符串解析成整数列表。"""
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_str_list(raw_value: str) -> List[str]:
    """把逗号分隔的字符串解析成字符串列表。"""
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values
