import importlib
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent


def get_clip_module():
    """
    功能：加载仓库内自带的 OpenAI CLIP 包。

    说明：
        这里优先使用项目里 vendored 的 `CLIP/` 目录，
        保证训练和推理环境一致。
    """
    clip_root = _project_root() / "CLIP"
    if clip_root.exists():
        clip_root_str = os.fspath(clip_root)
        if clip_root_str not in sys.path:
            # 把本仓库的 CLIP 放到搜索路径最前面，避免误用系统里其他同名包。
            sys.path.insert(0, clip_root_str)

    clip = importlib.import_module("clip")
    # 这里做一次接口检查，提前暴露导入到错误 clip 包的问题。
    if not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
        raise ImportError("Imported 'clip' module does not expose OpenAI CLIP APIs.")
    return clip


def load_clip_model(model_name: str, device, force_float: bool = False):
    """
    功能：加载指定名称的 CLIP 模型与预处理流程。

    参数：
        model_name: CLIP 模型名
        device: 模型加载设备
        force_float: 是否将模型转换为 float32
    """
    clip = get_clip_module()
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    if force_float:
        # 一些 prompt/adapter 训练会显式使用 float32，避免半精度带来的数值差异。
        model.float()
    return clip, model, preprocess
