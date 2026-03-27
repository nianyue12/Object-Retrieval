import importlib
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_clip_module():
    """Load the vendored OpenAI CLIP package shipped with this repo."""
    clip_root = _project_root() / "CLIP"
    if clip_root.exists():
        clip_root_str = os.fspath(clip_root)
        if clip_root_str not in sys.path:
            sys.path.insert(0, clip_root_str)

    clip = importlib.import_module("clip")
    if not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
        raise ImportError("Imported 'clip' module does not expose OpenAI CLIP APIs.")
    return clip


def load_clip_model(model_name: str, device, force_float: bool = False):
    clip = get_clip_module()
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    if force_float:
        model.float()
    return clip, model, preprocess
