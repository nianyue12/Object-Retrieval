from typing import Iterable, List, Optional, Sequence

import clip
import numpy as np
import torch


PROMPT_TEMPLATES = [
    "a 3D model of a {}",
    "a {} 3D object",
    "a CAD model of a {}",
    "a rendered view of a {}",
    "a depth rendering of a {}",
]


def _encode_text_prototypes(
    class_names: Iterable[str],
    model,
    device: torch.device,
    templates: List[str],
) -> np.ndarray:
    protos = []
    with torch.no_grad():
        for cls in class_names:
            name = cls.replace("_", " ")
            prompts = [template.format(name) for template in templates]
            tokens = clip.tokenize(prompts).to(device)

            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            proto = text_feat.mean(dim=0)
            proto = proto / proto.norm()
            protos.append(proto.cpu().numpy())

    protos = np.stack(protos).astype(np.float32)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    return protos


def build_text_prototypes(
    class_names: Iterable[str],
    clip_model: str,
    device: torch.device,
    templates: Optional[List[str]] = None,
) -> np.ndarray:
    templates = templates or PROMPT_TEMPLATES

    model, _ = clip.load(clip_model, device=device)
    model.eval()
    return _encode_text_prototypes(class_names, model, device, templates)


def build_text_prototypes_for_sets(
    class_name_sets: Sequence[Iterable[str]],
    clip_model: str,
    device: torch.device,
    templates: Optional[List[str]] = None,
) -> List[np.ndarray]:
    templates = templates or PROMPT_TEMPLATES

    model, _ = clip.load(clip_model, device=device)
    model.eval()

    return [
        _encode_text_prototypes(class_names, model, device, templates)
        for class_names in class_name_sets
    ]
