from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cocoop_prompt_learner import ConditionalPromptLearner
from models.prompt_learner import PromptLearner, TextEncoder
from utils.clip_utils import load_clip_model


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
    class_names = list(class_names)
    protos = []
    with torch.no_grad():
        for cls in class_names:
            name = cls.replace("_", " ")
            prompts = [template.format(name) for template in templates]
            tokens = clip_module.tokenize(prompts).to(device)

            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            proto = text_feat.mean(dim=0)
            proto = proto / proto.norm()
            protos.append(proto.cpu().numpy())

    protos = np.stack(protos).astype(np.float32)
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    return protos


def _safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_prompt_checkpoint(path: str, device) -> dict:
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
    prompt_learner = PromptLearner(
        class_names=class_names,
        clip_model=model,
        n_ctx=int(checkpoint["n_ctx"]),
        ctx_init=checkpoint.get("ctx_init", ""),
        class_token_position=checkpoint.get("class_token_position", "end"),
    ).to(device)
    prompt_learner.load_context(checkpoint["ctx"])
    text_encoder = TextEncoder(model).to(device)

    with torch.no_grad():
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts
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
    templates = templates or PROMPT_TEMPLATES

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
                f"build_text_prototypes only supports CoOp checkpoints, "
                f"but got {checkpoint.get('prompt_mode')}."
            )
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
    templates = templates or PROMPT_TEMPLATES

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
        return [
            _encode_learned_prompt_prototypes(class_names, model, device, checkpoint)
            for class_names in class_name_sets
        ]

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
    if probs.shape[1] <= 1:
        return torch.ones(probs.shape[0], dtype=probs.dtype, device=probs.device)

    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)
    return 1.0 - entropy / np.log(probs.shape[1])


def load_cocoop_prompt_components(
    class_names: Iterable[str],
    clip_model: str,
    device: torch.device,
    prompt_checkpoint: str,
) -> Tuple[torch.nn.Module, ConditionalPromptLearner, TextEncoder, dict]:
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

    _, model, _ = load_clip_model(
        clip_model,
        device=device,
        force_float=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    prompt_learner = ConditionalPromptLearner(
        class_names=class_names,
        clip_model=model,
        image_feature_dim=int(checkpoint.get("feature_dim", model.text_projection.shape[1])),
        n_ctx=int(checkpoint["n_ctx"]),
        ctx_init=checkpoint.get("ctx_init", ""),
        meta_hidden_dim=int(checkpoint.get("meta_hidden_dim", 64)),
        class_token_position=checkpoint.get("class_token_position", "end"),
    ).to(device)
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
    feats = np.asarray(feats, dtype=np.float32)
    num_items = feats.shape[0]
    num_classes = prompt_learner.n_cls

    logits_out = np.empty((num_items, num_classes), dtype=np.float32)
    probs_out = np.empty((num_items, num_classes), dtype=np.float32)
    embed_out = np.empty((num_items, feats.shape[1]), dtype=np.float32)
    confidence_out = np.empty((num_items,), dtype=np.float32)

    device = prompt_learner.ctx.device
    temperature = float(max(temperature, 1e-6))

    with torch.no_grad():
        for start in tqdm(
            range(0, num_items, batch_size),
            desc=desc,
        ):
            end = min(start + batch_size, num_items)
            batch_feats = torch.from_numpy(feats[start:end]).to(device)
            batch_feats = F.normalize(batch_feats, dim=-1)

            text_features = prompt_learner.get_text_features(
                batch_feats,
                text_encoder,
                prompt_chunk_size=prompt_chunk_size,
            )
            logits = torch.einsum("bd,bcd->bc", batch_feats, text_features) / temperature
            raw_probs = F.softmax(logits, dim=-1)
            probs = F.normalize(raw_probs, dim=-1)
            semantic_embed = torch.sum(raw_probs.unsqueeze(-1) * text_features, dim=1)
            semantic_embed = F.normalize(semantic_embed, dim=-1)
            confidence = _entropy_confidence_torch(raw_probs)

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
