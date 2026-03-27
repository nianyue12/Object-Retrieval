from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from utils.clip_utils import get_clip_module


class MetaNet(nn.Module):
    def __init__(self, input_dim: int, ctx_dim: int, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ctx_dim),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.net(image_features)


class ConditionalPromptLearner(nn.Module):
    def __init__(
        self,
        class_names: Iterable[str],
        clip_model,
        image_feature_dim: int,
        n_ctx: int = 8,
        ctx_init: str = "",
        meta_hidden_dim: int = 64,
        class_token_position: str = "end",
    ):
        super().__init__()
        if n_ctx <= 0:
            raise ValueError("n_ctx must be positive.")
        if class_token_position != "end":
            raise ValueError(
                f"Unsupported class_token_position: {class_token_position}"
            )

        self.class_names = [name.replace("_", " ") for name in class_names]
        self.n_cls = len(self.class_names)
        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        self.class_token_position = class_token_position
        self.dtype = clip_model.ln_final.weight.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.device = clip_model.token_embedding.weight.device
        self.image_feature_dim = image_feature_dim
        self.meta_hidden_dim = meta_hidden_dim

        clip = get_clip_module()
        ctx = self._init_context(clip, clip_model)
        self.ctx = nn.Parameter(ctx)
        self.meta_net = MetaNet(
            input_dim=image_feature_dim,
            ctx_dim=self.ctx_dim,
            hidden_dim=meta_hidden_dim,
        )

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompt_texts = [f"{prompt_prefix} {name}." for name in self.class_names]
        tokenized_prompts = clip.tokenize(prompt_texts)
        tokenized_prompts = tokenized_prompts.to(self.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

    def _init_context(self, clip, clip_model):
        if not self.ctx_init.strip():
            ctx = torch.empty(
                self.n_ctx,
                self.ctx_dim,
                dtype=self.dtype,
                device=self.device,
            )
            nn.init.normal_(ctx, std=0.02)
            return ctx

        tokenized_ctx = clip.tokenize(self.ctx_init).to(self.device)
        eot_index = int(tokenized_ctx[0].argmax().item())
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_ctx).type(self.dtype)
        init_ctx = embedding[0, 1:eot_index, :]

        if init_ctx.shape[0] > self.n_ctx:
            raise ValueError(
                f"ctx_init token length {init_ctx.shape[0]} exceeds n_ctx={self.n_ctx}."
            )

        if init_ctx.shape[0] < self.n_ctx:
            pad = torch.empty(
                self.n_ctx - init_ctx.shape[0],
                self.ctx_dim,
                dtype=self.dtype,
                device=init_ctx.device,
            )
            nn.init.normal_(pad, std=0.02)
            init_ctx = torch.cat([init_ctx, pad], dim=0)

        return init_ctx

    def get_context(self) -> torch.Tensor:
        return self.ctx.detach().clone()

    def load_context(self, ctx_tensor: torch.Tensor) -> None:
        if tuple(ctx_tensor.shape) != tuple(self.ctx.shape):
            raise ValueError(
                f"Prompt context shape mismatch: expected {tuple(self.ctx.shape)}, "
                f"got {tuple(ctx_tensor.shape)}."
            )
        with torch.no_grad():
            self.ctx.copy_(
                ctx_tensor.to(device=self.ctx.device, dtype=self.ctx.dtype)
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = image_features.to(device=self.device, dtype=self.dtype)
        conditional_bias = self.meta_net(image_features)
        conditional_ctx = self.ctx.unsqueeze(0) + conditional_bias.unsqueeze(1)
        conditional_ctx = conditional_ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        batch_size = image_features.shape[0]
        token_prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        token_suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([token_prefix, conditional_ctx, token_suffix], dim=2)

    def get_text_features(
        self,
        image_features,
        text_encoder,
        prompt_chunk_size: int = 0,
    ) -> torch.Tensor:
        prompts = self(image_features)
        batch_size, n_cls, prompt_len, ctx_dim = prompts.shape
        prompts = prompts.reshape(batch_size * n_cls, prompt_len, ctx_dim)
        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        tokenized_prompts = tokenized_prompts.reshape(batch_size * n_cls, -1)

        if prompt_chunk_size and prompt_chunk_size > 0:
            chunks = []
            total_prompts = prompts.shape[0]
            for start in range(0, total_prompts, prompt_chunk_size):
                end = min(start + prompt_chunk_size, total_prompts)
                chunks.append(
                    text_encoder(prompts[start:end], tokenized_prompts[start:end])
                )
            text_features = torch.cat(chunks, dim=0)
        else:
            text_features = text_encoder(prompts, tokenized_prompts)

        text_features = text_features.reshape(batch_size, n_cls, -1)
        return F.normalize(text_features, dim=-1)
