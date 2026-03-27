from typing import Iterable

import torch
from torch import nn

from utils.clip_utils import get_clip_module


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.ln_final.weight.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        indices = torch.arange(x.shape[0], device=x.device)
        x = x[indices, tokenized_prompts.argmax(dim=-1)]
        return x @ self.text_projection


class PromptLearner(nn.Module):
    def __init__(
        self,
        class_names: Iterable[str],
        clip_model,
        n_ctx: int = 4,
        ctx_init: str = "",
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

        clip = get_clip_module()
        ctx = self._init_context(clip, clip_model)
        self.ctx = nn.Parameter(ctx)

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompt_texts = [f"{prompt_prefix} {name}." for name in self.class_names]
        tokenized_prompts = clip.tokenize(prompt_texts)
        tokenized_prompts = tokenized_prompts.to(clip_model.token_embedding.weight.device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

    def _init_context(self, clip, clip_model):
        if not self.ctx_init:
            ctx = torch.empty(
                self.n_ctx,
                self.ctx_dim,
                dtype=self.dtype,
                device=self.device,
            )
            nn.init.normal_(ctx, std=0.02)
            return ctx

        tokenized_ctx = clip.tokenize(self.ctx_init)
        tokenized_ctx = tokenized_ctx.to(clip_model.token_embedding.weight.device)
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

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
