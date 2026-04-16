import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPResidualAdapter(nn.Module):
    """
    轻量级特征 Adapter：两层 MLP + 残差连接。

    设计目标：
    - 接在缓存好的 CLIP 视觉特征之后
    - 只训练很少量参数
    - 默认尽量接近恒等映射，避免破坏原始特征空间
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_scale = float(residual_scale)

        self.down = nn.Linear(self.dim, self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.up = nn.Linear(self.hidden_dim, self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        delta = self.up(self.drop(self.act(self.down(x))))
        out = residual + self.residual_scale * delta
        return F.normalize(out, dim=-1)
