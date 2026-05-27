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

        # Adapter 的核心是一个瓶颈 MLP：先降维压缩，再升维回到原 CLIP 特征维度。
        # 这样新增参数量比直接训练完整 CLIP 小很多。
        self.down = nn.Linear(self.dim, self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.up = nn.Linear(self.hidden_dim, self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # up 层初始化为 0，使训练刚开始时 delta 接近 0，
        # 整个 Adapter 接近“输入什么就输出什么”的恒等映射。
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual 保存原始 CLIP 特征，delta 是 Adapter 学到的轻量修正量。
        residual = x
        delta = self.up(self.drop(self.act(self.down(x))))
        # residual_scale 控制修正量强度，避免一开始大幅破坏原始特征空间。
        out = residual + self.residual_scale * delta
        # 检索/分类后续通常使用余弦相似度，因此输出继续保持 L2 归一化。
        return F.normalize(out, dim=-1)
