# distill_config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class DistillConfig:
    # T = 16 的时间轴（注意最后我用 999 避免越界）
    tau: List[int] = field(default_factory=lambda: [
        0, 63, 125, 188, 250, 313, 375, 438,
        470, 485, 500, 515, 530, 625, 750, 875, 999
    ])
#插入帧在t = 500
#等分，500附近一般密集，非常密集（一共16步 500 附近直接给他9步 497-503），progressive distillion
#其他种类的蒸馏：压缩结构的蒸馏t = 1000, construction keep t = 1000, compare with original full model at t = 1000

    # 想在哪些原始时间步上做蒸馏（例如只蒸 500，或者 [470,500,530]）
    distill_timesteps: List[int] = field(default_factory=lambda: [500])

    # 损失的权重
    lambda_trans: float = 1.0
    lambda_score: float = 1.0

    # 训练相关
    lr: float = 1e-4
