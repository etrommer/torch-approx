from typing import List, Optional
import copy
import torch
from torch.nn.modules.batchnorm import _NormBase


class MultiBatchNorm(torch.nn.Module):
    def __init__(self, batch_norm: _NormBase, size: int):
        super().__init__()
        assert size >= 1, "Need at least one forward dimension"
        self._shadow_norms: List[_NormBase] = [
            copy.deepcopy(batch_norm) for _ in range(size)
        ]
        self._mul_idx: int = 0
        self.mul_idx = 0
        self.fwd_norm: Optional[_NormBase]

    def forward(self, x):
        return self.fwd_norm(x)

    @property
    def mul_idx(self):
        return self._mul_idx

    @mul_idx.setter
    def mul_idx(self, new_idx: int):
        assert new_idx < len(
            self._shadow_norms
        ), f"Bad Index {new_idx} for size {len(self._shadow_norms)}"
        self._mul_idx = new_idx
        self.fwd_norm = self._shadow_norms[self._mul_idx]
