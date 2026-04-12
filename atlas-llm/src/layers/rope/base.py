from abc import ABC, abstractmethod
from typing import Tuple, Type, TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound="RopeBase")

class RopeBase(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def from_cfg(cls: Type[T], cfg) -> T:
        pass

    @abstractmethod
    def forward(self, pos_start, pos_end) -> Tuple[torch.Tensor, torch.Tensor]:
        pass