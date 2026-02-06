from abc import abstractmethod, ABC
from typing import Literal, Optional
from torch.nn import Module
from torch import Tensor
from torch_geometric.data import Data
from torch.types import Number

Stage = Literal['train', 'val', 'test']
Metrics = dict[str, Number]


class TrainableModule(Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def step(self, data: Data, stage: Stage, global_weights) -> tuple[Optional[Tensor], Metrics]: pass

    @abstractmethod
    def predict(self, data: Data) -> Tensor: pass

    @abstractmethod
    def reset_parameters(self): pass
