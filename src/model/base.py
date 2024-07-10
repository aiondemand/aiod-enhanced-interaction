from abc import ABC, abstractmethod 
import torch

class EmbeddingModel(ABC):
    @abstractmethod
    def forward(self, texts: list[str]) -> torch.Tensor:
        pass