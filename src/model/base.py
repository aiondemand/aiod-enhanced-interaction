from abc import ABC, abstractmethod 
import torch

class EmbeddingModel(ABC):
    @abstractmethod
    def __call__(self, texts: list[str]) -> torch.Tensor:
        pass