from abc import ABC, abstractmethod
import torch

from data_types import SemanticSearchResult


class EmbeddingModel(ABC):
    @abstractmethod
    def forward(self, texts: list[str]) -> list[torch.Tensor]:
        """
        Main endpoint that wraps the logic of two functions
        'preprocess_input' and '_forward'

        Returns a list of tensors representing either entire documents or
        the chunks documents consist of
        """

    @abstractmethod
    def _forward(self, encodings: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Function called to perform a model forward pass on a input data
        that is represented by the 'encodings' argument
        """
        pass

    @abstractmethod
    def preprocess_input(self, texts: list[str]) -> dict:
        """
        Function to process a batch of data and return it a format that is
        further fed into a model
        """
        pass


class RetrievalSystem(ABC):
    @abstractmethod
    def forward(self, queries: SemanticSearchResult) -> list[list[str]]:
        """
        Function encapsulating the entire pipeline of processing queries and
        retrieving (or potentionally further) generating results

        This function outputs a IDs of the retrieved, most similar documents
        """
        pass
