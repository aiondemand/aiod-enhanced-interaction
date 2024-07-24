from abc import ABC, abstractmethod
import torch

from dataset import QueryDatapoint


class EmbeddingModel(ABC):
    @abstractmethod
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Main endpoint that wraps the logic of two functions
        'preprocess_input' and '_forward'
        """

    @abstractmethod
    def _forward(self, encodings: dict[str, torch.Tensor]) -> torch.Tensor: 
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
    def forward(self, queries: list[QueryDatapoint]) -> list[list[str]]:
        """
        Function encapsulating the entire pipeline of processing queries and
        retrieving (or potentionally further) generating results

        This function outputs a list of document IDs for each user query
        """
        pass


    

