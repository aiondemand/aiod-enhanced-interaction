from __future__ import annotations

import torch
from transformers.utils import logging

from app.config import settings

from .architecture import (
    Basic_EmbeddingModel,
    EmbeddingModel,
    Hierarchical_EmbeddingModel,
    SentenceTransformerToHF,
    TokenizerTextSplitter,
)

# Hide unwanted warnings regarding tokenization
logging.set_verbosity(40)


class AiModel:
    @classmethod
    def get_device(cls) -> str:
        return "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"

    def __init__(self, device: torch.device = "cpu") -> None:
        self.use_chunking = settings.MILVUS.STORE_CHUNKS
        self.model = AiModel.load_model(self.use_chunking, device)

    @staticmethod
    def load_model(use_chunking: bool, device: torch.device = "cpu") -> EmbeddingModel:
        transformer = SentenceTransformerToHF(settings.MODEL_LOADPATH, trust_remote_code=True)
        text_splitter = TokenizerTextSplitter(
            transformer.tokenizer, chunk_size=512, chunk_overlap=0.25
        )

        if use_chunking:
            model = Hierarchical_EmbeddingModel(
                transformer,
                tokenizer=transformer.tokenizer,
                token_pooling="none",  # noqa: S106
                chunk_pooling="none",
                max_supported_chunks=11,
                text_splitter=text_splitter,
                dev=device,
            )
        else:
            model = Basic_EmbeddingModel(
                transformer,
                transformer.tokenizer,
                pooling="none",
                document_max_length=4096,
                dev=device,
            )

        model.eval()
        model.to(device)
        return model

    @torch.no_grad()
    def compute_asset_embeddings(self, assets: list[str]) -> list[torch.Tensor]:
        chunks_embeddings_of_multiple_docs = self.model(assets)
        if self.use_chunking is False:
            chunks_embeddings_of_multiple_docs = [
                emb[None] for emb in chunks_embeddings_of_multiple_docs
            ]
        return chunks_embeddings_of_multiple_docs

    @torch.no_grad()
    def compute_query_embeddings(self, query: str) -> list[list[float]]:
        embedding = self.model(query)[0]

        if self.use_chunking is False:
            embedding = embedding[None]
        return embedding.cpu().numpy().tolist()

    def to_device(self, device: torch.device = "cpu") -> None:
        self.model.dev = device
        self.model.to(device)
