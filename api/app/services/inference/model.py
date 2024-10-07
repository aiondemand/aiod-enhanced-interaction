from __future__ import annotations

import torch
from app.config import settings

from .architecture import (
    Basic_EmbeddingModel,
    EmbeddingModel,
    Hierarchical_EmbeddingModel,
    SentenceTransformerToHF,
    TokenizerTextSplitter,
)


class AiModel:
    def __init__(self, device: torch.device = "cpu") -> None:
        self.use_chunking = settings.MILVUS.STORE_CHUNKS
        self.model = self.load_model(self.use_chunking, device)

    def load_model(
        self, use_chunking: bool, device: torch.device = "cpu"
    ) -> EmbeddingModel:
        transformer = SentenceTransformerToHF(
            settings.MODEL_LOADPATH, trust_remote_code=True
        )
        text_splitter = TokenizerTextSplitter(
            transformer.tokenizer, chunk_size=512, chunk_overlap=0.25
        )

        if use_chunking:
            model = Hierarchical_EmbeddingModel(
                transformer,
                tokenizer=transformer.tokenizer,
                token_pooling="none",
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

    def compute_asset_embeddings(self, assets: list[str]) -> list[torch.Tensor]:
        with torch.no_grad():
            chunks_embeddings_of_multiple_docs = self.model(assets)
        if self.use_chunking is False:
            chunks_embeddings_of_multiple_docs = [
                emb[None] for emb in chunks_embeddings_of_multiple_docs
            ]
        return chunks_embeddings_of_multiple_docs

    def compute_query_embeddings(self, queries: list[str]) -> list[list[float]]:
        with torch.no_grad():
            embeddings = self.model(queries)
        if self.use_chunking:
            return torch.vstack([emb[0] for emb in embeddings]).cpu().numpy().tolist()
        return torch.vstack(embeddings).cpu().numpy().tolist()

    def to_device(self, device: torch.device = "cpu") -> None:
        self.model.dev = device
        self.model.to(device)
