from __future__ import annotations

import torch

from .architecture import (
    EmbeddingModel,
    Hierarchical_EmbeddingModel,
    SentenceTransformerToHF,
    TokenizerTextSplitter,
)


class AiModel:
    def init(self) -> None:
        self.model = self.load_model()

    def load_model(self) -> EmbeddingModel:
        transformer = SentenceTransformerToHF(
            "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
        )
        text_splitter = TokenizerTextSplitter(
            transformer.tokenizer, chunk_size=512, chunk_overlap=0.25
        )
        model = Hierarchical_EmbeddingModel(
            transformer,
            tokenizer=transformer.tokenizer,
            token_pooling="none",
            chunk_pooling="none",
            max_supported_chunks=11,
            text_splitter=text_splitter,
            dev="cpu",
        )
        model.eval()
        return model

    def compute_embeddings(self, queries: list[str]) -> list[list[float]]:
        # TODO later -> queries can be potentionally longer than 512 tokens
        # for now we dont bother with long queries, they will truncated...
        with torch.no_grad():
            embeddings = self.model(queries)
            return torch.vstack([emb[0] for emb in embeddings]).cpu().numpy().tolist()

    def to_device(self, device: torch.device = "cpu") -> None:
        self.model.dev(device)
        self.model.to(device)


class AiModelForUserQueries(AiModel):
    _instance: AiModel | None = None

    def __new__(cls) -> AiModel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance


class AiModelForBatchProcessing(AiModel):
    _instance: AiModel | None = None

    def __new__(cls) -> AiModel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance
