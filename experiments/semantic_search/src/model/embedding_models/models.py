from typing import Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

from model.base import EmbeddingModel
import utils


class TokenizerTextSplitter:
    def __init__(
        self, tokenizer: PreTrainedTokenizer,
        chunk_size: int | None = None, chunk_overlap: float = 0.25
    ) -> None:
        self.tokenizer = tokenizer
        if chunk_size is None:
            chunk_size = tokenizer.model_max_length - 2

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __call__(self, text: str) -> list[str]:
        # TODO Suppress warnings
        tokens = self.tokenizer.tokenize(text)

        chunks = []
        start_idx = 0
        while True:
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            start_idx = end_idx - int(self.chunk_size * self.chunk_overlap)
            if end_idx == len(tokens):
                break

        return chunks


class SentenceTransformerToHF(torch.nn.Module):
    """
    Class for transforming SentenceTransformer object from sentence_transformers
    package into a HuggingFace interface

    Important extracted objects from the SentenceTransformer (attributes):
    'hf_transformer': HuggingFace model
    'tokenizer': HuggingFace tokenizer
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.sentence_transformer = SentenceTransformer(*args, **kwargs, device="cpu")

        self.hf_transformer = self.sentence_transformer[0].auto_model
        self.tokenizer = self.sentence_transformer[0].tokenizer
        self.tokenizer.model_max_length = self.sentence_transformer[0].max_seq_length

        enc = self.tokenizer("test", return_tensors="pt")
        self.embedding_dim = self.hf_transformer(**enc)[0].shape[-1]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        encodings = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        encodings.update(kwargs)
        for it, module in enumerate(self.sentence_transformer):
            if it == 0:
                # transformer step
                token_embeddings = self.hf_transformer(**encodings)[0]
                encodings.update({"token_embeddings": token_embeddings})
            else:
                # additional steps -> pooling, projection, normalization...
                encodings = module(encodings)

        return [encodings["sentence_embedding"]]


class Basic_EmbeddingModel(torch.nn.Module, EmbeddingModel):
    """
    Class representing models that process the input documents in their entirety
    without needing to divide them into separate chunks.
    """
    def __init__(
        self, transformer: PreTrainedModel | SentenceTransformerToHF,
        tokenizer: PreTrainedTokenizer, pooling: str = "max",
        document_max_length: int = -1,
        global_attention_mask: bool = False,
        preprocess_text_fn: Callable[[str], str] | None = None
    ) -> None:
        """
        Arguments:
        'pooling': represents how the final representation of the input
        documents are computed
        """
        super().__init__()
        assert pooling in ["mean", "max", "CLS_token", "none"]

        self.transformer = transformer
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.document_max_length = document_max_length
        self.global_attention_mask = global_attention_mask
        self.preprocess_text_fn = preprocess_text_fn

    def forward(self, texts: list[str]) -> list[torch.Tensor]:
        encoding = self.preprocess_input(texts)
        return self._forward(encoding)

    def _forward(self, encodings: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        out = self.transformer(**encodings)[0]
        out = _pool(out, encodings["attention_mask"], self.pooling)
        return [emb for emb in out]

    def preprocess_input(self, texts: list[str]) -> dict:
        if self.preprocess_text_fn is not None:
            texts = [self.preprocess_text_fn(t) for t in texts]

        doc_max_length = (
            self.document_max_length
            if self.document_max_length > 0
            else self.tokenizer.model_max_length
        )
        encodings = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True,
            max_length=doc_max_length
        ).to(utils.get_device())

        if self.global_attention_mask:
            encodings["global_attention_mask"] = torch.zeros_like(
                encodings["attention_mask"], device=utils.get_device()
            )
            encodings["global_attention_mask"][:, 0] = 1
        return encodings


class Hierarchical_EmbeddingModel(torch.nn.Module, EmbeddingModel):
    """
    Class representing models that process the input documents by firstly individually
    processing their chunks before further accumulating the chunk information to
    compute the representations of the whole documents
    """
    def __init__(
        self, input_transformer: PreTrainedModel | SentenceTransformerToHF,
        chunk_transformer: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        token_pooling: str = "CLS_token",
        chunk_pooling: str = "mean",
        parallel_chunk_processing: bool = True,
        max_supported_chunks: int = -1,
        text_splitter: TokenizerTextSplitter | None = None,
        preprocess_text_fn: Callable[[str], str] | None = None
    ) -> None:
        """
        Arguments:
        'input_transformer': First-level transformer (pre-trained and
        imported from HuggingFace) used for processing the document chunks
        'chunk_transformer': Second-level transformer used for processing
        the chunk representations and computes the final document embeddings
        'token_pooling': represents how the chunk representations of the input
        documents are computed
        'chunk_pooling' represents how the document representations of the input
        documents are computed
        'parallel_chunk_processing': whether to compute chunk representations in
        parallel (which is more time efficient and slightly more memory-demanding)
        or not
        'max_supported_chunks': maximum number of chunks found in one document
        the model supports. If there are more document chunks than supported,
        chunks will be truncated
        """
        super().__init__()
        assert token_pooling in [
            "mean", "max", "CLS_token", "none"
        ]
        assert chunk_pooling in ["mean", "max", "none"]

        self.input_transformer = input_transformer
        self.chunk_transformer = chunk_transformer
        self.tokenizer = tokenizer

        self.token_pooling = token_pooling
        self.chunk_pooling = chunk_pooling

        self.parallel_chunk_processing = parallel_chunk_processing
        self.max_supported_chunks = max_supported_chunks

        if text_splitter is None:
            text_splitter = TokenizerTextSplitter(self.tokenizer)
        self.text_splitter = text_splitter

        self.preprocess_text_fn = preprocess_text_fn

    def forward(self, texts: list[str]) -> list[torch.Tensor]:
        encoding = self.preprocess_input(texts)
        return self._forward(encoding)

    def _forward(self, encodings: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        chunk_embeddings = self._first_level_forward(
            encodings["input_encodings"],
            encodings["max_num_chunks"]
        )
        doc_embeddings = self._second_level_forward(
            chunk_embeddings,
            encodings["chunk_attn_mask"],
        )
        doc_embeddings = [emb for emb in doc_embeddings]

        # get rid of padding chunks if there exists
        if self.chunk_pooling == "none":
            doc_embeddings = [
                emb[chunk_mask.to(torch.bool)]
                for emb, chunk_mask in
                zip(doc_embeddings, encodings["chunk_attn_mask"])
            ]

        return doc_embeddings

    def _first_level_forward(
            self, input_encodings: list[dict] | dict, max_num_chunks: int
        ) -> list[torch.Tensor]:
        """Function representing a forward pass of the first-level transformer"""
        if self.parallel_chunk_processing:
            out = self.input_transformer(**input_encodings)[0]
            out = _pool(
                out, input_encodings["attention_mask"],
                pooling_method=self.token_pooling
            )
            out = out.reshape(out.shape[0] // max_num_chunks, max_num_chunks, -1)
            return out

        chunk_embeddings = []
        for inp in input_encodings:
            out = self.input_transformer(**inp)[0]
            chunk_embeddings.append(_pool(
                out, inp["attention_mask"],
                pooling_method=self.token_pooling,
            ))
        return chunk_embeddings

    def _second_level_forward(
        self, chunks_embeddings: list[torch.Tensor],
        chunk_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Function representing a forward pass of the second-level transformer"""
        if self.parallel_chunk_processing:
            chunk_encodings = {
                "inputs_embeds": chunks_embeddings,
                "attention_mask": chunk_attn_mask
            }
            chunk_out = chunks_embeddings
            if self.chunk_transformer is not None:
                chunk_out = self.chunk_transformer(**chunk_encodings)[0]
            doc_embedding = _pool(
                chunk_out, chunk_attn_mask,
                pooling_method=self.chunk_pooling
            )
            return doc_embedding

        all_embeddings = torch.zeros(
            chunk_attn_mask.shape[0],
            chunk_attn_mask.shape[1],
            chunks_embeddings[-1].shape[-1]
        ).to(utils.get_device())

        for chunk_it in range(len(chunks_embeddings)):
            indices = torch.where(chunk_attn_mask[:, chunk_it] != 0)[0]
            all_embeddings[indices, chunk_it] = chunks_embeddings[chunk_it]
        chunk_encodings = {
            "inputs_embeds": all_embeddings,
            "attention_mask": chunk_attn_mask
        }

        chunk_out = all_embeddings
        if self.chunk_transformer is not None:
            chunk_out = self.chunk_transformer(**chunk_encodings)[0]
        doc_embedding = _pool(
            chunk_out, chunk_attn_mask,
            pooling_method=self.chunk_pooling
        )
        return doc_embedding

    def preprocess_input(self, texts: list[str]) -> dict:
        if self.preprocess_text_fn is not None:
            texts = [self.preprocess_text_fn(t) for t in texts]
        chunked_texts = [self.text_splitter(t) for t in texts]

        # apply chunk cap limit
        if self.max_supported_chunks > 0:
            chunked_texts = [
                chunks[:self.max_supported_chunks] for chunks in chunked_texts
            ]

        num_chunks = [len(chunks) for chunks in chunked_texts]
        max_chunks = max(num_chunks)
        padded_texts = [
            chunks + [""] * (max_chunks - len(chunks)) for chunks in chunked_texts
        ]
        chunk_mask = torch.tensor(
            (np.array(padded_texts) != "").astype("int"),
            device=utils.get_device()
        )

        if self.parallel_chunk_processing:
            rectangular_texts = np.array(padded_texts).reshape(-1).tolist()
            encoding = self.tokenizer(
                rectangular_texts,
                return_tensors="pt",
                truncation=True, padding=True
            ).to(utils.get_device())
            return {
                "input_encodings": encoding,
                "chunk_attn_mask": chunk_mask,
                "max_num_chunks": max_chunks
            }

        # input_encoddings
        transposed_texts = np.array(padded_texts).T.tolist()
        encodings = [
            self.tokenizer(
                [ch for ch in chunks_of_docs if len(ch) > 0],
                return_tensors="pt",
                truncation=True, padding=True
            ).to(utils.get_device())
            for chunks_of_docs in transposed_texts
        ]
        return_obj = {
            "input_encodings": encodings,
            "chunk_attn_mask": chunk_mask,
        }
        return return_obj


def _pool(
    inp: torch.Tensor, attention_mask: torch.Tensor, pooling_method: str
) -> torch.Tensor:
    """
    Wrapper function for performing a specific pooling method that aggregates values
    from multiple sources and merges them into one representation
    """
    if pooling_method == "mean":
        return mean_pooling(inp, attention_mask)
    if pooling_method == "max":
        return max_pooling(inp, attention_mask)
    if pooling_method == "CLS_token":
        return cls_pooling(inp)
    if pooling_method == "none":
        return inp
    return None


def cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pooling method: Take CLS token only"""
    return hidden_states[:, 0]


def mean_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pooling method: Average all the representations"""
    input_mask_expanded = (attention_mask
        .unsqueeze(-1)
        .expand(hidden_states.size())
        .float()
    )
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pooling method: Take the max features found in all the representations"""
    masked_hidden_states = (
        hidden_states +
        (1 - attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()) *
        -1e9
    )
    max_values, _ = torch.max(masked_hidden_states, dim=1)
    return max_values


def weighted_sum_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Pooling method: Apply a weighted average of all the representations"""
    input_mask_expanded = (attention_mask
        .unsqueeze(-1)
        .expand(hidden_states.size())
        .float()
    )
    weights_expanded = (weights
        .unsqueeze(-1)
        .expand(hidden_states.size())
        .to(utils.get_device())
    )
    return torch.sum(hidden_states * input_mask_expanded * weights_expanded, dim=1)
