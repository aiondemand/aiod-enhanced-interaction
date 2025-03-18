from abc import ABC, abstractmethod
from typing import Callable, Generic, Literal, TypeVar

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel, PreTrainedTokenizer


class PreprocessedInput(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BasicPreprocessedInput(PreprocessedInput):
    input_encodings: dict[str, torch.Tensor]


class HierarchicalPreprocessedInput(PreprocessedInput):
    all_chunk_input_encodings: dict[str, torch.Tensor] | None = None
    separate_chunk_input_encodings: list[dict[str, torch.Tensor]] | None = None
    chunk_attn_mask: torch.Tensor
    num_chunks: int


GenericPreprocessedInput = TypeVar("GenericPreprocessedInput", bound=PreprocessedInput)


class EmbeddingModel(torch.nn.Module, Generic[GenericPreprocessedInput], ABC):
    @abstractmethod
    def forward(self, texts: list[str] | str) -> list[torch.Tensor]:
        """
        Main endpoint that wraps the logic of two functions
        'preprocess_input' and '_forward'

        Returns a list of tensors representing either entire documents or
        the chunks documents consist of
        """
        raise NotImplementedError

    @abstractmethod
    def _forward(self, encodings: GenericPreprocessedInput) -> list[torch.Tensor]:
        """
        Function called to perform a model forward pass on a input data
        that is represented by the 'encodings' argument
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess_input(self, texts: list[str]) -> GenericPreprocessedInput:
        """
        Function to process a batch of data and return it a format that is
        further fed into a model
        """
        raise NotImplementedError


class TokenizerTextSplitter:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int | None = None,
        chunk_overlap: float = 0.25,
    ) -> None:
        self.tokenizer = tokenizer
        if chunk_size is None:
            chunk_size = tokenizer.model_max_length - 2

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __call__(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        tokens_with_special_tokens = self.tokenizer.tokenize(text, add_special_tokens=True)
        num_special_tokens = len(tokens_with_special_tokens) - len(tokens)
        actual_chunk_size = self.chunk_size - num_special_tokens

        chunks = []
        start_idx = 0
        while True:
            end_idx = min(start_idx + actual_chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            start_idx = end_idx - int(actual_chunk_size * self.chunk_overlap)
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
        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}
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


class Basic_EmbeddingModel(EmbeddingModel[BasicPreprocessedInput]):
    """
    Class representing models that process the input documents in their entirety
    without needing to divide them into separate chunks.
    """

    def __init__(
        self,
        transformer: PreTrainedModel | SentenceTransformerToHF,
        tokenizer: PreTrainedTokenizer,
        pooling: Literal["mean", "max", "CLS_token", "none"] = "max",
        document_max_length: int = -1,
        global_attention_mask: bool = False,
        preprocess_text_fn: Callable[[str], str] | None = None,
        dev: torch.device = "cpu",
    ) -> None:
        """
        Arguments:
        'pooling': represents how the final representation of the input
        documents are computed
        """
        super().__init__()
        if not pooling in ["mean", "max", "CLS_token", "none"]:
            raise ValueError("Invalid value for 'pooling'")

        self.transformer = transformer
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.document_max_length = document_max_length
        self.global_attention_mask = global_attention_mask
        self.preprocess_text_fn = preprocess_text_fn

        self.dev = dev

    def forward(self, texts: list[str] | str) -> list[torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        encoding = self.preprocess_input(texts)
        return self._forward(encoding)

    def _forward(self, encodings: BasicPreprocessedInput) -> list[torch.Tensor]:
        inp = encodings.input_encodings

        out = self.transformer(**inp)[0]
        out = _pool(out, inp["attention_mask"], self.pooling)
        return [emb for emb in out]

    def preprocess_input(self, texts: list[str]) -> BasicPreprocessedInput:
        if self.preprocess_text_fn is not None:
            texts = [self.preprocess_text_fn(t) for t in texts]

        doc_max_length = (
            self.document_max_length
            if self.document_max_length > 0
            else self.tokenizer.model_max_length
        )
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=doc_max_length,
        ).to(self.dev)

        if self.global_attention_mask:
            encodings["global_attention_mask"] = torch.zeros_like(
                encodings["attention_mask"], device=self.dev
            )
            encodings["global_attention_mask"][:, 0] = 1

        return BasicPreprocessedInput(input_encodings=encodings)


class Hierarchical_EmbeddingModel(EmbeddingModel[HierarchicalPreprocessedInput]):
    """
    Class representing models that process the input documents by firstly individually
    processing their chunks before further accumulating the chunk information to
    compute the representations of the whole documents
    """

    def __init__(
        self,
        input_transformer: PreTrainedModel | SentenceTransformerToHF,
        tokenizer: PreTrainedTokenizer,
        chunk_transformer: PreTrainedModel | None = None,
        token_pooling: Literal["mean", "max", "CLS_token", "none"] = "CLS_token",  # noqa: S107
        chunk_pooling: Literal["mean", "max", "none"] = "mean",
        parallel_chunk_processing: bool = True,
        max_supported_chunks: int = -1,
        text_splitter: TokenizerTextSplitter | None = None,
        preprocess_text_fn: Callable[[str], str] | None = None,
        dev: torch.device = "cpu",
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
        if not token_pooling in ["mean", "max", "CLS_token", "none"]:
            raise ValueError("Invalid value for 'token_pooling'")
        if not chunk_pooling in ["mean", "max", "none"]:
            raise ValueError("Invalid value for 'chunk_pooling'")

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

        self.dev = dev

    def forward(self, texts: list[str] | str) -> list[torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        inner_input = self.preprocess_input(texts)
        return self._forward(inner_input)

    def _forward(self, inner_input: HierarchicalPreprocessedInput) -> list[torch.Tensor]:
        chunk_embeddings = self._first_level_forward(inner_input)
        doc_embeddings = self._second_level_forward(
            chunk_embeddings,
            inner_input.chunk_attn_mask,
        )

        # get rid of padding chunks if we wish to return all the chunk embeddings
        if self.chunk_pooling == "none":
            doc_embeddings = [
                emb[chunk_mask.to(torch.bool)]
                for emb, chunk_mask in zip(doc_embeddings, inner_input.chunk_attn_mask)
            ]

        return doc_embeddings

    def _first_level_forward(
        self, inner_input: HierarchicalPreprocessedInput
    ) -> list[torch.Tensor]:
        """Function representing a forward pass of the first-level transformer"""
        if self.parallel_chunk_processing and inner_input.all_chunk_input_encodings is not None:
            encodings = inner_input.all_chunk_input_encodings
            num_chunks = inner_input.num_chunks

            out = self.input_transformer(**encodings)[0]
            out = _pool(
                out,
                encodings["attention_mask"],
                pooling_method=self.token_pooling,
            )
            out = out.reshape(out.shape[0] // num_chunks, num_chunks, -1)
            return [o for o in out]

        if (
            self.parallel_chunk_processing is False
            and inner_input.separate_chunk_input_encodings is not None
        ):
            chunk_embeddings = []
            for inp in inner_input.separate_chunk_input_encodings:
                out = self.input_transformer(**inp)[0]
                chunk_embeddings.append(
                    _pool(
                        out,
                        inp["attention_mask"],
                        pooling_method=self.token_pooling,
                    )
                )

            chunk_mask = inner_input.chunk_attn_mask
            padded_chunk_embeddings = torch.zeros(
                chunk_mask.shape[0],
                chunk_mask.shape[1],
                chunk_embeddings[0].shape[-1],
            ).to(self.dev)

            for chunk_it in range(len(chunk_embeddings)):
                indices = torch.where(chunk_mask[:, chunk_it] != 0)[0]
                padded_chunk_embeddings[indices, chunk_it] = chunk_embeddings[chunk_it]

            return [emb for emb in padded_chunk_embeddings]

        else:
            raise ValueError("Invalid input")

    def _second_level_forward(
        self, chunks_embeddings: list[torch.Tensor], chunk_attn_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        """Function representing a forward pass of the second-level transformer"""
        chunk_out = torch.stack(chunks_embeddings)
        chunk_encodings = {
            "inputs_embeds": chunk_out,
            "attention_mask": chunk_attn_mask,
        }

        if self.chunk_transformer is not None:
            chunk_out = self.chunk_transformer(**chunk_encodings)[0]
        doc_embeddings = _pool(chunk_out, chunk_attn_mask, pooling_method=self.chunk_pooling)
        return [emb for emb in doc_embeddings]

    def preprocess_input(self, texts: list[str]) -> HierarchicalPreprocessedInput:
        if self.preprocess_text_fn is not None:
            texts = [self.preprocess_text_fn(t) for t in texts]
        chunked_texts = [self.text_splitter(t) for t in texts]

        # apply chunk cap limit
        if self.max_supported_chunks > 0:
            chunked_texts = [chunks[: self.max_supported_chunks] for chunks in chunked_texts]

        num_chunks = [len(chunks) for chunks in chunked_texts]
        max_chunks = max(num_chunks)
        padded_texts = [chunks + [""] * (max_chunks - len(chunks)) for chunks in chunked_texts]
        chunk_mask = torch.tensor((np.array(padded_texts) != "").astype("int"), device=self.dev)

        if self.parallel_chunk_processing:
            rectangular_texts = np.array(padded_texts).reshape(-1).tolist()
            encoding = self.tokenizer(
                rectangular_texts, return_tensors="pt", truncation=True, padding=True
            ).to(self.dev)

            # Zero out encodings for empty strings
            empty_indices = np.where(np.array(rectangular_texts) == "")[0]
            if len(empty_indices) > 0:
                encoding["input_ids"][empty_indices] = 0
                encoding["attention_mask"][empty_indices] = 0

            return HierarchicalPreprocessedInput(
                all_chunk_input_encodings=encoding,
                chunk_attn_mask=chunk_mask,
                num_chunks=max_chunks,
            )

        # input_encodings
        transposed_texts = np.array(padded_texts).T.tolist()
        encodings = [
            self.tokenizer(
                [ch for ch in chunks_of_docs if len(ch) > 0],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.dev)
            for chunks_of_docs in transposed_texts
        ]
        return HierarchicalPreprocessedInput(
            separate_chunk_input_encodings=encodings,
            chunk_attn_mask=chunk_mask,
            num_chunks=max_chunks,
        )


def _pool(
    inp: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_method: Literal["mean", "max", "CLS_token", "none"],
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

    raise ValueError("Invalid pooling method")


def cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pooling method: Take CLS token only"""
    return hidden_states[:, 0]


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pooling method: Average all the representations"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pooling method: Take the max features found in all the representations"""
    masked_hidden_states = (
        hidden_states
        + (1 - attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()) * -1e9
    )
    max_values, _ = torch.max(masked_hidden_states, dim=1)
    return max_values


def weighted_sum_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: torch.Tensor,
    dev: torch.device = "cpu",
) -> torch.Tensor:
    """Pooling method: Apply a weighted average of all the representations"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    weights_expanded = weights.unsqueeze(-1).expand(hidden_states.size()).to(dev)
    return torch.sum(hidden_states * input_mask_expanded * weights_expanded, dim=1)
