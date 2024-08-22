from typing import Literal
from transformers import (
    AutoModel, AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaConfig, RobertaModel,
)

from model.base import EmbeddingModel
from model.embedding_models.models import (
    Basic_EmbeddingModel, 
    Hierarchical_EmbeddingModel,
    SentenceTransformerToHF,
    TokenizerTextSplitter
)

import utils

# TODO this model setup is still pretty ugly for my taste...
# For now we are interested in only the following functions initializing LM
    # 

class ModelSetup:
    """
    Class containing various wrappers for setting up and initializing the correct 
    transformer architecture
    """
    tested_sentence_transformers = [
        "sentence-transformers/gtr-t5-large",
        "sentence-transformers/gtr-t5-xl",
        "sentence-transformers/sentence-t5-xl",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "sentence-transformers/all-MiniLM-L12-v2",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "llmrails/ember-v1",
        "thenlper/gte-large",
        "intfloat/e5-large-v2",
        "Alibaba-NLP/gte-large-en-v1.5",
    ]
    tested_hf_hierarchical_transformers = [
        "WhereIsAI/UAE-Large-V1",
    ]
    tested_hf_non_hierarchical_transformers = [
        "togethercomputer/m2-bert-80M-8k-retrieval",
        "severinsimmler/xlm-roberta-longformer-base-16384",
    ]

    @classmethod
    def setup_hierarchical_model(
        cls, model_path: str, max_num_chunks: int,
        use_chunk_transformer: bool = False, 
        token_pooling: str = "CLS_token",
        chunk_pooling: str = "mean",
        parallel_chunk_processing: bool = True,
    ) -> Hierarchical_EmbeddingModel:
        if model_path not in (
            cls.tested_sentence_transformers + 
            cls.tested_hf_hierarchical_transformers
        ):
            raise ValueError(f"{model_path} model is not supported yet")
        
        if model_path in cls.tested_sentence_transformers:
            transformer = SentenceTransformerToHF(model_path)
            tokenizer = transformer.tokenizer
            token_pooling = "none"
        elif model_path in cls.tested_hf_hierarchical_transformers:
            transformer = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
                        
        chunk_transformer = None
        if use_chunk_transformer:
            chunk_transformer = cls._init_chunk_transformer(
                transformer.embedding_dim, max_num_chunks=max_num_chunks
            )
        model = Hierarchical_EmbeddingModel(
            transformer, tokenizer=tokenizer, token_pooling=token_pooling,
            chunk_transformer=chunk_transformer,
            chunk_pooling=chunk_pooling,
            parallel_chunk_processing=parallel_chunk_processing,
            max_supported_chunks=max_num_chunks
        )
        model.to(utils.get_device())
        model.eval()
        return model
    
    @classmethod
    def setup_non_hierarchical_model(
        cls, model_path: str, document_max_length: int = -1,
        pooling: str = "CLS_token"
    ) -> EmbeddingModel:
        if model_path not in (
            cls.tested_sentence_transformers +
            cls.tested_hf_non_hierarchical_transformers
        ):
            raise ValueError(f"{model_path} model is not supported yet")
        if model_path in cls.tested_sentence_transformers:
            return cls._setup_sentence_transformer_no_hierarchical(
                model_path, document_max_length=document_max_length
            )
        if model_path == "togethercomputer/m2-bert-80M-8k-retrieval":
            return cls._setup_m2_bert_8k(document_max_length=document_max_length)
        elif model_path == "severinsimmler/xlm-roberta-longformer-base-16384":
            return cls._setup_longformer(
                pooling=pooling, document_max_length=document_max_length
            )
        
        raise ValueError(f"{model_path} model is not supported yet")
            
    @classmethod
    def _setup_sentence_transformer_no_hierarchical(
        cls, model_path: str, document_max_length: int = -1
    ) -> Basic_EmbeddingModel:
        transformer = SentenceTransformerToHF(model_path)
        model = Basic_EmbeddingModel(
            transformer, transformer.tokenizer, pooling="none",
            document_max_length=document_max_length
        )
        model.to(utils.get_device())
        model.eval()
        return model

    @classmethod
    def _setup_longformer(
        cls, pooling: str = "mean", document_max_length: int = -1
    ) -> Basic_EmbeddingModel:
        p = "severinsimmler/xlm-roberta-longformer-base-16384"
        transformer = AutoModel.from_pretrained(p)
        tokenizer = AutoTokenizer.from_pretrained(p)
        tokenizer.model_max_length = 16384

        model = Basic_EmbeddingModel(
            transformer, tokenizer, 
            pooling=pooling,
            document_max_length=document_max_length
        )
        model.eval()
        model.to(utils.get_device())
        return model

    @classmethod
    def _setup_m2_bert_8k(
        cls, document_max_length: int = -1
    ) -> Basic_EmbeddingModel:
        transformer = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-8k-retrieval",
            trust_remote_code=True
        )       
        transformer.register_forward_hook(
            lambda module, inp, out: [out["sentence_embedding"]]
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.model_max_length = 8192
        
        model = Basic_EmbeddingModel(
            transformer, tokenizer, 
            pooling="none", document_max_length=document_max_length
        )
        model.eval()
        model.to(utils.get_device())
        return model
        
    @classmethod
    def _setup_multilingual_e5_large(
        cls, chunk_pooling: Literal["mean", "none"] = "none",
        max_supported_chunks: int = 11 #around 4k tokens
    ) -> Hierarchical_EmbeddingModel:    
        transformer = SentenceTransformerToHF("intfloat/multilingual-e5-large")
        prepend_prompt_fn = lambda t: "query: " + t

        text_splitter = TokenizerTextSplitter(
            transformer.tokenizer, chunk_size=512, chunk_overlap=0.25
        )
        model = Hierarchical_EmbeddingModel(
            transformer, 
            tokenizer=transformer.tokenizer,
            token_pooling="none",
            chunk_pooling=chunk_pooling,
            max_supported_chunks=max_supported_chunks,
            text_splitter=text_splitter,
            preprocess_text_fn=prepend_prompt_fn
        )
        model.eval()
        model.to(utils.get_device())
        return model

    @classmethod
    def _setup_gte_large(
        cls, model_max_length: int | None = 4096
    ) -> Basic_EmbeddingModel: 
        transformer = SentenceTransformerToHF(
            "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
        )

        if model_max_length is None:
            model_max_length = transformer.tokenizer.model_max_length

        model = Basic_EmbeddingModel(
            transformer, 
            transformer.tokenizer, 
            pooling="none", 
            document_max_length=model_max_length
        )
        model.eval()
        model.to(utils.get_device())
        return model
    
    @classmethod
    def _setup_gte_large_hierarchical(
        cls, chunk_pooling: Literal["mean", "none"] = "none", 
        max_supported_chunks: int = 11
    ) -> Hierarchical_EmbeddingModel:
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
            chunk_pooling=chunk_pooling,
            max_supported_chunks=max_supported_chunks,
            text_splitter=text_splitter,
        )
        model.eval()
        model.to(utils.get_device())
        return model
        
    @classmethod
    def _setup_gte_qwen2(
        cls, prepend_prompt: str | None = None,
    ) -> Basic_EmbeddingModel:
        transformer = SentenceTransformerToHF(
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
        )
        if prepend_prompt is None:
            prepend_prompt = (
                "Instruct: Given a search query, "
                + "retrieve relevant assets defined in the query\nQuery: "
            )
        prepend_prompt_fn = lambda t: prepend_prompt + t

        model = Basic_EmbeddingModel(
            transformer, 
            transformer.tokenizer,
            pooling="none",
            document_max_length=transformer.tokenizer.model_max_length,
            preprocess_text_fn=prepend_prompt_fn
        )
        model.eval()
        model.to(utils.get_device())
        return model
    
    @classmethod
    def _setup_bge_large(cls, chunk_pooling: Literal["mean", "none"] = "none",
        max_supported_chunks: int = 11 #around 4k tokens)
    ) -> Hierarchical_EmbeddingModel:
        transformer = SentenceTransformerToHF("BAAI/bge-large-en-v1.5")

        text_splitter = TokenizerTextSplitter(
            transformer.tokenizer, chunk_size=512, chunk_overlap=0.25
        )
        model = Hierarchical_EmbeddingModel(
            transformer, 
            tokenizer=transformer.tokenizer,
            token_pooling="none",
            chunk_pooling=chunk_pooling,
            max_supported_chunks=max_supported_chunks,
            text_splitter=text_splitter,
        )
        model.eval()
        model.to(utils.get_device())
        return model
            
    @staticmethod
    def _init_chunk_transformer(
        hidden_size: int, max_num_chunks: int, num_layers: int = 6
    ) -> RobertaModel:
        num_attention_heads = hidden_size // 64
        assert num_attention_heads == hidden_size / 64, \
        "'num_attention_heads' is not an integer"

        kwargs = { 
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_layers,
            "max_position_embeddings": max_num_chunks + 2
        }
        config = RobertaConfig(**kwargs)
        chunk_transformer = RobertaModel(config, add_pooling_layer=False)

        return chunk_transformer
