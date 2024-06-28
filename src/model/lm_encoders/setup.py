from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel, AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaConfig, RobertaModel,
    PreTrainedTokenizer
)

from .models import (
    RepresentationModel, 
    Basic_RepresentationModel, 
    HierarchicalLM_TrainingModel,
    SentenceTransformerToHF
)

import utils


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
    ) -> HierarchicalLM_TrainingModel:
        if model_path not in (
            cls.tested_sentence_transformers + 
            cls.tested_hf_hierarchical_transformers
        ):
            raise ValueError(f"{model_path} model is not supported yet")
        
        if model_path in cls.tested_sentence_transformers:
            transformer = SentenceTransformerToHF(model_path=model_path)
            tokenizer = transformer.tokenizer
            token_pooling = "none"
        elif model_path in cls.tested_hf_hierarchical_transformers:
            transformer = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
        enc = tokenizer("test", return_tensors="pt")
        transformer.embedding_dim = transformer(**enc)[0].shape[-1]
        
        chunk_transformer = None
        if use_chunk_transformer:
            chunk_transformer = cls._init_chunk_transformer(
                transformer.embedding_dim, max_num_chunks=max_num_chunks
            )
        model = HierarchicalLM_TrainingModel(
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
    ) -> RepresentationModel:
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
    ) -> Basic_RepresentationModel:
        transformer = SentenceTransformerToHF(model_path=model_path)
        model = Basic_RepresentationModel(
            transformer, transformer.tokenizer, pooling="none",
            document_max_length=document_max_length
        )
        model.to(utils.get_device())
        model.eval()
        return model

    @classmethod
    def _setup_longformer(
        cls, pooling: str = "mean", document_max_length: int = -1
    ) -> Basic_RepresentationModel:
        p = "severinsimmler/xlm-roberta-longformer-base-16384"
        transformer = AutoModel.from_pretrained(p)
        tokenizer = AutoTokenizer.from_pretrained(p)
        tokenizer.model_max_length = 16384

        model = Basic_RepresentationModel(
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
    ) -> Basic_RepresentationModel:
        transformer = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-8k-retrieval",
            trust_remote_code=True
        )       
        transformer.register_forward_hook(
            lambda module, inp, out: [out["sentence_embedding"]]
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.model_max_length = 8192
        
        model = Basic_RepresentationModel(
            transformer, tokenizer, 
            pooling="none", document_max_length=document_max_length
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
