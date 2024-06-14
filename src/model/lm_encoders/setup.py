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

# TODO this file needs to be updated a bit, because its ugly

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
        "BAAI/bge-small-en-v1.5",
        "llmrails/ember-v1",
        "thenlper/gte-large",
        "intfloat/e5-large-v2"
    ]
    tested_other_transformers = [
        "togethercomputer/m2-bert-80M-8k-retrieval",
        "WhereIsAI/UAE-Large-V1",
        "severinsimmler/xlm-roberta-longformer-base-16384"
    ]

    @classmethod
    def setup_model_wrapper(
        cls, model_path: str, pooling: str, is_hierarchical: bool,
        max_num_chunks: int | None = None
    ) -> RepresentationModel:
        model_setup_kwargs = {
            "pooling": pooling
        }
        if is_hierarchical:
            model_setup_kwargs.update({
                "use_chunk_transformer": False,
                "max_num_chunks": max_num_chunks if max_num_chunks is not None else -1
            })
        return ModelSetup.setup_model(
            model_path, hierarchical_processing=is_hierarchical, **model_setup_kwargs
        )

    @classmethod
    def setup_model(
        cls, model_path: str, 
        hierarchical_processing: bool = True, 
        *args, **kwargs
    ) -> RepresentationModel:
        if hierarchical_processing:
            return cls._setup_hierarchical_model(model_path, *args, **kwargs)
        return cls._setup_non_hierarchical_model(model_path, *args, **kwargs)
        
    @classmethod
    def _setup_hierarchical_model(
        cls, model_path: str, *args, **kwargs
    ) -> RepresentationModel:
        if model_path in cls.tested_sentence_transformers:
            return cls._setup_sentence_transformer_hierarchical(
                model_path, *args, **kwargs
            )
        if model_path not in cls.tested_other_transformers:
            raise ValueError(f"{model_path} model is not supported yet")
        
        if model_path == "WhereIsAI/UAE-Large-V1":
            return cls._setup_uae_large_v1(*args, **kwargs)
        return None
    
    @classmethod
    def _setup_non_hierarchical_model(
        cls, model_path: str, *args, **kwargs
    ) -> RepresentationModel:
        if model_path in cls.tested_sentence_transformers:
            return cls._setup_sentence_transformer_no_hierarchical(
                model_path, *args, **kwargs
            )
        if model_path not in cls.tested_other_transformers:
            raise ValueError(f"{model_path} model is not supported yet")
        
        if model_path == "togethercomputer/m2-bert-80M-8k-retrieval":
            return cls._setup_m2_bert_8k(*args, **kwargs)
        elif model_path == "severinsimmler/xlm-roberta-longformer-base-16384":
            return cls._setup_longformer(*args, **kwargs)
        return None
    
    @classmethod
    def setup_tokenizer(cls, model_path: str, *args, **kwargs) -> PreTrainedTokenizer:
        if model_path in cls.tested_sentence_transformers:
            transformer = SentenceTransformer(model_path)
            tokenizer = transformer[0].tokenizer
            tokenizer.model_max_length = transformer[0].max_seq_length
            return tokenizer
        if model_path not in cls.tested_other_transformers:
            raise ValueError(f"{model_path} model is not supported yet")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # this is a bit ugly...
        if model_path == "togethercomputer/m2-bert-80M-8k-retrieval":
            tokenizer.model_max_length = 8192
        elif model_path == "WhereIsAI/UAE-Large-V1":
            tokenizer.model_max_length = 512
        elif model_path == "severinsimmler/xlm-roberta-longformer-base-16384":
            tokenizer.model_max_length = 16384
        
        return tokenizer
    
    @classmethod
    def _setup_sentence_transformer_hierarchical(
        cls, model_path: str, max_num_chunks: int,
        use_chunk_transformer: bool = False, 
        pooling: str = "mean",
        parallel_chunk_processing: bool = True,
        **kwargs
    ) -> HierarchicalLM_TrainingModel:
        assert model_path in cls.tested_sentence_transformers, \
        f"{model_path} hasn't been tested to work in this code yet"
        
        transformer = SentenceTransformerToHF(model_path=model_path)
        enc = transformer.tokenizer("test", return_tensors="pt")
        transformer.embedding_dim = transformer(**enc)[0].shape[-1]
        
        chunk_transformer = None
        if use_chunk_transformer:
            chunk_transformer = cls._init_chunk_transformer(
                transformer.embedding_dim, max_num_chunks=max_num_chunks
            )
        model = HierarchicalLM_TrainingModel(
            transformer, tokenizer=transformer.tokenizer, token_pooling="none", 
            chunk_transformer=chunk_transformer,
            chunk_pooling=pooling,
            parallel_chunk_processing=parallel_chunk_processing,
            max_supported_chunks=max_num_chunks
        )
        model.to(utils.get_device())
        model.eval()
        return model
    
    @classmethod
    def _setup_sentence_transformer_no_hierarchical(
        cls, model_path: str, pooling: str = "mean"
    ) -> Basic_RepresentationModel:
        transformer = SentenceTransformerToHF(model_path=model_path)
        model = Basic_RepresentationModel(
            transformer, transformer.tokenizer, pooling=pooling
        )
        model.to(utils.get_device())
        model.eval()
        return model

    @classmethod
    def _setup_longformer(
        cls, pooling: str = "mean", **kwargs
    ) -> Basic_RepresentationModel:
        p = "severinsimmler/xlm-roberta-longformer-base-16384"
        transformer = AutoModel.from_pretrained(p)
        tokenizer = AutoTokenizer.from_pretrained(p)
        tokenizer.model_max_length = 16384

        model = Basic_RepresentationModel(
            transformer, tokenizer, 
            pooling=pooling
        )
        model.eval()
        model.to(utils.get_device())
        return model

    @classmethod
    def _setup_m2_bert_8k(cls, **kwargs) -> Basic_RepresentationModel:
        transformer = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-8k-retrieval",
            trust_remote_code=True
        )       
        transformer.register_forward_hook(
            lambda module, inp, out: [out["sentence_embedding"]]
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.model_max_length = 8192
        
        model = Basic_RepresentationModel(transformer, tokenizer, pooling="none")
        model.eval()
        model.to(utils.get_device())
        return model

    @classmethod
    def _setup_uae_large_v1(
        cls,
        max_num_chunks: int,
        train_chunk_transformer: bool = False, 
        pooling: str = "mean",
        **kwargs
    ) -> HierarchicalLM_TrainingModel:
        p = "WhereIsAI/UAE-Large-V1"
        transformer = AutoModel.from_pretrained(p)
        tokenizer = AutoTokenizer.from_pretrained(p)

        enc = tokenizer("test", return_tensors="pt")
        transformer.embedding_dim = transformer(**enc)[0].shape[-1]

        chunk_transformer = None
        if train_chunk_transformer:
            chunk_transformer = cls._init_chunk_transformer(
                transformer.embedding_dim, max_num_chunks
            )

        model = HierarchicalLM_TrainingModel(
            transformer, tokenizer=tokenizer, 
            token_pooling="CLS_token", 
            chunk_transformer=chunk_transformer, 
            chunk_pooling=pooling
        )
        model.to(utils.get_device())
        model.eval()
        return model
    

    @staticmethod
    def _init_chunk_transformer(
        hidden_size: int, max_num_chunks: int, num_layers: int = 6,
        cutoff_p_theshold: float = 0.98
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
