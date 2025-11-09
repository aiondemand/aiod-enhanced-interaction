from typing import List, Optional
from pydantic import Field

from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata
from app.schemas.asset_metadata.new_schemas.types import Varchar32, Varchar64


class MlModel_AiExtractedMetadata(AssetSpecificMetadata):
    """
    Metadata fields that apply only to assets of type 'ml_model'.
    Every attribute is optional so an agent can omit values it cannot
    infer with confidence.
    """

    parameter_count_millions: Optional[float] = Field(
        None,
        description=(
            "Approximate total number of trainable parameters in millions, e.g. 7 000 for a 7B model."
        ),
    )

    model_size_gigabytes: Optional[float] = Field(
        None,
        description=("Compressed checkpoint size on disk in gigabytes"),
    )

    architecture_families: Optional[List[Varchar32]] = Field(
        None,
        description=(
            "High-level architecture chosen from "
            "['transformer', 'cnn', 'rnn', 'gnn', 'diffusion', 'boosted_trees', "
            "'linear', 'ensemble', 'other']."
        ),
        max_length=16,
    )

    fine_tuned: Optional[bool] = Field(
        None,
        description=(
            "True if additional task-specific fine-tuning was performed. This is different to instruction finetuning performed on LLMs."
        ),
    )

    training_techniques: Optional[List[Varchar64]] = Field(
        None,
        description=(
            "High-level training methods applied, e.g. "
            "['supervised', 'self_supervised', 'rlhf', 'dpo', "
            "'distillation', 'pruning', 'quantization_aware']."
        ),
        max_length=16,
    )

    training_compute_gpu_hours: Optional[float] = Field(
        None,
        description=("Approximate aggregated GPU hours consumed during training, e.g. 1000."),
    )

    ######### LLM specific fields #########

    llm_max_sequence_length: Optional[int] = Field(
        None,
        description=(
            "Maximum input sequence length supported, e.g. 4096 tokens. Use this attribute for LLMs only."
        ),
    )

    vocab_size: Optional[int] = Field(
        None,
        description=(
            "Size of the tokenizer vocabulary, e.g. 128 000 for 128k tokens. Use this attribute for LLMs only."
        ),
    )

    llm_training_data_volume_tokens: Optional[int] = Field(
        None,
        description=(
            "Total number of text tokens (or equivalent units) seen during pre-training, e.g. 1 000 000 000 000 for 1 trillion tokens. "
            "Use this attribute for LLMs only."
        ),
    )
