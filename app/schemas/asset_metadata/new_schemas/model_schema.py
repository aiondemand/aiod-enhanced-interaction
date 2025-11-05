from typing import List, Optional
from pydantic import Field

from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata


class Model_AiExtractedMetadata(AssetSpecificMetadata):
    """
    Metadata fields that apply only to assets of type 'ml_model'.
    Every attribute is optional so an agent can omit values it cannot
    infer with confidence.
    """

    parameter_count_millions: Optional[int] = Field(
        None,
        description=(
            "Approximate total number of trainable parameters in millions, e.g. 7 000 for a 7B model."
        ),
    )

    model_size_megabytes: Optional[int] = Field(
        None,
        description=(
            "Compressed checkpoint size on disk in megabytes, e.g. 13 200 for 13.2GB model."
        ),
    )

    # TODO create a enum for this field
    architecture_family: Optional[str] = Field(
        None,
        description=(
            "High-level architecture chosen from "
            "['transformer', 'cnn', 'rnn', 'gnn', 'diffusion', 'boosted_trees', "
            "'linear', 'ensemble', 'other']."
        ),
    )

    # TODO create a enum for this field
    quantization_level: Optional[str] = Field(
        None,
        description=(
            "Precision of the published weights. "
            "Enum: ['fp32', 'bf16', 'fp16', 'int8', 'int4', 'mixed', 'none']."
        ),
    )

    fine_tuned: Optional[bool] = Field(
        None,
        description=(
            "True if additional task-specific fine-tuning was performed. This is different to instruction finetuning performed on LLMs."
        ),
    )

    # TODO create a enum for this field
    training_techniques: Optional[List[str]] = Field(
        None,
        description=(
            "High-level training methods applied, e.g. "
            "['supervised', 'self_supervised', 'rlhf', 'dpo', "
            "'distillation', 'pruning', 'quantization_aware']."
        ),
    )

    training_compute_gpu_hours: Optional[int] = Field(
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

    llm_type: Optional[str] = Field(
        None,
        description=(
            "Type of language model - either 'base_llm' for models that only continue text "
            "or 'instruction_following' for models fine-tuned to follow instructions/commands. "
            "Use this attribute for LLMs only."
        ),
        enum=["base_llm", "instruction_following"],
    )

    llm_function_calling: Optional[bool] = Field(
        None,
        description=(
            "Whether the model has native support for structured function calling via JSON schema. "
            "Use this attribute for LLMs only."
        ),
    )
