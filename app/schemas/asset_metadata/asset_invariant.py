from typing import List, Optional
from pydantic import BaseModel, Field

# ASSETS TO SUPPORT
# Datasets
# ML models
# educational_resources
# publications


class AutomaticallyExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and are automatically extracted from the asset's JSON schema without the need to use an LLM.
    """

    platform: str = Field(
        ...,
        description=(
            "The platform on which the asset is hosted, e.g. 'Hugging Face', 'Zenodo', 'OpenML'."
        ),
    )
    name: str = Field(..., description=("The name of the asset on the original platform."))
    date_published: str = Field(
        ..., description=("The date the asset was published on the original platform.")
    )
    same_as: str = Field(
        ..., description=("The link pointing to the asset on the original platform.")
    )


class Base_AiExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and which frequently need to be inferred from unstructured text.
    """

    # Exists a AIoD taxonomy for this field
    research_areas: Optional[List[str]] = Field(
        None,
        description=(
            "Primary AI research disciplines, areas of research addressed by the asset, "
            "e.g. ['Natural Language Processing', 'Computer Vision', 'Reinforcement Learning']."
        ),
    )

    # TODO We need to come up with a enum for this field
    ml_tasks: Optional[List[str]] = Field(
        None,
        description=(
            "Core machine-learning task the asset tackles e.g. 'classification', 'segmentation', 'question-answering', "
            "'machine translation', etc. These task names align with common task categories used on the "
            "Hugging Face platform."
        ),
    )

    # TODO incorporate ModalityEnum from enums.py
    modalities: Optional[List[str]] = Field(
        None,
        description=(
            "Data modalities utilized within the asset, e.g. ['text'], ['image', 'text'], ['audio', 'video']."
        ),
    )

    languages: Optional[List[str]] = Field(
        None,
        description=(
            "Human language(s) associated with the asset, using ISO-639-1/2 codes, "
            "e.g. ['en', 'de', 'fr']."
        ),
    )

    # Exists a AIoD taxonomy for this field
    bussiness_sectors: Optional[List[str]] = Field(
        None,
        description=(
            "Industry or economic sector(s) in which the asset is intended to be applied, e.g. "
            "['Manufacturing', 'Financial Services', 'Healthcare', "
            "'Agriculture', 'Public Administration']."
        ),
    )

    # Exists a taxonomy for this field
    business_problems: Optional[List[str]] = Field(
        None,
        description=(
            "The specific business problem(s) the asset aims to solve, e.g. "
            "['Predictive Maintenance', 'Fraud Detection', "
            "'Customer Churn Prevention', 'Supply-Chain Optimisation']."
        ),
    )

    # Exists a AIoD taxonomy for this field
    license: Optional[str] = Field(
        None,
        description=("License associated with the asset, e.g. 'apache-2.0', 'cc-by-4.0', 'mit'."),
    )

    # Attribute with ton of values
    frameworks: Optional[List[str]] = Field(
        None,
        description=(
            "Software frameworks, libraries or standards referenced, e.g. ['PyTorch', 'TensorFlow', 'ONNX']."
        ),
    )

    # Attribute with ton of values
    author_affiliations: Optional[List[str]] = Field(
        None,
        description=(
            "Institutional affiliations of the main authors/maintainers, e.g. "
            "['Fraunhofer IAIS', 'University College Cork']."
        ),
    )

    # TODO for publications and educational resources
    publication_type: Optional[str] = Field(
        None,
        description=(
            "Type of publication the asset is, e.g. 'book', 'chapter', 'journal_article', "
            "'conference_paper', 'preprint', 'blog'"
        ),
    )


class Dataset_AiExtractedMetadata(Base_AiExtractedMetadata):
    """
    Metadata fields that apply only to assets of type 'dataset'.
    Every attribute is optional so an agent can omit values it cannot
    infer with confidence.
    """

    datapoint_count: Optional[int] = Field(
        None,
        description=(
            "Approximate number of rows / instances / datapoints in the dataset, e.g. 120 000."
        ),
    )

    feature_count: Optional[int] = Field(
        None,
        description=("Number of features / columns per instance / datapoint, e.g. 42."),
    )

    data_formats: Optional[List[str]] = Field(
        None,
        description=(
            "One or more storage formats e.g. '.png', '.mp3', '.csv', '.json', '.parquet', etc."
        ),
    )

    # TODO incorporate LabelTypeEnum from enums.py
    label_types: Optional[List[str]] = Field(
        None,
        description=(
            "Kinds of annotations, label types  present , e.g. "
            "['single_label_class', 'numeric_scalar', 'sequence_label', "
            "'text_generation', 'bounding_box', 'segmentation_mask', "
            "'keypoints', 'pairwise_rank', 'ordered_rank']."
        ),
    )

    # TODO We need to come up with a enum for this field
    collection_methods: Optional[List[str]] = Field(
        None,
        description=(
            "How the raw data was gathered, e.g. "
            "['sensor', 'survey', 'web_scraping', 'simulation', 'synthetic_generation'"
            "'crowdsourcing', 'third_party', 'internal_systems']."
        ),
    )

    # TODO We need to come up with a enum for this field
    source_type: Optional[str] = Field(
        None,
        description=(
            "What source the data comes from, e.g. "
            "['real_world', 'synthetic', 'simulated', 'hybrid']."
        ),
    )

    # TODO We need to come up with a enum for this field
    update_frequency: Optional[str] = Field(
        None,
        description=(
            "How often the dataset is updated, e.g. "
            "['never', 'on_demand', 'daily', 'weekly', 'monthly', "
            "'quarterly', 'annually']."
        ),
    )

    dataset_size_megabytes: Optional[int] = Field(
        None,
        description=("Total compressed size of the dataset files in megabytes, e.g. 4700."),
    )

    class_count: Optional[int] = Field(
        None,
        description=("Number of unique classes / labels (for classification datasets), e.g. 10."),
    )

    geo_coverage: Optional[List[str]] = Field(
        None,
        description=(
            "List of ISO-3166 country codes or region names represented in the data, e.g. ['US', 'DE', 'CN']."
        ),
    )

    temporal_coverage_start: Optional[str] = Field(
        None,
        description=("Start date of temporal coverage in the dataset (YYYY-MM-DD format)."),
    )

    temporal_coverage_end: Optional[str] = Field(
        None,
        description=("End date of temporal coverage in the dataset (YYYY-MM-DD format)."),
    )


class Model_AiExtractedMetadata(Base_AiExtractedMetadata):
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


class EducationalResource_AiExtractedMetadata(Base_AiExtractedMetadata):
    pass


class Publication_AiExtractedMetadata(Base_AiExtractedMetadata):
    pass
