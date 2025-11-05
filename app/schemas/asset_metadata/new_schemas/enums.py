from enum import Enum

# ASSET INVARIANT


class ModalityEnum(str, Enum):
    """Kinds of data the asset handles."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    THREE_D = "3d"


# PUBLICATION


class PublicationTypeEnum(str, Enum):
    """Type of document when the asset is a publication."""

    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    WHITEPAPER = "whitepaper"
    TECH_REPORT = "technical_report"
    THESIS = "thesis"
    PATENT = "patent"
    BLOG_POST = "blog_post"
    WORKSHOP_PAPER = "workshop_paper"
    POSTER = "poster"
    TUTORIAL = "tutorial"
    SURVEY = "survey"
    DEMO_PAPER = "demo_paper"
    DISSERTATION = "dissertation"
    MAGAZINE_ARTICLE = "magazine_article"


# DATASET


class LabelTypeEnum(str, Enum):
    """Kinds of annotations / targets present in the dataset."""

    SINGLE_LABEL_CLASS = "single_label_class"
    MULTI_LABEL_CLASS = "multi_label_class"
    NUMERIC_SCALAR = "numeric_scalar"
    SEQUENCE_LABEL = "sequence_label"
    SPAN = "span"
    TEXT_GENERATION = "text_generation"
    BOUNDING_BOX = "bounding_box"
    SEGMENTATION_MASK = "segmentation_mask"
    KEYPOINTS = "keypoints"
    PAIRWISE_RANK = "pairwise_rank"
    ORDERED_RANK = "ordered_rank"


class CollectionMethodEnum(str, Enum):
    """How the raw data was gathered."""

    SENSOR = "sensor"
    SURVEY = "survey"
    WEB_SCRAPING = "web_scraping"
    CROWDSOURCING = "crowdsourcing"
    SIMULATION = "simulation"
    SYNTHETIC_GENERATION = "synthetic_generation"
    THIRD_PARTY = "third_party"
    INTERNAL_SYSTEM = "internal_system"
    API = "api"


class SourceTypeEnum(str, Enum):
    """Whether the data comes from real-world observations, synthetic generation, simulations, or a mix."""

    REAL_WORLD = "real_world"
    SYNTHETIC = "synthetic"
    SIMULATED = "simulated"
    HYBRID = "hybrid"


class UpdateFrequencyEnum(str, Enum):
    """How often the dataset is updated."""

    NEVER = "never"
    ON_DEMAND = "on_demand"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


# MODEL


class ArchitectureFamilyEnum(str, Enum):
    """High-level neural architecture families."""

    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GNN = "gnn"
    DIFFUSION = "diffusion"
    MLP = "mlp"
    GAN = "gan"
    AUTOENCODER = "autoencoder"
    DECISION_TREES = "decision_trees"
    RANDOM_FOREST = "random_forest"
    BOOSTED_TREES = "boosted_trees"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    LINEAR = "linear"
    KNN = "knn"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"


class QuantizationLevelEnum(str, Enum):
    """Weight precision / quantisation setting."""

    NONE = "none"
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


class TrainingTechniqueEnum(str, Enum):
    """High-level training or post-training methods applied."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    SELF_SUPERVISED = "self_supervised"
    WEAK_SUPERVISION = "weak_supervision"
    FEW_SHOT_LEARNING = "few_shot_learning"
    ZERO_SHOT_LEARNING = "zero_shot_learning"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    FEDERATED_LEARNING = "federated_learning"
    MULTI_TASK = "multi_task"
    CONTRASTIVE = "contrastive"
    FINE_TUNING = "fine_tuning"
    PEFT = "peft"
    INSTRUCTION_FINE_TUNING = "instruction_fine_tuning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    RLHF = "rlhf"
    RLVR = "rlvr"
    DPO = "dpo"
    DISTILLATION = "distillation"
    PRUNING = "pruning"
