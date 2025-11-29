from enum import Enum
from typing import Annotated, Literal, TypeAlias
from types import UnionType
from typing import Type, Union, get_args, get_origin
from pydantic import Field


class AnnotationOperations:
    @classmethod
    def strip_optional_and_list_types(cls, annotation: Type) -> Type:
        """Strip annotation of the form Optional[List[TYPE]] to TYPE"""
        return cls.strip_list_type(cls.strip_optional_type(annotation))

    @classmethod
    def is_optional_type(cls, annotation: Type) -> bool:
        if get_origin(annotation) is Union or get_origin(annotation) is UnionType:
            return type(None) in get_args(annotation)
        return False

    @classmethod
    def is_list_type(cls, annotation: Type) -> bool:
        return get_origin(annotation) is list

    @classmethod
    def strip_optional_type(cls, annotation: Type) -> Type:
        if cls.is_optional_type(annotation):
            return next(arg for arg in get_args(annotation) if arg is not type(None))
        return annotation

    @classmethod
    def strip_list_type(cls, annotation: Type) -> Type:
        if cls.is_list_type(annotation):
            return get_args(annotation)[0]
        return annotation


###### TYPES

PrimitiveTypes: TypeAlias = int | float | str | bool
ComparisonOperator = Literal["<", ">", "<=", ">=", "==", "!="]
LogicalOperator = Literal["AND", "OR"]

Varchar16 = Annotated[str, Field(max_length=16)]
Varchar32 = Annotated[str, Field(max_length=32)]
Varchar64 = Annotated[str, Field(max_length=64)]
Varchar128 = Annotated[str, Field(max_length=128)]
Varchar256 = Annotated[str, Field(max_length=256)]

CountryCode = Annotated[str, Field(description="ISO-3166 country code", pattern=r"^[A-Z]{2}$")]
LanguageCode = Annotated[str, Field(description="ISO-639-1 language code", pattern=r"^[a-z]{2}$")]
DateString = Annotated[
    str, Field(description="Date in the format: YYYY-MM-DD", pattern=r"^\d{4}-\d{2}-\d{2}")
]
DatePublished = Annotated[
    str,
    Field(
        description="Date in the format: YYYY-MM-DDTHH:MM:SS",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
    ),
]
FileExtension = Annotated[
    str, Field(description="File extension", pattern=r"^\..*$", max_length=16)
]


###### ENUMS


class PlatformEnum(str, Enum):
    AIOD = "aiod"
    EXAMPLE = "example"
    OPENML = "openml"
    HUGGINGFACE = "huggingface"
    ZENODO = "zenodo"
    AI4EXPERIMENTS = "ai4experiments"
    STAIRWAI = "stairwai"
    BONSEYES = "bonseyes"
    AIDA_CMS = "aida_cms"
    ROBOTICS4EU = "robotics4eu"
    ADRA_E = "adra_e"
    AIBUILDER = "aibuilder"
    AI4EUROPE_CMS = "ai4europe_cms"


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


class SourceTypeEnum(str, Enum):
    """Whether the data comes from real-world observations, synthetic generation, simulations, or a mix."""

    REAL_WORLD = "real_world"
    SYNTHETIC = "synthetic"
    SIMULATED = "simulated"
    HYBRID = "hybrid"


class NewsCategory(str, Enum):
    """The thematic category describing the focus of the publication or announcement."""

    BUSINESS = "Business"
    DEVELOPMENT = "Development"
    SOCIETY = "Society"
    RESEARCH = "Research"


class EducationalCompetency(str, Enum):
    """The expected learner proficiency level targeted by the resource."""

    ADVANCED = "advanced"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"


class LearningMode(str, Enum):
    """The primary delivery format through which the learning resource is accessed."""

    HYBRID = "Hybrid"
    OFFLINE = "Offline"
    ONLINE = "Online"


class EducationalPace(str, Enum):
    """Indicates the expected pacing or scheduling structure of the learning resource."""

    FULL_TIME = "full-time"
    SCHEDULED = "scheduled"
    SELF_PACED = "self-paced"


class EducationalTargetAudience(str, Enum):
    """Specifies the intended audience or learner group for the resource."""

    PROFESSIONALS = "professionals"
    WORKING_PROFESSIONALS = "working professionals"
    STUDENTS_HIGHER_EDU = "students in higher education"
    TEACHERS_SECONDARY = "teachers in secondary school"


class EducationalResourceType(str, Enum):
    """Classifies the format or medium through which the educational content is delivered."""

    BOOK = "book"
    DATASET = "dataset"
    LECTURES_PRESENTATIONS = "lectures-presentations"
    LECTURES_PRESENTATIONS_VIDEO = "lectures-presentations video recordings"
    PAPER = "paper"
    PRESENTATION = "presentation"
    VIDEO_RECORDINGS = "video recordings"
