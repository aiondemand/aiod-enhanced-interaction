from enum import Enum
from typing import Annotated

from pydantic import Field

###### TYPES

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
FileExtension = Annotated[str, Field(description="File extension", pattern=r"^\..*$")]


###### ENUMS


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


# PUBLICATIONS / EDUCATIONAL RESOURCES

# TODO LATER
# TODO make Enums lowercase


class EducationalCompetency(str, Enum):
    """The expected learner proficiency level targeted by the resource."""

    ADVANCED = "advanced"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"


class EducationalLevel(str, Enum):
    """The formal education stage for which the resource is intended."""

    DOCTORAL_OR_EQUIVALENT = "Doctoral or equivalent level"
    EARLY_CHILDHOOD = "Early childhood education"
    LOWER_SECONDARY = "Lower secondary education"
    NOT_ELSEWHERE_CLASSIFIED = "Not elsewhere classified"
    POST_SECONDARY_NON_TERTIARY = "Post-secondary non-tertiary education"
    PRIMARY = "Primary education"
    SHORT_CYCLE_TERTIARY = "Short-cycle tertiary education"
    UPPER_SECONDARY = "Upper secondary education"
    BACHELOR_OR_EQUIVALENT = "Bachelor’s or equivalent level"
    MASTER_OR_EQUIVALENT = "Master’s or equivalent level"


class LearningMode(str, Enum):
    """The primary delivery format through which the learning resource is accessed."""

    HYBRID = "Hybrid"
    OFFLINE = "Offline"
    ONLINE = "Online"


class NewsCategory(str, Enum):
    """The thematic category describing the focus of the publication or announcement."""

    BUSINESS = "Business"
    DEVELOPMENT = "Development"
    SOCIETY = "Society"
    RESEARCH = "Research"


class PublicationType(str, Enum):
    """The document type or format used to publish or present the work."""

    ARTICLE = "Article"
    BOOK = "Book"
    ABSTRACT = "Abstract"
    CHAPTER = "Chapter"
    CONFERENCE_POSTER = "Conference Poster"
    DATASHEET = "datasheet"
    DEMO_PAPER = "Demo Paper"
    EDITORIAL = "Editorial"
    INSTRUCTION_MANUAL = "Instruction Manual"
    LECTURE_NOTES = "Lecture Notes"
    MANUSCRIPT = "Manuscript"
    PERIODICAL_ISSUE = "Periodical Issue"
    PERIODICAL_VOLUME = "Periodical Volume"
    POLICY_DOCUMENT = "Policy Document"
    POSTER_PAPER = "Poster Paper"
    PROCEEDINGS_PAPER = "Proceedings Paper"
    REPORT_DOCUMENT = "Report Document"
    SPECIFICATION_DOCUMENT = "Specification Document"
    WEB_CONTENT = "Web Content"


class EducationalPace(str, Enum):
    """Indicates the expected pacing or scheduling structure of the learning resource."""

    FULL_TIME = "full-time"
    SCHEDULED = "scheduled"
    SELF_PACED = "self-paced"


class EducationalPrerequisite(str, Enum):
    """Specifies prior knowledge or background expected before using the resource."""

    BASIC_MATH_FOUNDATION = "basic understanding of math"
    BASIC_LINEAR_ALGEBRA_CALCULUS_PROB_STATS = (
        "a basic understanding of linear algebra, basic calculus, probability and statistics."
    )
    UNDERGRAD_STATISTICS = "undergraduate knowledge of statistics"
    BASIC_AI_OR_CS_RECOMMENDED = (
        "basic knowledge of ai or computer science is recommended. open to bsc, msc, phd students, "
        "researchers, and professionals with interest in ai. programming or data analysis experience is useful but not required."
    )
    BASIC_AI_OR_DIGITAL_LAW = "basic knowledge of ai or digital law"
    DATA_PROFESSIONAL_1_YEAR = (
        "a data professional with 1+ year of experience in a coding-based role"
    )
    GRAD_LINEAR_ALGEBRA = "graduate knowledge of linear algebra"
    FAMILIARITY_SUPERVISED_CONSTRAINED_OPT = (
        "familiarity with supervised learning and with some method for constrained optimization"
    )
    PROB_STATS_LINEAR_ALGEBRA_CALCULUS_LOGIC_PYTHON = "probability and statistics, linear algebra, basic calculus, symbolic logic. programming in python is useful."
    ML_WITH_NEURAL_NETWORKS = "machine learning with neural networks"
    COMPLIANCE_OR_ETHICAL_AI_INTEREST = "interest in compliance or ethical ai"
    INTERNATIONAL = "international"
    RECOMMENDED_BSC_ENGINEERING = "it is recommended to have at least bsc in engineering, some content might require more advanced skills and competences."
    NONE = "none"


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
    LECTURES_PRESENTATIONS_VIDEO = "lectures-presentationsvideo recordings"
    PAPER = "paper"
    PRESENTATION = "presentation"
    VIDEO_RECORDINGS = "video recordings"
