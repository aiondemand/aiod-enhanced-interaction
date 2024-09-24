from enum import Enum


class QueryStatus(Enum):
    QUEUED = "Queued"
    IN_PROGESS = "In_progress"
    COMPLETED = "Completed"
