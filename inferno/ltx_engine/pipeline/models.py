from enum import Enum

class ConditioningMethod(Enum):
    UNCONDITIONAL = "unconditional"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    FIRST_AND_LAST_FRAME = "first_and_last_frame"
