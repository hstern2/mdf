from enum import Enum


class MDFFormat(str, Enum):
    """fmt options for reading and writing mdf files"""

    csv = "csv"
    smi = "smi"
    sdf = "sdf"
    viz = "viz"


class MergeHow(str, Enum):
    left = "left"
    right = "right"
    inner = "inner"
    full = "full"


class CatHow(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"
    diagonal = "diagonal"
