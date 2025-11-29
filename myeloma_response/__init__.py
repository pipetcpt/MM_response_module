"""
Multiple Myeloma Response Evaluator

A Python application to evaluate treatment response in multiple myeloma patients
based on SPEP, serum Kappa, serum Lambda, and UPEP values.
"""

from .parser import ExcelParser
from .classifier import PatientClassifier, PatientType
from .evaluator import ResponseEvaluator, ResponseType

__version__ = "1.0.0"
__all__ = [
    "ExcelParser",
    "PatientClassifier",
    "PatientType",
    "ResponseEvaluator",
    "ResponseType",
]
