"""LLM4AD integration utilities for ResearchClaw pipeline."""

from .algorithm_extractor import extract_proposed_algorithms
from .code_marker import mark_evolution_boundaries, mark_class_in_module
from .metric_aggregator import create_metric_aggregator
from .task_builder import build_task_package
from .evaluator_generator import generate_evaluator_code
from .comparison_reporter import generate_comparison_report, generate_summary_report

__all__ = [
    "extract_proposed_algorithms",
    "mark_evolution_boundaries",
    "mark_class_in_module",
    "create_metric_aggregator",
    "build_task_package",
    "generate_evaluator_code",
    "generate_comparison_report",
    "generate_summary_report",
]
