"""Algorithm-evolution (LLM4AD) domain prompt adapter.

Minimal adapter: the ML prompt bank already covers the narrative stages well
for algorithm-design tasks, so the stage methods return empty blocks. The
only domain-specific overlay is a light experiment-design framing that nudges
hypothesis/experiment prose toward "evolve one function, score it against a
seed baseline" — which also improves the description the LLM4AD build phase
receives downstream.
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class AlgorithmEvolutionPromptAdapter(PromptAdapter):
    """Adapter for automatic algorithm design via LLM4AD island GA."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        design_context = (
            "This is an **automatic algorithm-design** experiment run via an "
            "island genetic algorithm (LLM4AD).\n\n"
            "Key principles:\n"
            "1. Frame the problem as evolving ONE function with a clear, typed "
            "signature and a single scalar objective.\n"
            "2. Establish a simple, correct seed algorithm as the baseline.\n"
            "3. Define a deterministic evaluator; success means the evolved "
            "best individual strictly beats the seed in the objective's "
            "direction.\n"
        )
        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                "Report best-vs-seed improvement and the fitness trajectory "
                "across generations; a single seeded run is acceptable but "
                "note the stochasticity of evolution."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Algorithm-evolution result analysis:\n"
                "- Report the best individual's score and the seed baseline.\n"
                "- Describe how the evolved algorithm differs from the seed.\n"
                "- Note convergence behaviour across generations/islands.\n"
            ),
        )

    def get_export_publish_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks()
