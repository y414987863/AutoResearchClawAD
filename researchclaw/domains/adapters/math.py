"""Mathematics domain prompt adapter."""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class MathPromptAdapter(PromptAdapter):
    """Adapter for numerical mathematics and optimization domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        paradigm = domain.experiment_paradigm

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance or (
                "Numerical methods are typically fast.\n"
                "Use 5-8 refinement levels for convergence plots.\n"
                "Step sizes: geometric sequence (h, h/2, h/4, ...)"
            ),
            dataset_guidance=domain.dataset_guidance or (
                "Use standard test problems with known solutions:\n"
                "- ODE: Lotka-Volterra, Van der Pol, stiff systems\n"
                "- Quadrature: smooth, oscillatory, singular integrands\n"
                "- Linear algebra: Hilbert matrix, tridiagonal\n"
                "- Do NOT download external datasets"
            ),
            code_generation_hints=domain.code_generation_hints or self._hints(paradigm),
            output_format_guidance=self._output_format(paradigm),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        # Specialized guidance for optimization domains
        if domain.domain_id in ("mathematics_optimization", "mathematics_numerical"):
            design_context = (
                f"This is a **{domain.display_name}** experiment.\n\n"
                "## Optimization Algorithm Runtime Estimation\n"
                "- A single optimization run (1 seed × 1 test function × 1 algorithm) with budget B evaluations "
                "typically takes **B/100 to B/50 seconds** (depending on function complexity)\n"
                "- Example: B=2000 (200*d for d=10) takes ~20-40 seconds per run\n"
                "- Example: B=4000 (200*d for d=20) takes ~40-80 seconds per run\n"
                "- **CRITICAL**: Keep scale small to fit time budget:\n"
                "  * Use 2-3 seeds (not 5-10) for tight budgets\n"
                "  * Use 2-4 test functions (not 8-10)\n"
                "  * Use 3-5 optimizers total (baselines + proposed + ablations)\n"
                "  * Total runs should be ≤ time_budget_sec / 30\n\n"
                "Focus on:\n"
                "1. Correctness (verify against known optima)\n"
                "2. Convergence quality (final objective value)\n"
                "3. Efficiency (wall time, overhead fraction)\n"
                "4. Test functions: Use standard benchmarks (Rosenbrock, Rastrigin, Ackley, Sphere)\n"
            )
        else:
            design_context = (
                f"This is a **{domain.display_name}** experiment.\n"
                "Focus on:\n"
                "1. Correctness (verify against known solutions)\n"
                "2. Convergence order (expected vs observed)\n"
                "3. Efficiency (operations count, wall time)\n"
            )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance="Use paired statistical tests (Wilcoxon or t-test) for optimizer comparison across seeds.",
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=(
                "Numerical methods analysis:\n"
                "- Convergence: fit log(error) vs log(h)\n"
                "- Stability: check for growth in error over long runs\n"
                "- Efficiency: compare accuracy per unit computation"
            ),
        )

    def _hints(self, paradigm: str) -> str:
        if paradigm == "convergence":
            return (
                "Numerical methods convergence study:\n"
                "1. Implement methods from scratch (not just scipy wrappers)\n"
                "2. Use test problems with KNOWN exact solutions\n"
                "3. Run at 5+ refinement levels\n"
                "4. Compute error: ||u_h - u_exact||_2\n"
                "5. Report convergence order: p = log(e_h / e_{h/2}) / log(2)\n"
                "6. Output results.json with convergence data"
            )
        return (
            "Numerical/optimization code:\n"
            "1. Implement algorithms from scratch\n"
            "2. Test on standard benchmark functions\n"
            "3. Compare accuracy and efficiency\n"
            "4. Output results.json"
        )

    def _output_format(self, paradigm: str) -> str:
        if paradigm == "convergence":
            return (
                "Output convergence results to results.json:\n"
                '{"convergence": {"method": [{"h": 0.1, "error": 0.05}, ...]}}'
            )
        return (
            "Output results to results.json:\n"
            '{"conditions": {"optimizer": {"iterations": 100, "final_value": 0.001}}}'
        )
