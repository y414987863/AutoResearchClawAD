"""
End-to-end example: LLM4AD Boost Layer integration with ResearchClaw.

This example demonstrates how Stage 13.5 enhances a TSP algorithm through
evolutionary refinement.

Setup:
    pip install researchclaw llm4ad

Run:
    python examples/llm4ad_boost_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.pipeline.runner import execute_pipeline


def create_tsp_config() -> dict:
    """Create a ResearchClaw config for TSP optimization with LLM4AD boost."""
    return {
        "project": {
            "name": "tsp-llm4ad-boost-demo",
            "mode": "full-auto",
            "profile": "mathematics_optimization",
        },
        "research": {
            "topic": "Novel ATSP algorithm with hybrid local search",
            "domains": ["optimization"],
            "quality_threshold": 4.0,
        },
        "runtime": {
            "timezone": "UTC",
            "max_parallel_tasks": 2,
            "approval_timeout_hours": 12,
            "retry_limit": 2,
        },
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "gpt-4",
            "timeout_sec": 300,
        },
        "experiment": {
            "mode": "sandbox",
            "time_budget_sec": 600,
            "max_iterations": 5,
            "metric_key": "optimality_gap",
            "metric_direction": "minimize",
            # ──────────────────────────────────────────────────────────
            # LLM4AD Boost Layer Configuration
            # ──────────────────────────────────────────────────────────
            "llm4ad_boost": {
                "enabled": True,  # ← Enable evolutionary refinement
                "fail_silently": True,
                # Flat spelling of the hot knobs; the nested `evolution:` /
                # `resources:` form is equivalent and takes precedence.
                "max_generations": 15,
                "population_size": 8,
                "time_budget_sec": 900,  # 15 minutes
            },
            "sandbox": {
                "python_path": "python",
                "gpu_required": False,
                "max_memory_mb": 4096,
            },
        },
        "literature_search": {
            "sources": ["openalex"],
            "max_results_per_query": 20,
            "inter_query_delay_sec": 1.5,
        },
        "security": {
            "hitl_required_stages": [],  # Auto-approve all gates for demo
            "allow_publish_without_approval": False,
            "redact_sensitive_logs": True,
        },
        "notifications": {
            "channel": "console",
            "on_stage_start": True,
            "on_stage_fail": True,
        },
        "openclaw_bridge": {
            "use_cron": False,
            "use_message": False,
            "use_memory": False,
        },
    }


def main():
    """Run ResearchClaw pipeline with LLM4AD Boost enabled."""
    print("=" * 80)
    print("LLM4AD Boost Layer - End-to-End Example")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("artifacts/tsp-llm4ad-demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_dict = create_tsp_config()
    config = RCConfig.from_dict(config_dict)

    print(f"Output directory: {output_dir}")
    print(f"LLM4AD Boost: {'ENABLED' if config.experiment.llm4ad_boost.enabled else 'DISABLED'}")
    print(f"Max generations: {config.experiment.llm4ad_boost.evolution['max_generations']}")
    print(f"Population size: {config.experiment.llm4ad_boost.evolution['population_size']}")
    print()

    # Execute pipeline
    print("Starting ResearchClaw pipeline...")
    print("-" * 80)

    try:
        results = execute_pipeline(
            config=config,
            output_dir=output_dir,
            from_stage=None,
            to_stage=None,
            auto_approve_gates=True,  # Auto-approve for demo
        )

        print()
        print("-" * 80)
        print("Pipeline completed!")
        print()

        # Print summary
        print("Stage Summary:")
        for result in results:
            status_icon = "✓" if result.status == "done" else "✗"
            print(f"  {status_icon} Stage {int(result.stage):02d}: {result.stage.name}")

        # Check if Stage 13.5 ran and produced improvement
        stage_13_5_dir = output_dir / "stage-13.5"
        if stage_13_5_dir.exists():
            boost_meta_path = stage_13_5_dir / "boost_meta.json"
            if boost_meta_path.exists():
                boost_meta = json.loads(boost_meta_path.read_text(encoding="utf-8"))
                print()
                print("=" * 80)
                print("LLM4AD Boost Results:")
                print("=" * 80)
                print(f"  Status: {boost_meta.get('status', 'unknown')}")
                if boost_meta.get("status") == "success":
                    print(f"  Baseline metric: {boost_meta.get('baseline_metric', 'N/A')}")
                    print(f"  Best evolved metric: {boost_meta.get('best_metric', 'N/A')}")
                    print(f"  Improvement: {boost_meta.get('improvement_pct', 0):.2f}%")
                    print(f"  Generations: {boost_meta.get('generations_completed', 'N/A')}")
                elif boost_meta.get("status") == "skipped":
                    print(f"  Reason: {boost_meta.get('reason', 'unknown')}")
        else:
            print()
            print("Stage 13.5 (LLM4AD Boost) was not executed.")

    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
