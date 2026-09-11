"""Unit tests for the ``evolve_scope`` classification contract.

The scope filter selects which algorithms get evolved from stage-10's
``algorithms_classification.json``. That file is a *category* classification only
when it maps each algorithm to a proposed/baseline/ablation role; real products
also ship *method-family groupings* whose values name the algorithm set rather
than a role. Flattening those into ``{algo: family}`` made a ``{"categories":
["proposed"]}`` run match every algorithm against the literal string
``"proposed"``, get the empty set, and silently skip evolution.

These tests pin the reader's contract: the two canonical shapes decode to a
per-algorithm role mapping, and every non-canonical grouping makes it return
``None`` so ``filter_by_scope`` fails closed instead of classifying nothing.

The later sections cover the other half — how a role is assigned by the
classifier (one LLM call over the plan and the directory names, validated to
cover exactly the directories that exist), and that ``filter_by_scope`` never
falls back to "evolve everything" when it has no classification to work from.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.pipeline.llm4ad_utils.classification import (
    ABLATION,
    BASELINE,
    PROPOSED,
    classify_algorithms,
    filter_by_scope,
    read_classification,
    write_classification,
)

_read_algorithms_classification = read_classification


def _exp_with(cls_text: str) -> Path:
    import tempfile

    exp = Path(tempfile.mkdtemp()) / "e"
    (exp / "algorithms").mkdir(parents=True)
    (exp / "algorithms_classification.json").write_text(cls_text, encoding="utf-8")
    return exp


# --- canonical shapes decode to a role mapping ------------------------------

def test_canonical_wrapper_shape():
    exp = _exp_with(
        '{"classification": {"nelder_mead": "proposed", "random": "baseline", "gp": "ablation"}}'
    )
    assert _read_algorithms_classification(exp) == {
        "nelder_mead": "proposed", "random": "baseline", "gp": "ablation",
    }


def test_flat_top_level_scalar_shape():
    exp = _exp_with('{"nelder_mead": "proposed", "baseline_solver": "baseline"}')
    assert _read_algorithms_classification(exp) == {
        "nelder_mead": "proposed", "baseline_solver": "baseline",
    }


# --- non-canonical groupings return None (fail-closed) -----------------------

def test_family_grouping_dict_of_lists():
    """ml25: ``{"reservoir": ["esn"], "mlp": ["mlp"], ...}`` — the VALUES name
    the algorithm set, not a proposed/baseline/ablation role."""
    exp = _exp_with('{"reservoir": ["esn"], "gaussian_process": ["gp"], "baseline": ["persistence"]}')
    assert _read_algorithms_classification(exp) is None


def test_family_grouping_wrapper():
    """ml25: ``{"forecasting": {esn: "reservoir_computing", ...}}`` nested under a
    family name."""
    exp = _exp_with('{"forecasting": {"esn": "reservoir_computing", "linear_ar": "autoregressive"}}')
    assert _read_algorithms_classification(exp) is None


def test_type_grouping_dict_of_lists():
    """ml23: ``{"algorithm_types": {"pointwise": [...]}}`` — family grouping."""
    exp = _exp_with('{"algorithm_types": {"pointwise": ["pointwise_linear", "pointwise_ridge"]}}')
    assert _read_algorithms_classification(exp) is None


# --- invented / non-whitelisted categories are dropped ----------------------

def test_invented_category_values_dropped():
    """A classifier that invents a role (``"sota"``) instead of one of the three
    whitelisted categories is unusable; if nothing valid remains, None."""
    exp = _exp_with('{"nelder_mead": "sota", "gp": "custom"}')
    assert _read_algorithms_classification(exp) is None


def test_invented_category_mixed_with_valid_keeps_valid():
    exp = _exp_with('{"nelder_mead": "sota", "gp": "proposed"}')
    assert _read_algorithms_classification(exp) == {"gp": "proposed"}


# --- classifying: one LLM call over plan + directory names -------------------

class _FakeLLM:
    """Replies with queued JSON payloads; records the prompts it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def _tree(names: list[str]) -> Path:
    import tempfile

    exp = Path(tempfile.mkdtemp()) / "e"
    for n in names:
        d = exp / "algorithms" / n
        d.mkdir(parents=True)
        (d / f"{n}.py").write_text("", encoding="utf-8")
    return exp


def _reply(mapping: dict[str, str]) -> str:
    return json.dumps({"classification": mapping})


def test_model_assigns_every_discovered_role():
    exp = _tree(["cma_es", "random_search", "cma_es_no_floor"])
    llm = _FakeLLM(_reply({
        "cma_es": "proposed",
        "random_search": "baseline",
        "cma_es_no_floor": "ablation",
    }))
    got = classify_algorithms(exp, "proposed: cma_es\nbaselines: random_search", llm)
    assert got == {
        "cma_es": PROPOSED,
        "random_search": BASELINE,
        "cma_es_no_floor": ABLATION,
    }


def test_classification_is_written_in_canonical_form():
    """The file must exist and be readable — a missing file fails the scope closed."""
    exp = _tree(["cma_es", "random_search"])
    llm = _FakeLLM(_reply({"cma_es": "proposed", "random_search": "baseline"}))
    classify_algorithms(exp, "plan text", llm)
    assert _read_algorithms_classification(exp) == {
        "cma_es": PROPOSED, "random_search": BASELINE,
    }
    on_disk = json.loads((exp / "algorithms_classification.json").read_text(encoding="utf-8"))
    assert on_disk["source"] == "llm"
    assert "generated" in on_disk and "signature" in on_disk


def test_omitted_algorithm_defaults_to_proposed():
    """A missing role is what empties a scope, so it can never stay missing."""
    exp = _tree(["cma_es", "random_search"])
    got = classify_algorithms(exp, "plan", _FakeLLM(_reply({"random_search": "baseline"})))
    assert got == {"cma_es": PROPOSED, "random_search": BASELINE}


def test_invented_directory_is_ignored():
    exp = _tree(["cma_es"])
    got = classify_algorithms(exp, "plan", _FakeLLM(_reply({
        "cma_es": "proposed", "ghost_algo": "baseline",
    })))
    assert got == {"cma_es": PROPOSED}


def test_non_canonical_label_falls_back_to_default():
    exp = _tree(["cma_es"])
    got = classify_algorithms(exp, "plan", _FakeLLM(_reply({"cma_es": "sota"})))
    assert got == {"cma_es": PROPOSED}


def test_synonym_labels_are_accepted():
    exp = _tree(["cma_es", "random_search"])
    got = classify_algorithms(exp, "plan", _FakeLLM(_reply({
        "cma_es": "ours", "random_search": "reference",
    })))
    assert got == {"cma_es": PROPOSED, "random_search": BASELINE}


def test_unusable_reply_is_retried_then_used():
    exp = _tree(["cma_es"])
    llm = _FakeLLM("I think cma_es is the proposed method.", _reply({"cma_es": "proposed"}))
    assert classify_algorithms(exp, "plan", llm) == {"cma_es": PROPOSED}
    assert len(llm.prompts) == 2


def test_total_failure_still_yields_a_complete_mapping():
    """Two bad replies must not leave the file unwritten: every run that has an
    algorithm tree gets a usable classification, or a category scope evolves
    nothing."""
    exp = _tree(["cma_es", "random_search"])
    got = classify_algorithms(exp, "plan", _FakeLLM("no json", "still no json"))
    assert got == {"cma_es": PROPOSED, "random_search": PROPOSED}
    assert _read_algorithms_classification(exp) == got


def test_no_llm_classifies_everything_as_proposed():
    exp = _tree(["cma_es", "random_search"])
    assert classify_algorithms(exp, "plan", None) == {
        "cma_es": PROPOSED, "random_search": PROPOSED,
    }


def test_no_algorithm_tree_returns_empty():
    import tempfile

    exp = Path(tempfile.mkdtemp()) / "e"
    exp.mkdir(parents=True)
    assert classify_algorithms(exp, "plan", _FakeLLM(_reply({}))) == {}


def test_prompt_carries_the_plan_and_the_names():
    exp = _tree(["cma_es", "random_search"])
    llm = _FakeLLM(_reply({"cma_es": "proposed", "random_search": "baseline"}))
    classify_algorithms(exp, "PROPOSED METHOD: cma_es with a covariance floor", llm)
    sent = llm.prompts[0]
    assert "cma_es with a covariance floor" in sent
    assert "- cma_es" in sent and "- random_search" in sent
    assert "proposed" in sent and "baseline" in sent and "ablation" in sent


def test_unchanged_tree_can_be_reused():
    exp = _tree(["cma_es"])
    first = classify_algorithms(exp, "plan", _FakeLLM(_reply({"cma_es": "baseline"})))
    llm = _FakeLLM(_reply({"cma_es": "proposed"}))
    again = classify_algorithms(exp, "plan", llm, reuse_existing=True)
    assert again == first == {"cma_es": BASELINE}
    assert llm.prompts == []


# --- filtering fails closed --------------------------------------------------

def _exp_tree(names: list[str]) -> Path:
    return _tree(names)


def test_category_scope_without_classification_evolves_nothing():
    """No readable classification -> [] and a warning. Falling back to the full
    list here would evolve the baselines, which is the one outcome scoping
    exists to prevent, and it would do so invisibly."""
    exp = _exp_tree(["cma_es", "random_search"])
    algo_root = exp / "algorithms"
    algos = [(d.name, d / f"{d.name}.py") for d in sorted(algo_root.iterdir())]
    assert filter_by_scope(algos, exp, {"categories": ["proposed"]}) == []


def test_category_scope_selects_only_matching_roles():
    exp = _exp_tree(["cma_es", "random_search"])
    write_classification(
        exp, {"cma_es": PROPOSED, "random_search": BASELINE}, source="test"
    )
    algo_root = exp / "algorithms"
    algos = [(d.name, d / f"{d.name}.py") for d in sorted(algo_root.iterdir())]
    kept = filter_by_scope(algos, exp, {"categories": ["proposed"]})
    assert [a for a, _ in kept] == ["cma_es"]


def test_empty_scope_keeps_everything():
    exp = _exp_tree(["cma_es", "random_search"])
    algo_root = exp / "algorithms"
    algos = [(d.name, d / f"{d.name}.py") for d in sorted(algo_root.iterdir())]
    assert filter_by_scope(algos, exp, {}) == algos
    assert filter_by_scope(algos, exp, None) == algos


def test_unrecognised_category_is_not_treated_as_empty_scope():
    """A typo'd category selects nothing; it must not degrade to "evolve all"."""
    exp = _exp_tree(["cma_es", "random_search"])
    algo_root = exp / "algorithms"
    algos = [(d.name, d / f"{d.name}.py") for d in sorted(algo_root.iterdir())]
    assert filter_by_scope(algos, exp, {"categories": ["proposedd"]}) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
