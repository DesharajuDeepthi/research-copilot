import json
import sys
import time
import types
from pathlib import Path

# ragas 0.4.3 still imports langchain_community.chat_models.vertexai for an isinstance
# check, but langchain-community has since removed that submodule. Stub it out rather
# than pin an older langchain-community, which would break langgraph/langchain-openai.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _StubChatVertexAI:
    pass


_vertexai_stub.ChatVertexAI = _StubChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import Faithfulness  # noqa: E402

from research_copilot.agents import nodes  # noqa: E402
from research_copilot.agents.graph import app as corrective_app  # noqa: E402
from research_copilot.config import settings  # noqa: E402
from research_copilot.validation.citations import validate_citations  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"

_faithfulness_metric = Faithfulness(
    llm=llm_factory(
        settings.OPENAI_MODEL,
        client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        max_tokens=4096,
    )
)


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _initial_state(query: str) -> dict:
    return {
        "query": query,
        "rewritten_query": None,
        "retrieved_docs": [],
        "graded_docs": [],
        "generation": None,
        "citations": [],
        "agent_trace": [],
        "fallback_used": False,
        "grounding_passed": False,
        "citation_validated": False,
        "retry_count": 0,
    }


def run_baseline_hybrid_rag(query: str) -> dict:
    """Vanilla hybrid RAG: retrieve -> synthesize, no grading/rewrite/fallback/verification."""
    state = _initial_state(query)
    state.update(nodes.retrieve(state))
    state["graded_docs"] = state["retrieved_docs"]
    state.update(nodes.synthesize(state))
    return state


def run_corrective_rag(query: str) -> dict:
    return corrective_app.invoke(_initial_state(query), config={"recursion_limit": 25})


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0

    # The eval has hit both truncated-output and transient connection errors from the
    # OpenAI/ragas call chain; a couple of retries avoids losing an entire run to a blip.
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            result = _faithfulness_metric.score(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            return result.value
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 * (2**attempt))


def evaluate_result(question: str, result: dict) -> dict:
    contexts = [doc.get("page_content", "") for doc in result.get("graded_docs", [])]
    answer = result.get("generation") or ""
    citation_valid, _ = validate_citations(result.get("citations", []), result.get("graded_docs", []))

    return {
        "answer": answer,
        "faithfulness": score_faithfulness(question, answer, contexts),
        "citation_valid": citation_valid,
        "fallback_used": result.get("fallback_used", False),
        "retry_count": result.get("retry_count", 0),
    }


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary_row(label: str, rows: list[dict]) -> list[str]:
    baseline_faithfulness = [row["baseline"]["faithfulness"] for row in rows]
    corrective_faithfulness = [row["corrective"]["faithfulness"] for row in rows]
    baseline_citation_rate = _avg([row["baseline"]["citation_valid"] for row in rows])
    corrective_citation_rate = _avg([row["corrective"]["citation_valid"] for row in rows])
    rewrite_rate = _avg([row["corrective"]["retry_count"] > 0 for row in rows])
    fallback_rate = _avg([row["corrective"]["fallback_used"] for row in rows])

    return [
        f"### {label} ({len(rows)} questions)",
        "",
        "| Metric | Hybrid RAG (baseline) | Corrective RAG |",
        "|---|---|---|",
        f"| Avg. faithfulness | {_avg(baseline_faithfulness):.3f} | {_avg(corrective_faithfulness):.3f} |",
        f"| Citation validity rate | {baseline_citation_rate:.0%} | {corrective_citation_rate:.0%} |",
        f"| Query rewrite triggered | n/a | {rewrite_rate:.0%} |",
        f"| Live API fallback triggered | n/a | {fallback_rate:.0%} |",
        "",
    ]


def write_results(rows: list[dict]) -> None:
    in_corpus_rows = [row for row in rows if row["category"] == "in_corpus"]
    adversarial_rows = [row for row in rows if row["category"] == "adversarial"]

    lines = [
        "# Evaluation Results: Hybrid RAG vs Corrective RAG",
        "",
        f"Golden set size: {len(rows)} questions "
        f"({len(in_corpus_rows)} in-corpus, {len(adversarial_rows)} adversarial).",
        "",
        "## Summary",
        "",
        *_summary_row("Overall", rows),
        *(_summary_row("In-corpus questions", in_corpus_rows) if in_corpus_rows else []),
        *(_summary_row("Adversarial questions", adversarial_rows) if adversarial_rows else []),
        "## Per-question results",
        "",
        "| # | Category | Question | Baseline Faithfulness | Corrective Faithfulness | Baseline Citations OK | Corrective Citations OK | Fallback Used |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for i, row in enumerate(rows, start=1):
        b, c = row["baseline"], row["corrective"]
        lines.append(
            f"| {i} | {row['category']} | {row['question']} | {b['faithfulness']:.2f} | "
            f"{c['faithfulness']:.2f} | {'✓' if b['citation_valid'] else '✗'} | "
            f"{'✓' if c['citation_valid'] else '✗'} | {'✓' if c['fallback_used'] else ''} |"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {RESULTS_PATH}")


def main() -> None:
    golden_set = load_golden_set()
    rows = []

    for i, entry in enumerate(golden_set, start=1):
        question = entry["question"]
        print(f"[{i}/{len(golden_set)}] {question}")

        baseline_result = run_baseline_hybrid_rag(question)
        corrective_result = run_corrective_rag(question)

        rows.append(
            {
                "question": question,
                "category": entry.get("category", "in_corpus"),
                "baseline": evaluate_result(question, baseline_result),
                "corrective": evaluate_result(question, corrective_result),
            }
        )
        write_results(rows)  # write after every question so a crash doesn't lose progress


if __name__ == "__main__":
    main()
