import json
from pathlib import Path

import typer

from health_agent.config import get_settings

app = typer.Typer(help="Health Agent — RAG-powered wellness assistant")


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index even if resources are unchanged"),
):
    """Ingest wellness resources from the resources directory."""
    from health_agent.rag.ingest import ingest_resources
    from health_agent.rag.retriever import needs_reindex

    settings = get_settings()
    if not settings.database_url.strip():
        raise typer.BadParameter("DATABASE_URL must be set before running ingest.")

    if not force and not needs_reindex(settings):
        print("Index is up to date — skipping ingestion. Use --force to rebuild.")
        return

    result = ingest_resources(settings, force=force)
    print(
        "Ingest complete: "
        f"{result.added_resources} added, "
        f"{result.updated_resources} updated, "
        f"{result.deleted_resources} deleted, "
        f"{result.chunk_rows_written} chunks written."
    )


@app.command("eval-rag")
def eval_rag(
    strategy: str = typer.Option(
        "both",
        "--strategy",
        help="Retrieval strategy to evaluate: legacy, hybrid_v2, or both.",
    ),
    mode: str = typer.Option(
        "retrieval",
        "--mode",
        help=(
            "Eval mode: retrieval, production-context, production-packed, "
            "or oracle-context."
        ),
    ),
    cases_path: Path = typer.Option(
        Path("evals/rag_context_cases.json"),
        "--cases",
        help="Oracle-context case fixture path.",
    ),
    oracle_path: Path = typer.Option(
        Path("evals/rag_context_oracle.json"),
        "--oracle",
        help="Oracle-context label fixture path.",
    ),
    include_unreviewed: bool = typer.Option(
        False,
        "--include-unreviewed",
        help="Allow draft oracle labels in oracle-context scoring.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit raw JSON instead of a human-readable summary.",
    ),
):
    """Evaluate retrieval against the committed seed question set."""
    from health_agent.rag.evaluation import evaluate_rag_strategies

    settings = get_settings()
    if not settings.database_url.strip():
        raise typer.BadParameter("DATABASE_URL must be set before running RAG eval.")

    if strategy == "both":
        strategies = ["legacy", "hybrid_v2"]
    elif strategy in {"legacy", "hybrid_v2"}:
        strategies = [strategy]
    else:
        raise typer.BadParameter("Strategy must be one of: legacy, hybrid_v2, both.")

    if mode not in {
        "retrieval",
        "production-context",
        "production-packed",
        "oracle-context",
    }:
        raise typer.BadParameter(
            "Mode must be one of: retrieval, production-context, production-packed, "
            "oracle-context."
        )

    if mode == "oracle-context":
        from health_agent.rag.oracle_eval import evaluate_oracle_context_strategies

        results = evaluate_oracle_context_strategies(
            settings,
            strategies,
            cases_path=cases_path,
            oracle_path=oracle_path,
            include_unreviewed=include_unreviewed,
        )
    else:
        results = evaluate_rag_strategies(settings, strategies, mode=mode)
    if json_output:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        summary = result["summary"]
        if result["mode"] == "oracle-context":
            print(
                f"{result['strategy']} ({result['mode']}): "
                f"sufficient_context_rate={summary['sufficient_context_rate']:.2f} "
                f"claim_coverage={summary['mean_required_claim_coverage']:.2f} "
                f"first_support_mrr={summary['mean_first_support_mrr']:.2f} "
                f"gold_recall={summary['mean_gold_context_recall']:.2f} "
                f"noise={summary['mean_context_noise_rate']:.2f} "
                f"mean_latency_ms={summary['mean_latency_ms']:.1f}"
            )
            for case in result["cases"]:
                if not case["evaluable"]:
                    print(f"  skipped question={case['question']}")
                    continue
                top_sources = ", ".join(case["top_sources"])
                supporting_sources = ", ".join(case["supporting_sources"]) or "-"
                sufficient = "yes" if case["sufficient_context"] else "no"
                print(
                    f"  sufficient={sufficient} "
                    f"claims={case['supported_claims']}/{case['required_claims']} "
                    f"coverage={case['required_claim_coverage']:.2f} "
                    f"noise={case['context_noise_rate']:.2f} "
                    f"question={case['question']}"
                )
                print(f"    supporting_sources={supporting_sources}")
                print(f"    top_sources={top_sources}")
            continue

        print(
            f"{result['strategy']} ({result['mode']}): "
            f"hit_rate={summary['hit_rate']:.2f} "
            f"mrr={summary['mrr']:.2f} "
            f"unique_source_mrr={summary['unique_source_mrr']:.2f} "
            f"coverage={summary['mean_expected_source_coverage']:.2f} "
            f"mean_latency_ms={summary['mean_latency_ms']:.1f}"
        )
        for case in result["cases"]:
            rank = case["rank"] if case["rank"] is not None else "-"
            unique_rank = (
                case["unique_source_rank"]
                if case["unique_source_rank"] is not None
                else "-"
            )
            top_sources = ", ".join(case["top_sources"])
            found_sources = ", ".join(case["expected_sources_found"]) or "-"
            print(
                f"  rank={rank} unique_source_rank={unique_rank} "
                f"coverage={case['expected_source_coverage']:.2f} "
                f"question={case['question']}"
            )
            print(f"    expected_found={found_sources}")
            print(f"    top_sources={top_sources}")


@app.command("build-rag-oracle")
def build_rag_oracle(
    cases_path: Path = typer.Option(
        Path("evals/rag_context_cases.json"),
        "--cases",
        help="RAG context eval case fixture path.",
    ),
    output_path: Path = typer.Option(
        Path("evals/rag_context_oracle.json"),
        "--output",
        help="Oracle label output path.",
    ),
    case_id: list[str] | None = typer.Option(
        None,
        "--case-id",
        help="Limit oracle generation to specific case ids. Can be passed multiple times.",
    ),
    max_candidates: int = typer.Option(
        80,
        "--max-candidates",
        help="Maximum broad candidate chunks to send to the oracle judge per case.",
    ),
    refresh_reviewed: bool = typer.Option(
        False,
        "--refresh-reviewed",
        help="Regenerate cases that already contain reviewed oracle labels.",
    ),
):
    """Build draft claim-level RAG context oracle labels."""
    from health_agent.rag.oracle_eval import (
        build_rag_context_oracle,
        load_context_eval_cases,
        write_context_oracle,
    )

    settings = get_settings()
    if not settings.database_url.strip():
        raise typer.BadParameter("DATABASE_URL must be set before building the oracle.")

    cases = load_context_eval_cases(cases_path)
    oracle = build_rag_context_oracle(
        settings,
        cases=cases,
        cases_path=cases_path,
        existing_oracle_path=output_path,
        case_ids=case_id,
        max_candidates=max_candidates,
        preserve_reviewed=not refresh_reviewed,
    )
    write_context_oracle(oracle, output_path)
    claim_count = sum(len(case.claims) for case in oracle.cases)
    print(
        "Oracle draft written: "
        f"{output_path} "
        f"({len(oracle.cases)} cases, {claim_count} claims). "
        "Review labels and set reviewed=true before using default scoring."
    )


@app.command()
def chat():
    """Start an interactive chat session with the health agent."""
    from langchain_core.messages import HumanMessage

    from health_agent.graph import build_graph
    from health_agent.rag.retriever import needs_reindex

    settings = get_settings()

    if not settings.database_url.strip():
        print(
            "DATABASE_URL is not configured. RAG retrieval will be unavailable until Postgres "
            "is configured and ingested.\n"
        )
    elif needs_reindex(settings):
        print(
            "RAG index is missing or stale. Run `health-agent ingest` to rebuild it. "
            "Continuing without automatic ingestion.\n"
        )

    graph = build_graph(settings)
    messages = []

    print("Health Agent (Grok search + RAG + Claude synthesis) — type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": messages})
        messages = result["messages"]

        ai_message = messages[-1]
        print(f"\nAssistant: {ai_message.content}\n")


if __name__ == "__main__":
    app()
