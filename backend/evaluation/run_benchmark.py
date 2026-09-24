"""
RAG reliability benchmark runner.

Measures the real pipeline — pdfplumber + ClaudeExtractor for ingestion,
`ChromaVectorStore.search()` for retrieval, `run_chat()` for generation —
against `benchmark.json`, and writes the measured numbers to
`frontend/src/data/rag_run.json`, which `MetricsPage.tsx` imports.

Run it from anywhere (every path is resolved from `__file__`):

    backend/venv/bin/python3 backend/evaluation/run_benchmark.py
    backend/venv/bin/python3 backend/evaluation/run_benchmark.py --rebuild-corpus

Deliberately outside `backend/app/`, and therefore outside the scope of
`mypy --strict app/` and of the pytest suite: this is a one-off measurement
tool, not an application feature. See RAPPORT.md.

ISOLATION. The user's real data (`backend/data/infopay.db`,
`backend/data/chroma_data/`) is never opened. This script builds its own
SQLite file and its own Chroma directory under `backend/evaluation/.corpus/`
(gitignored) and rebinds the two module-level globals through which the
application would otherwise reach the real ones:

  * `app.services.analytics.engine` — `analytics.py` does
    `from app.db import engine`, so the name lives in *its* module globals
    and is resolved at call time by `_load_dataframe()`. Rebinding
    `app.db.engine` would not help; rebinding this one does.
  * `app.agent.tools.get_vector_store` — same reasoning: `tools.py`
    imported the name, and calls it inside the tool body.

The corpus is reused across runs unless `--rebuild-corpus` is passed, so a
second run costs zero extraction calls.
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

# ANTHROPIC_API_KEY, needed the moment `app.agent.graph` is imported (it
# builds its `ChatAnthropic` at module level).
load_dotenv(BACKEND_DIR / ".env")

from langchain_anthropic import ChatAnthropic  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

import app.agent.graph as graph_mod  # noqa: E402
import app.agent.tools as tools_mod  # noqa: E402
import app.services.analytics as analytics_mod  # noqa: E402
from app.models.payslip import Payslip  # noqa: E402
from app.services.extraction import (  # noqa: E402
    EXTRACTION_MODEL,
    ClaudeExtractor,
    extract_text_from_pdf,
)
from app.services.vectorstore import (  # noqa: E402
    SIMILARITY_DISTANCE_THRESHOLD,
    ChromaVectorStore,
    _chunk_text,
)

EVALUATION_DIR = Path(__file__).resolve().parent
BENCHMARK_FILE = EVALUATION_DIR / "benchmark.json"
CORPUS_DIR = EVALUATION_DIR / ".corpus"
EXAMPLES_DIR = REPO_ROOT / "frontend" / "public" / "exemples"
OUTPUT_FILE = REPO_ROOT / "frontend" / "src" / "data" / "rag_run.json"

# BatiRenov is a known extraction failure (its layout defeats the extractor;
# it ships in `public/exemples/` precisely to demonstrate that limitation, see
# HelpPage's "Exemples de PDF non acceptés"). The real app never indexes it
# either, so indexing it here would measure something the product doesn't do.
# Case U12 asks about it on purpose.
EXCLUDED_PDFS = ("bulletin_batirenov_06_2022.pdf",)

# An abstention is read off the wording of the reply, because this
# architecture exposes no structured "I don't know" signal: `run_chat()`
# returns free text plus optional sources, and `sources is None` can't serve
# as the signal (a general-knowledge answer — which the system prompt asks for
# on off-topic questions — also carries no sources, while being the opposite
# of an abstention). Markers are matched against the reply lowercased and
# accent-stripped, so "n'apparaît pas" and "n'apparait pas" both count.
ABSTENTION_MARKERS = (
    "aucune information",
    "aucune mention",
    "aucune ligne",
    "aucune donnee",
    "aucun element",
    "aucun bulletin",
    "aucun document",
    "pas d'information",
    "n'ai trouve",
    "ne trouve pas",
    "ne figure pas",
    "ne figurent pas",
    "n'apparait pas",
    "n'apparaissent pas",
    "ne contient pas",
    "ne contiennent pas",
    "ne mentionne pas",
    "ne mentionnent pas",
    "n'est pas mentionne",
    "ne sont pas mentionne",
    "n'est pas present",
    "ne dispose pas",
    "n'est pas importe",
    "n'a pas ete importe",
)

REPLY_EXCERPT_CHARS = 600
CHUNK_EXCERPT_CHARS = 200


def normalize(text: str) -> str:
    """Lowercase, strip accents, and unify the two apostrophes French text
    mixes (U+2019 and U+0027), so marker matching doesn't hinge on which one
    the model happened to emit."""
    folded = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def has_abstained(reply: str) -> bool:
    normalized = normalize(reply)
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def ratio(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": round(count / denominator, 4) if denominator else None,
    }


class CountingLLM:
    """Wraps `graph._llm` to count how many times the graph actually reaches
    the Anthropic API, so the run's cost is measured rather than estimated.
    One `run_chat()` costs at least two invocations when a tool is called
    (decide → answer), one when the model replies directly."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._inner.invoke(*args, **kwargs)


def build_corpus(rebuild: bool) -> tuple[Any, ChromaVectorStore, int]:
    """Returns (engine, vector store, number of extraction API calls made).

    Reuses an existing corpus unless `rebuild` is set: re-extracting costs one
    Anthropic call per PDF and nothing about the PDFs changes between runs.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{CORPUS_DIR / 'eval.db'}")
    SQLModel.metadata.create_all(engine)
    store = ChromaVectorStore(path=str(CORPUS_DIR / "chroma"), collection_name="payslips_eval")

    with Session(engine) as session:
        existing = list(session.exec(select(Payslip)).all())

    if existing and not rebuild:
        print(f"Corpus réutilisé ({len(existing)} bulletins) — 0 appel d'extraction.")
        return engine, store, 0

    if existing:
        print("Reconstruction du corpus…")
        with Session(engine) as session:
            for payslip in session.exec(select(Payslip)).all():
                assert payslip.id is not None
                store.delete(payslip.id)
                session.delete(payslip)
            session.commit()

    extractor = ClaudeExtractor()
    pdfs = sorted(p for p in EXAMPLES_DIR.glob("*.pdf") if p.name not in EXCLUDED_PDFS)
    for pdf in pdfs:
        raw_text = extract_text_from_pdf(pdf.read_bytes())
        extracted = extractor.extract(raw_text)
        with Session(engine) as session:
            payslip = Payslip(**extracted.model_dump(), raw_text=raw_text, filename=pdf.name)
            session.add(payslip)
            session.commit()
            session.refresh(payslip)
            payslip_id, mois_annee = payslip.id, payslip.mois_annee
        assert payslip_id is not None
        store.index(payslip_id, mois_annee, raw_text)
        print(f"  indexé {pdf.name} -> {mois_annee}")

    return engine, store, len(pdfs)


def verify_benchmark(cases: list[dict[str, Any]], absent_terms: list[str]) -> None:
    """Fails loudly if the benchmark makes a claim the real extracted text
    doesn't support. This is what keeps the cases honest: every
    `expected_evidence` must fit inside one real chunk of the named PDF (the
    Evidence Hit metric tests a single retrieved chunk, so evidence split
    across two chunks would be unreachable by construction), and every term
    the unanswerable cases assume is missing must really be missing from all
    seven documents."""
    texts = {
        pdf.name: extract_text_from_pdf(pdf.read_bytes())
        for pdf in sorted(EXAMPLES_DIR.glob("*.pdf"))
        if pdf.name not in EXCLUDED_PDFS
    }
    problems: list[str] = []

    for case in cases:
        if not case["answerable"]:
            continue
        document = case["document"]
        if document not in texts:
            problems.append(f"{case['id']} : document inconnu {document!r}")
            continue
        chunks = _chunk_text(texts[document])
        if not any(all(term in chunk for term in case["expected_evidence"]) for chunk in chunks):
            missing = [t for t in case["expected_evidence"] if t not in texts[document]]
            problems.append(
                f"{case['id']} : expected_evidence introuvable dans un seul chunk de {document}"
                + (f" — absent du texte entier : {missing}" if missing else "")
            )

    for term in absent_terms:
        found = [name for name, text in texts.items() if normalize(term) in normalize(text)]
        if found:
            problems.append(f"terme supposé absent {term!r} présent dans {found}")

    if problems:
        raise SystemExit("Benchmark invalide :\n  " + "\n  ".join(problems))


def collect_provenance(
    store: ChromaVectorStore, engine: Any, extraction_calls: int
) -> dict[str, Any]:
    """Every value here is read from the code or from the live objects rather
    than written down by hand — a hand-copied `top_k: 3` would keep saying 3
    the day `search()`'s default changes."""
    collection = store._collection  # noqa: SLF001 -- evaluation harness, see module docstring
    embedding_fn = collection._embedding_function  # noqa: SLF001
    chunk_params = signature(_chunk_text).parameters
    search_params = signature(ChromaVectorStore.search).parameters

    with Session(engine) as session:
        payslips = list(session.exec(select(Payslip)).all())

    return {
        "embedding_model": getattr(embedding_fn, "MODEL_NAME", type(embedding_fn).__name__),
        "embedding_runtime": type(embedding_fn).__name__,
        "generation_model": graph_mod.CHAT_MODEL,
        "extraction_model": EXTRACTION_MODEL,
        "provider": ChatAnthropic.__module__.split(".")[0],
        "retrieval_top_k": search_params["n_results"].default,
        "similarity_distance_threshold": SIMILARITY_DISTANCE_THRESHOLD,
        "chunk_size": chunk_params["chunk_size"].default,
        "chunk_overlap": chunk_params["overlap"].default,
        "indexed_pdfs": len(payslips),
        "indexed_documents": len(payslips),
        "indexed_chunks": collection.count(),
        "indexed_months": sorted(p.mois_annee for p in payslips),
        "excluded_pdfs": list(EXCLUDED_PDFS),
        "extraction_api_calls_this_run": extraction_calls,
    }


def run_cases(store: ChromaVectorStore, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One retrieval pass and one full-graph pass per case."""
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"[{index}/{len(cases)}] {case['id']} — {question}")

        hits = store.search(question)
        # `search()` drops distances on purpose (they're an implementation
        # detail of the vector-store Protocol, not something its callers get
        # to see). The harness wants them, so it re-runs the same query
        # straight against the collection — deterministic, local, and free.
        raw = store._collection.query(query_texts=[question], n_results=3)  # noqa: SLF001
        distances = (raw.get("distances") or [[]])[0]

        chat = graph_mod.run_chat(question)
        sources = chat["sources"] or []

        results.append(
            {
                "id": case["id"],
                "question": question,
                "answerable": case["answerable"],
                "unanswerable_kind": case.get("unanswerable_kind"),
                "expected_source": case["expected_source"],
                "expected_evidence": case["expected_evidence"],
                "retrieved": [
                    {
                        "rank": rank,
                        "mois_annee": hit["mois_annee"],
                        "distance": round(float(distance), 4),
                        "text": hit["text"],
                    }
                    for rank, (hit, distance) in enumerate(zip(hits, distances), start=1)
                ],
                "abstained": has_abstained(chat["reply"]),
                "sources_total": len(sources),
                # "Well formed" replaces the classic citation-ID check, which
                # this architecture has no equivalent of: a source here is a
                # (mois_annee, extrait) pair built mechanically from the tool's
                # own output, so what can be checked is that both halves are
                # actually populated. See RAPPORT.md.
                "sources_well_formed": sum(
                    1
                    for source in sources
                    if source["mois_annee"].strip() and source["extrait"].strip()
                ),
                "reply_excerpt": chat["reply"][:REPLY_EXCERPT_CHARS],
            }
        )

    return results


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    def source_hit(result: dict[str, Any], k: int) -> bool:
        return any(h["mois_annee"] == result["expected_source"] for h in result["retrieved"][:k])

    def evidence_hit(result: dict[str, Any], k: int) -> bool:
        return any(
            all(term in hit["text"] for term in result["expected_evidence"])
            for hit in result["retrieved"][:k]
        )

    kinds = sorted({r["unanswerable_kind"] for r in unanswerable})
    total_sources = sum(r["sources_total"] for r in results)

    return {
        "retrieval": {
            "source_hit_at_1": ratio(sum(source_hit(r, 1) for r in answerable), len(answerable)),
            "source_hit_at_3": ratio(sum(source_hit(r, 3) for r in answerable), len(answerable)),
            "evidence_hit_at_1": ratio(
                sum(evidence_hit(r, 1) for r in answerable), len(answerable)
            ),
            "evidence_hit_at_3": ratio(
                sum(evidence_hit(r, 3) for r in answerable), len(answerable)
            ),
            "unanswerable_with_retrieved_chunks": ratio(
                sum(1 for r in unanswerable if r["retrieved"]), len(unanswerable)
            ),
        },
        "response_behavior": {
            "correct_abstentions": ratio(
                sum(1 for r in unanswerable if r["abstained"]), len(unanswerable)
            ),
            "false_abstentions": ratio(
                sum(1 for r in answerable if r["abstained"]), len(answerable)
            ),
            "correct_abstentions_by_kind": {
                kind: ratio(
                    sum(
                        1
                        for r in unanswerable
                        if r["unanswerable_kind"] == kind and r["abstained"]
                    ),
                    sum(1 for r in unanswerable if r["unanswerable_kind"] == kind),
                )
                for kind in kinds
            },
            "detection": {
                "method": (
                    "marqueurs lexicaux appliqués à la réponse normalisée "
                    "(minuscules, accents retirés)"
                ),
                "markers": list(ABSTENTION_MARKERS),
                "caveat": (
                    "Heuristique, pas un juge LLM : une abstention formulée hors de ces "
                    "marqueurs serait comptée comme une réponse. Chaque réponse est conservée "
                    "(tronquée) dans `cases` pour permettre une vérification manuelle."
                ),
            },
        },
        "traceability": {
            "answerable_responses_with_sources": ratio(
                sum(1 for r in answerable if r["sources_total"]), len(answerable)
            ),
            "well_formed_sources": ratio(
                sum(r["sources_well_formed"] for r in results), total_sources
            ),
            "adaptation_note": (
                "Cette architecture n'émet pas d'identifiant de citation dans le texte de la "
                "réponse : les sources sont un tableau (mois_annee, extrait) reconstruit "
                "mécaniquement à partir des ToolMessage du graphe (voir _extract_sources). "
                "« Citation valide » est donc mesuré comme « source bien formée » : les deux "
                "champs sont non vides. Le lien entre une phrase précise de la réponse et la "
                "source qui la soutient n'est PAS vérifié ici — cela relèverait de la "
                "relecture humaine déclarée non applicable."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de fiabilité du RAG InfoPay AI.")
    parser.add_argument(
        "--rebuild-corpus",
        action="store_true",
        help="ré-extrait les PDF (1 appel Anthropic par bulletin) au lieu de réutiliser .corpus/",
    )
    args = parser.parse_args()

    benchmark = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = benchmark["cases"]

    engine, store, extraction_calls = build_corpus(args.rebuild_corpus)
    analytics_mod.engine = engine
    tools_mod.get_vector_store = lambda: store

    verify_benchmark(cases, benchmark["absent_terms"])
    print(f"Benchmark vérifié contre le texte réellement extrait : {len(cases)} cas.\n")

    counting_llm = CountingLLM(graph_mod._llm)  # noqa: SLF001
    graph_mod._llm = counting_llm  # noqa: SLF001

    results = run_cases(store, cases)
    metrics = compute_metrics(results)

    unanswerable = [r for r in results if not r["answerable"]]
    output = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark": {
            "file": str(BENCHMARK_FILE.relative_to(REPO_ROOT)),
            "total_cases": len(cases),
            "answerable_cases": len(cases) - len(unanswerable),
            "unanswerable_cases": len(unanswerable),
            "unanswerable_breakdown": dict(Counter(r["unanswerable_kind"] for r in unanswerable)),
        },
        **metrics,
        "human_review": {
            "answer_correctness": "not_applicable",
            "grounding": "not_applicable",
            "coverage": "not_applicable",
            "support": "not_applicable",
            "note": (
                "Ces quatre métriques exigent une relecture humaine cas par cas, qui n'a pas "
                "été menée. Elles sont déclarées non applicables plutôt qu'affichées à zéro, "
                "ce qui laisserait croire à une mesure ratée au lieu d'une mesure absente."
            ),
        },
        "provenance": collect_provenance(store, engine, extraction_calls),
        "api_calls": {
            "extraction": extraction_calls,
            "chat_runs": len(cases),
            "chat_llm_invocations": counting_llm.calls,
            "note": (
                "Comptage réel, pas une estimation : chat_llm_invocations est incrémenté dans "
                "le nœud agent du graphe (un appel pour décider de l'outil, un autre pour "
                "rédiger la réponse à partir de son résultat)."
            ),
        },
        "caveat": (
            "Mesuré sur un benchmark restreint (36 cas) construit manuellement à partir de "
            "7 bulletins fictifs. Ces chiffres décrivent ce corpus, pas la performance du "
            "système en production."
        ),
        "cases": [
            {
                **result,
                "retrieved": [
                    {**hit, "text": hit["text"][:CHUNK_EXCERPT_CHARS]}
                    for hit in result["retrieved"]
                ],
            }
            for result in results
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nRésultats écrits dans {OUTPUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
