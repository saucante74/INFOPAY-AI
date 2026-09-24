import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

import runData from "../data/rag_run.json";

/**
 * The "/metriques" route: what the RAG layer actually scores on a fixed
 * benchmark, measured rather than claimed.
 *
 * The numbers come from `src/data/rag_run.json`, written by
 * `backend/evaluation/run_benchmark.py` against an isolated SQLite +
 * ChromaDB corpus built from `public/exemples/`. It's imported as a module,
 * not fetched: the file is a build-time artifact of this repo, so importing
 * it gives the page a compile-time type inferred from the real file (a
 * renamed metric breaks `tsc -b` here, the same guarantee `satisfies keyof
 * Payslip` buys elsewhere), and removes the loading/error/empty states a
 * `fetch()` of a file that cannot be missing would otherwise need. See
 * RAPPORT.md for why this won over a backend endpoint.
 *
 * Nothing on this page is hardcoded from a past run — every figure, and the
 * decision to show the source-vs-evidence insight at all, is derived from
 * the imported data, so re-running the benchmark updates the page by
 * rebuilding.
 */

interface Ratio {
  count: number;
  denominator: number;
  rate: number | null;
}

const percentFormatter = new Intl.NumberFormat("fr-FR", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

function formatRate(rate: number | null): string {
  return rate === null ? "—" : percentFormatter.format(rate);
}

/**
 * The run file carries the sentinel `"not_applicable"` for the metrics that
 * would need a human reviewer, rather than omitting them or writing 0 —
 * translated here instead of hardcoded, so a run that one day carries a real
 * value displays it rather than still claiming the metric wasn't measured.
 */
function formatHumanReview(value: string): string {
  return value === "not_applicable" ? "Non applicable" : value;
}

/**
 * The gap that makes "finding the source" and "finding the evidence" worth
 * calling out as two different things. Below it, the two metrics tell the
 * same story and the insight would be noise.
 */
const INSIGHT_GAP_THRESHOLD = 0.05;

function gap(a: Ratio, b: Ratio): number {
  return a.rate === null || b.rate === null ? 0 : a.rate - b.rate;
}

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <section className="mt-6 rounded-lg border border-border bg-surface-raised p-5">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      {description !== undefined && <p className="mt-1 text-sm text-ink-soft">{description}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

/**
 * `tabular` (see index.css) keeps the digits from shifting width between
 * values, so a column of percentages lines up.
 */
function MetricTile({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-md border border-border p-3">
      <dt className="text-xs font-medium text-ink-soft">{label}</dt>
      <dd className="tabular mt-1 text-2xl font-bold text-ink">{value}</dd>
      {detail !== undefined && <p className="mt-1 text-xs text-ink-soft">{detail}</p>}
    </div>
  );
}

function RatioTile({ label, ratio }: { label: string; ratio: Ratio }) {
  return (
    <MetricTile
      label={label}
      value={formatRate(ratio.rate)}
      detail={`${String(ratio.count)} / ${String(ratio.denominator)} cas`}
    />
  );
}

export default function MetricsPage() {
  const { benchmark, retrieval, response_behavior, traceability, human_review, provenance } =
    runData;

  const evidenceGapAt1 = gap(retrieval.source_hit_at_1, retrieval.evidence_hit_at_1);
  const evidenceGapAt3 = gap(retrieval.source_hit_at_3, retrieval.evidence_hit_at_3);
  const showEvidenceInsight =
    evidenceGapAt1 >= INSIGHT_GAP_THRESHOLD || evidenceGapAt3 >= INSIGHT_GAP_THRESHOLD;

  const runDate = new Date(runData.generated_at).toLocaleDateString("fr-FR", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  const humanReviewMetrics = [
    { label: "Answer correctness", value: human_review.answer_correctness },
    { label: "Grounding", value: human_review.grounding },
    { label: "Coverage", value: human_review.coverage },
    { label: "Support", value: human_review.support },
  ];

  const provenanceRows = [
    { label: "Modèle d'embedding", value: provenance.embedding_model },
    { label: "Exécution de l'embedding", value: provenance.embedding_runtime },
    { label: "Modèle de génération", value: provenance.generation_model },
    { label: "Modèle d'extraction", value: provenance.extraction_model },
    { label: "Fournisseur", value: provenance.provider },
    { label: "Top-k de la recherche", value: String(provenance.retrieval_top_k) },
    { label: "Seuil de distance", value: String(provenance.similarity_distance_threshold) },
    { label: "Taille des chunks", value: `${String(provenance.chunk_size)} caractères` },
    { label: "Chevauchement", value: `${String(provenance.chunk_overlap)} caractères` },
    { label: "PDF indexés", value: String(provenance.indexed_pdfs) },
    { label: "Documents indexés", value: String(provenance.indexed_documents) },
    { label: "Chunks indexés", value: String(provenance.indexed_chunks) },
    { label: "Mois couverts", value: provenance.indexed_months.join(", ") },
    { label: "PDF exclus", value: provenance.excluded_pdfs.join(", ") },
  ];

  return (
    <div className="mx-auto w-full max-w-4xl">
      <h1 className="text-2xl font-bold">Fiabilité de l'assistant — mesures</h1>
      <p className="mt-1 text-sm text-ink-soft">
        Run du {runDate} — {benchmark.total_cases} questions de test ({benchmark.answerable_cases}{" "}
        auxquelles les bulletins permettent de répondre, {benchmark.unanswerable_cases} non), sur{" "}
        {provenance.indexed_documents} bulletins fictifs.
      </p>

      {/* Same wording discipline as the FAQ and the privacy page: say what
          the measurement does NOT establish, in the same place as the
          numbers, rather than in a footnote nobody reads. */}
      <div className="mt-6 flex gap-3 rounded-lg border border-alert bg-alert-soft p-4">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-alert" aria-hidden="true" />
        <p className="text-sm text-ink">{runData.caveat}</p>
      </div>

      <Section
        title="Recherche documentaire"
        description="Le bulletin attendu, puis l'information attendue, apparaissent-ils dans le 1er résultat de recherche, et dans les 3 premiers ?"
      >
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <RatioTile label="Bon bulletin — 1er résultat" ratio={retrieval.source_hit_at_1} />
          <RatioTile label="Bon bulletin — 3 premiers" ratio={retrieval.source_hit_at_3} />
          <RatioTile label="Bonne information — 1er résultat" ratio={retrieval.evidence_hit_at_1} />
          <RatioTile label="Bonne information — 3 premiers" ratio={retrieval.evidence_hit_at_3} />
        </dl>

        {showEvidenceInsight && (
          <p className="mt-4 border-l-2 border-accent pl-3 text-sm text-ink-soft">
            Trouver la source n'est pas la même chose que trouver la preuve : le bon bulletin
            remonte nettement plus souvent que l'extrait qui contient réellement l'information
            demandée ({formatRate(retrieval.source_hit_at_3.rate)} contre{" "}
            {formatRate(retrieval.evidence_hit_at_3.rate)} sur les 3 premiers résultats). Un
            bulletin correctement identifié mais dont le morceau retenu ne porte pas la bonne ligne
            ne permet pas de répondre.
          </p>
        )}

        <p className="mt-3 text-sm text-ink-soft">
          Sur les {retrieval.unanswerable_with_retrieved_chunks.denominator} questions sans réponse
          possible, {retrieval.unanswerable_with_retrieved_chunks.count} ont tout de même fait
          remonter au moins un extrait sous le seuil de distance — la recherche seule ne suffit pas
          à décider qu'il n'y a rien à dire.
        </p>
      </Section>

      <Section
        title="Comportement de réponse"
        description="L'assistant reconnaît-il qu'il n'a rien trouvé, et le fait-il au bon moment ?"
      >
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <RatioTile
            label="Abstentions correctes (questions sans réponse)"
            ratio={response_behavior.correct_abstentions}
          />
          <RatioTile
            label="Abstentions à tort (questions avec réponse)"
            ratio={response_behavior.false_abstentions}
          />
        </dl>

        <dl className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <RatioTile
            label="Hors-sujet complet"
            ratio={response_behavior.correct_abstentions_by_kind.off_topic}
          />
          <RatioTile
            label="Champ absent des bulletins"
            ratio={response_behavior.correct_abstentions_by_kind.absent_field}
          />
          <RatioTile
            label="Bulletin non importé"
            ratio={response_behavior.correct_abstentions_by_kind.not_imported}
          />
        </dl>

        <p className="mt-4 text-sm text-ink-soft">
          Le détail par type compte : sur une question de culture générale, le prompt système
          demande explicitement à l'assistant de répondre avec ses connaissances plutôt que de
          chercher dans les bulletins. Un taux d'abstention nul sur la catégorie « hors-sujet
          complet » est donc le comportement voulu, pas un défaut.
        </p>
        <p className="mt-2 text-sm text-ink-soft">
          L'abstention est détectée par {response_behavior.detection.method}. C'est une heuristique,
          pas un jugement humain ni un juge automatique.
        </p>
      </Section>

      <Section
        title="Traçabilité"
        description="Quand l'assistant répond, peut-on remonter à ce sur quoi il s'est appuyé ?"
      >
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <RatioTile
            label="Réponses accompagnées d'au moins une source"
            ratio={traceability.answerable_responses_with_sources}
          />
          <RatioTile
            label="Sources bien formées (mois et extrait renseignés)"
            ratio={traceability.well_formed_sources}
          />
        </dl>
        <p className="mt-4 text-sm text-ink-soft">{traceability.adaptation_note}</p>
      </Section>

      <Section
        title="Qualité des réponses"
        description="Quatre mesures qui demandent une relecture humaine, cas par cas."
      >
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {humanReviewMetrics.map(({ label, value }) => (
            <MetricTile key={label} label={label} value={formatHumanReview(value)} />
          ))}
        </dl>
        <p className="mt-4 text-sm text-ink-soft">{human_review.note}</p>
      </Section>

      <Section
        title="Configuration du run"
        description="Relevée dans le code au moment du run, pas recopiée à la main."
      >
        <dl className="grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2">
          {provenanceRows.map(({ label, value }) => (
            <div key={label} className="flex justify-between gap-4 border-b border-border py-1.5">
              <dt className="text-sm text-ink-soft">{label}</dt>
              <dd className="tabular text-sm font-medium text-ink">{value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-sm text-ink-soft">
          Ce run a coûté {runData.api_calls.extraction} appels d'extraction et{" "}
          {runData.api_calls.chat_llm_invocations} appels au modèle de génération, pour{" "}
          {runData.api_calls.chat_runs} questions.
        </p>
      </Section>
    </div>
  );
}
