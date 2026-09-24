import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import runData from "../data/rag_run.json";
import { assertDefined } from "../test/helpers";
import MetricsPage from "./MetricsPage";

/**
 * Expected values are derived from the same imported run data rather than
 * pasted in as literals: re-running `backend/evaluation/run_benchmark.py`
 * changes every number on the page, and a test asserting "25,0 %" would fail
 * on the next run for no reason the test is actually about. What's verified
 * here is the rendering — that each metric reaches the DOM, in French percent
 * formatting, under the right label — plus the parts that are deliberate
 * editorial choices and must not silently disappear (the caveat, the four
 * "Non applicable" tiles, the provenance table).
 */
const percent = new Intl.NumberFormat("fr-FR", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

/** The `<div>` wrapping a tile's (or a provenance row's) `<dt>`/`<dd>` pair. */
function tileFor(label: string): HTMLElement {
  const term = screen.getByText(label);
  assertDefined(term.parentElement, `expected the tile labelled "${label}" to have a wrapper`);
  return term.parentElement;
}

/**
 * RTL's default matcher collapses whitespace runs in the DOM's text before
 * comparing, but leaves the string matcher untouched — so querying for
 * `Intl.NumberFormat`'s "25,0 %", whose space is a narrow no-break space
 * (U+202F) and real fr-FR formatting, silently fails to match text that has
 * already been collapsed to an ASCII space. Comparing both sides byte for
 * byte is the same fix `PayslipTable.test.tsx` applies to `formatEuros`;
 * see CONVENTIONS.md, "Gotchas found".
 */
const exactBytes = { normalizer: (text: string) => text };

describe("MetricsPage", () => {
  it("renders the page title and every section heading", () => {
    render(<MetricsPage />);

    expect(
      screen.getByRole("heading", { name: "Fiabilité de l'assistant — mesures", level: 1 })
    ).toBeInTheDocument();

    for (const heading of [
      "Recherche documentaire",
      "Comportement de réponse",
      "Traçabilité",
      "Qualité des réponses",
      "Configuration du run",
    ]) {
      expect(screen.getByRole("heading", { name: heading, level: 2 })).toBeInTheDocument();
    }
  });

  it("shows the four retrieval metrics with their rate and their raw counts", () => {
    render(<MetricsPage />);

    const tiles = [
      ["Bon bulletin — 1er résultat", runData.retrieval.source_hit_at_1],
      ["Bon bulletin — 3 premiers", runData.retrieval.source_hit_at_3],
      ["Bonne information — 1er résultat", runData.retrieval.evidence_hit_at_1],
      ["Bonne information — 3 premiers", runData.retrieval.evidence_hit_at_3],
    ] as const;

    for (const [label, ratio] of tiles) {
      const tile = within(tileFor(label));
      expect(tile.getByText(percent.format(ratio.rate), exactBytes)).toBeInTheDocument();
      expect(
        tile.getByText(`${String(ratio.count)} / ${String(ratio.denominator)} cas`)
      ).toBeInTheDocument();
    }
  });

  it("calls out that finding the source differs from finding the evidence when the gap justifies it", () => {
    render(<MetricsPage />);

    // Guard: this run's data is what makes the insight appear at all. If a
    // future run closes the gap, this test should be revisited alongside the
    // page rather than silently staying green.
    expect(
      runData.retrieval.source_hit_at_3.rate - runData.retrieval.evidence_hit_at_3.rate
    ).toBeGreaterThanOrEqual(0.05);
    expect(screen.getByText(/Trouver la source n'est pas la même chose/)).toBeInTheDocument();
  });

  it("reports both abstention metrics and the per-kind breakdown", () => {
    render(<MetricsPage />);

    for (const label of [
      "Abstentions correctes (questions sans réponse)",
      "Abstentions à tort (questions avec réponse)",
      "Hors-sujet complet",
      "Champ absent des bulletins",
      "Bulletin non importé",
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }

    // The editorial point the numbers alone would not carry: a zero
    // abstention rate on off-topic questions is the intended behaviour.
    expect(screen.getByText(/est donc le comportement voulu, pas un défaut/)).toBeInTheDocument();
  });

  it("presents traceability as adapted from classic citations, not as citation IDs", () => {
    render(<MetricsPage />);

    expect(screen.getByText("Réponses accompagnées d'au moins une source")).toBeInTheDocument();
    expect(
      screen.getByText("Sources bien formées (mois et extrait renseignés)")
    ).toBeInTheDocument();
    expect(screen.getByText(runData.traceability.adaptation_note)).toBeInTheDocument();
  });

  it("marks the four human-review metrics 'Non applicable' rather than zero", () => {
    render(<MetricsPage />);

    for (const label of ["Answer correctness", "Grounding", "Coverage", "Support"]) {
      expect(within(tileFor(label)).getByText("Non applicable")).toBeInTheDocument();
    }
  });

  it("shows the run's real configuration, read from the run file", () => {
    render(<MetricsPage />);

    // Scoped row by row: the extraction and generation models are the same
    // string in this run, so a page-wide `getByText` would find two nodes.
    const rows = [
      ["Modèle d'embedding", runData.provenance.embedding_model],
      ["Modèle de génération", runData.provenance.generation_model],
      ["Modèle d'extraction", runData.provenance.extraction_model],
      ["Top-k de la recherche", String(runData.provenance.retrieval_top_k)],
      ["Taille des chunks", `${String(runData.provenance.chunk_size)} caractères`],
      ["Mois couverts", runData.provenance.indexed_months.join(", ")],
    ] as const;

    for (const [label, value] of rows) {
      expect(within(tileFor(label)).getByText(value)).toBeInTheDocument();
    }
  });

  it("displays the limitation warning alongside the numbers", () => {
    render(<MetricsPage />);

    expect(screen.getByText(runData.caveat)).toBeInTheDocument();
    expect(screen.getByText(/benchmark restreint/)).toBeInTheDocument();
  });
});
