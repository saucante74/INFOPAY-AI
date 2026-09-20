import { AlertTriangle, Download } from "lucide-react";

import FaqAccordion from "../components/FaqAccordion";

/**
 * The 8 fictional sample payslips in `public/exemples/` — see RAPPORT.md
 * for how each was produced and what they represent (a fictional
 * employee's 4 most recent payslips at one employer, plus 4 prior jobs
 * with deliberately different visual layouts, to show extraction isn't
 * tied to one bulletin format).
 */
const RECENT_EXAMPLES = [
  { file: "bulletin_paie_01_2026.pdf", label: "Janvier 2026" },
  { file: "bulletin_paie_02_2026.pdf", label: "Février 2026" },
  { file: "bulletin_paie_03_2026.pdf", label: "Mars 2026" },
  { file: "bulletin_paie_04_2026.pdf", label: "Avril 2026" },
] as const satisfies readonly { file: string; label: string }[];

const VARIED_EXAMPLES = [
  { file: "bulletin_pharmaouest_03_2023.pdf", label: "PharmaOuest — Mars 2023" },
  { file: "bulletin_clinique_11_2024.pdf", label: "Clinique — Novembre 2024" },
  { file: "bulletin_aerospace_08_2025.pdf", label: "AeroSpace — Août 2025" },
] as const satisfies readonly { file: string; label: string }[];

const KNOWN_LIMITATION_EXAMPLE = {
  file: "bulletin_batirenov_06_2022.pdf",
  label: "BatiRenov — Juin 2022",
} as const satisfies { file: string; label: string };

/**
 * One downloadable example link — `download` forces a save instead of an
 * in-tab PDF preview. `variant="limitation"` swaps the hover color from
 * `accent` to `alert` (real WCAG contrast numbers against
 * `bg-surface-raised`, both themes, in RAPPORT.md) and adds a permanently
 * visible `AlertTriangle` icon, so the one example that's a known
 * extraction failure is flagged without requiring a hover to notice it.
 */
function ExampleLink({
  file,
  label,
  variant = "default",
}: {
  file: string;
  label: string;
  variant?: "default" | "limitation";
}) {
  return (
    <a
      href={`/exemples/${file}`}
      download
      className={`flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-ink transition-colors ${
        variant === "limitation"
          ? "hover:border-alert hover:text-alert"
          : "hover:border-accent hover:text-accent"
      }`}
    >
      <Download className="h-4 w-4 shrink-0" aria-hidden="true" />
      {label}
      {variant === "limitation" && (
        <AlertTriangle className="h-4 w-4 shrink-0 text-alert" aria-hidden="true" />
      )}
    </a>
  );
}

/**
 * The "/aide" route. Previously a modal (`HelpModal.tsx`, now removed) —
 * per user feedback, real pages were preferred once the app grew to 3
 * routes. `FaqAccordion` is reused unchanged in content; only the
 * surrounding chrome changed from dialog (backdrop, focus trap,
 * Escape-to-close) to a plain page. See RAPPORT.md.
 *
 * No outer wrapping card here (unlike the first version of this page):
 * `FaqAccordion` now renders each question as its own card, so a second
 * card wrapped around all of them would nest cards inside a card for no
 * reason — the accordion sits directly on the page background instead.
 */
export default function HelpPage() {
  return (
    <div className="mx-auto w-full max-w-2xl">
      <h1 className="text-2xl font-bold">Aide</h1>
      <p className="mt-1 text-sm text-ink-soft">
        Réponses aux questions les plus fréquentes sur InfoPay AI.
      </p>

      <div className="mt-6">
        <FaqAccordion />
      </div>

      <section id="exemples" className="mt-8 rounded-lg border border-border bg-surface-raised p-5">
        <h2 className="text-lg font-semibold text-ink">Exemples à télécharger</h2>
        <p className="mt-1 text-sm text-ink-soft">
          Vous pouvez tester l'application avec ces bulletins fictifs, sans avoir à fournir vos
          propres documents.
        </p>

        <div className="mt-4">
          <h3 className="text-sm font-medium text-ink-soft">Bulletins récents</h3>
          <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
            {RECENT_EXAMPLES.map((example) => (
              <ExampleLink key={example.file} {...example} />
            ))}
          </div>
        </div>

        <div className="mt-4">
          <h3 className="text-sm font-medium text-ink-soft">Exemples de formats variés</h3>
          <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
            {VARIED_EXAMPLES.map((example) => (
              <ExampleLink key={example.file} {...example} />
            ))}
          </div>
        </div>

        <div className="mt-4">
          <h3 className="text-sm font-medium text-alert">Exemples de PDF non acceptés</h3>
          <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div>
              <ExampleLink {...KNOWN_LIMITATION_EXAMPLE} variant="limitation" />
              <p className="mt-1 text-xs text-ink-soft">
                Ce bulletin illustre une limite actuelle de l'extraction (mise en page non
                standard).
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
