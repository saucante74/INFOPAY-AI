import { Link } from "react-router";

/**
 * "Contact" was dropped entirely — no page exists for it, and none is planned.
 *
 * "Fiabilité" sits here rather than in the Navbar on purpose: the Navbar
 * holds the two routes a user comes here to *use* (Analyseur, Aide), while
 * the footer already collects the pages that state what the service does and
 * doesn't do (Confidentialité, Conditions d'utilisation). A page of measured
 * RAG metrics is transparency material of exactly that kind — permanently
 * reachable, never on the critical path of importing a payslip.
 */
const FOOTER_LINKS = [
  { label: "Fiabilité", to: "/metriques" },
  { label: "Confidentialité", to: "/confidentialite" },
  { label: "Conditions d'utilisation", to: "/conditions-utilisation" },
] as const satisfies readonly { label: string; to: string }[];

export default function Footer() {
  return (
    <footer className="border-t border-border bg-surface-raised">
      {/* Matches `RootLayout`'s `<main>` width — see RAPPORT.md. */}
      <div className="mx-auto flex w-full max-w-[1540px] flex-col gap-4 px-4 py-6 text-sm text-ink-soft sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          {/* Same `logo-mark.svg` as Navbar, no frame needed — see its comment there. */}
          <img src="/logo-mark.svg" alt="" className="h-6 w-6" />
          {/* Computed, not the mockup's literal "2026", so it never goes
              stale — see RAPPORT.md. */}
          <span>© {new Date().getFullYear()} InfoPay AI. Tous droits réservés.</span>
        </div>

        <div className="flex items-center gap-6">
          {FOOTER_LINKS.map(({ label, to }) => (
            <Link key={label} to={to} className="transition-colors hover:text-ink">
              {label}
            </Link>
          ))}
        </div>
      </div>
    </footer>
  );
}
