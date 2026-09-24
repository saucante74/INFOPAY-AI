import { Link } from "react-router";

/**
 * "Contact" was dropped entirely — no page exists for it, and none is planned.
 *
 * "Fiabilité" points at the same `/metriques` page as the Navbar's
 * "Évaluation" tab. The two labels are deliberately different: the Navbar
 * one names the *page* in the tool's own vocabulary (short, one word, sits
 * beside "Analyseur" and "Aide"), while this one sits among the
 * transparency links (Confidentialité, Conditions d'utilisation) and says
 * what the reader is there to learn — how far the assistant can be trusted.
 * The page's own title, "Fiabilité de l'assistant — mesures", bridges both.
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
