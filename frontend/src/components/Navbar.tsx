import { LogIn, LogOut } from "lucide-react";
import { useState } from "react";
import { NavLink } from "react-router";

import { requireAuth } from "../auth/authModal";
import { clearToken, useAuthToken } from "../auth/tokenStore";
import { useRateLimits } from "../hooks/useRateLimits";
import ConfirmDialog from "./ConfirmDialog";
import RateLimitBadge from "./RateLimitBadge";
import ThemeToggle from "./ThemeToggle";

/**
 * `text-ink`, not `text-accent`, for the active state: the label needs to
 * stay legible on its own (text-accent on the dark navbar background
 * measures 3.02:1, short of the 4.5:1 normal-text WCAG threshold — see
 * RAPPORT.md). The underline carries the accent color instead, alongside
 * `NavLink`'s own `aria-current="page"` and the bolder weight, so "active"
 * is never signalled by color alone (WCAG SC 1.4.1).
 */
function navLinkClassName({ isActive }: { isActive: boolean }): string {
  return isActive
    ? "font-semibold text-ink underline decoration-accent decoration-2 underline-offset-4"
    : "text-ink-soft transition-colors hover:text-ink";
}

export default function Navbar() {
  const token = useAuthToken();
  const rateLimits = useRateLimits();
  const [isConfirmingLogout, setIsConfirmingLogout] = useState(false);

  return (
    <header className="border-b border-border bg-surface-raised">
      {/* Matches `RootLayout`'s `<main>` width — see RAPPORT.md. */}
      {/* Below `sm` the row wraps: logo + right-hand icons stay on the first
          line and the three tabs drop to a second one (`order-last w-full`
          on the <nav>). Measured, not assumed — see RAPPORT.md: with the
          brand text hidden, logo + 3 tabs + quota gauges + theme + login is
          ~460px of content, which overflows a 375px viewport on one line.
          `min-h-16` (not `h-16`) so the desktop bar is still exactly 64px
          while the mobile one is free to grow to two lines. */}
      <div className="mx-auto flex min-h-16 w-full max-w-[1540px] flex-wrap items-center gap-x-6 gap-y-1 px-4 py-2 sm:py-0">
        <div className="flex items-center gap-2">
          {/* `logo-mark.svg`: a genuinely transparent, hand-authored vector
              icon (a document outline + a teal checkmark seal), confirmed
              by rendering it with an explicitly transparent background and
              reading the output's alpha channel — 96.7% of the canvas came
              back alpha=0, unlike every previous logo asset used here (see
              RAPPORT.md, "Logo"). That's what makes displaying it directly,
              with no colored badge or background-matched chip, correct
              this time: there is no opaque content to hide. */}
          <img src="/logo-mark.svg" alt="" className="h-8 w-8" />
          <span className="hidden text-lg font-bold text-ink sm:inline">InfoPay AI</span>
        </div>

        <nav
          aria-label="Navigation principale"
          className="order-last flex w-full items-center gap-6 text-sm sm:order-none sm:w-auto"
        >
          {/* `end`: without it, NavLink treats "/" as a prefix match and
              would also report active on every other route. */}
          <NavLink to="/" end className={navLinkClassName}>
            Analyseur
          </NavLink>
          <NavLink to="/aide" className={navLinkClassName}>
            Aide
          </NavLink>
          <NavLink to="/metriques" className={navLinkClassName}>
            Évaluation
          </NavLink>
        </nav>

        <div className="ml-auto flex items-center gap-3">
          {/* `useRateLimits()` already returns `null` while logged out, so
              this alone covers "only for a logged-in user" — including the
              brief window right after login before the first fetch
              resolves, where nothing should render yet rather than a
              stale/zeroed badge. */}
          {rateLimits && <RateLimitBadge limits={rateLimits} />}
          <ThemeToggle />
          {/* No navigation either way: logging out no longer sends you
              anywhere (there's nowhere it needs to — "/" is public), and
              "Se connecter" opens the shared modal in place rather than a
              route change, so whatever page you were reading stays put. */}
          {token ? (
            <button
              type="button"
              onClick={() => {
                setIsConfirmingLogout(true);
              }}
              aria-label="Se déconnecter"
              title="Se déconnecter"
              className="flex h-8 w-8 items-center justify-center rounded-md text-ink-soft transition-colors hover:bg-surface hover:text-alert"
            >
              <LogOut className="h-4 w-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={() => {
                requireAuth();
              }}
              aria-label="Se connecter"
              title="Se connecter"
              className="flex h-8 w-8 items-center justify-center rounded-md text-ink-soft transition-colors hover:bg-surface hover:text-ink"
            >
              <LogIn className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Reuses `ConfirmDialog` as-is (title/message/labels via props) —
          the same generic component `PayslipTable` already uses for the
          delete-a-payslip confirmation, not a second dialog built for this
          one case. `clearToken` is synchronous, so unlike `PayslipTable`'s
          `onDelete` there's nothing to await and no `isConfirming` state
          to pass. */}
      <ConfirmDialog
        isOpen={isConfirmingLogout}
        title="Se déconnecter ?"
        message="Vous devrez vous reconnecter pour importer un bulletin ou poser une question à l'assistant."
        confirmLabel="Se déconnecter"
        onConfirm={() => {
          clearToken();
          setIsConfirmingLogout(false);
        }}
        onCancel={() => {
          setIsConfirmingLogout(false);
        }}
      />
    </header>
  );
}
