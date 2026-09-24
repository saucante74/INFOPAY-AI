import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LoginModal from "../auth/LoginModal";
import { getToken, setToken } from "../auth/tokenStore";
import { makeRateLimits } from "../test/fixtures";
import { renderWithRouter } from "../test/helpers";
import Navbar from "./Navbar";

// The real `useRateLimits` hook runs (it's cheap and not the focus of most
// tests here), but its one network call is mocked at the module boundary —
// same convention as every other component test — so tests that log in
// don't fire a real, unhandled request to `/api/rate-limits`.
vi.mock("../api/client", () => ({
  fetchRateLimits: vi.fn(),
}));

import { fetchRateLimits } from "../api/client";

const mockFetchRateLimits = vi.mocked(fetchRateLimits);

beforeEach(() => {
  // Never resolves unless a test opts in: most tests here don't care about
  // the rate-limit badge, and a pending promise just leaves it hidden.
  mockFetchRateLimits.mockReset().mockImplementation(() => new Promise(() => undefined));
});

describe("Navbar", () => {
  it("marks 'Analyseur' as the current page on '/'", () => {
    renderWithRouter(<Navbar />, "/");

    expect(screen.getByText("InfoPay AI")).toBeInTheDocument();

    const analyzer = screen.getByRole("link", { name: "Analyseur" });
    expect(analyzer).toHaveAttribute("aria-current", "page");
    expect(analyzer).toHaveAttribute("href", "/");

    const aide = screen.getByRole("link", { name: "Aide" });
    expect(aide).not.toHaveAttribute("aria-current");
    expect(aide).toHaveAttribute("href", "/aide");

    const evaluation = screen.getByRole("link", { name: "Évaluation" });
    expect(evaluation).not.toHaveAttribute("aria-current");
    expect(evaluation).toHaveAttribute("href", "/metriques");
  });

  it("marks 'Évaluation' as the current page on '/metriques', and only that tab", () => {
    renderWithRouter(<Navbar />, "/metriques");

    expect(screen.getByRole("link", { name: "Évaluation" })).toHaveAttribute(
      "aria-current",
      "page"
    );
    expect(screen.getByRole("link", { name: "Analyseur" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("link", { name: "Aide" })).not.toHaveAttribute("aria-current");
  });

  it("lists the three tabs in order inside the navigation landmark", () => {
    renderWithRouter(<Navbar />);

    const nav = screen.getByRole("navigation", { name: "Navigation principale" });
    expect(
      within(nav)
        .getAllByRole("link")
        .map((link) => link.textContent)
    ).toEqual(["Analyseur", "Aide", "Évaluation"]);
  });

  it("hides the 'InfoPay AI' text below the sm breakpoint but keeps it from sm up, leaving the logo visible", () => {
    const { container } = renderWithRouter(<Navbar />);

    // Class presence only: jsdom applies no stylesheet, so whether the text
    // is *actually* hidden at a given width can't be observed here — that
    // was measured in a real browser (see RAPPORT.md).
    const brand = screen.getByText("InfoPay AI");
    expect(brand).toHaveClass("hidden", "sm:inline");

    const logo = container.querySelector("img");
    expect(logo).not.toBeNull();
    expect(logo).not.toHaveClass("hidden");
  });

  it("marks 'Aide' as the current page on '/aide', not 'Analyseur'", () => {
    renderWithRouter(<Navbar />, "/aide");

    expect(screen.getByRole("link", { name: "Aide" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Analyseur" })).not.toHaveAttribute("aria-current");
  });

  it("renders the theme toggle with a clear aria-label", () => {
    renderWithRouter(<Navbar />);

    // ThemeToggle's own label depends on the resolved theme; asserting an
    // accessible name exists (rather than a specific one) keeps this test
    // decoupled from ThemeToggle's own tested behaviour.
    expect(screen.getByRole("button", { name: /Passer en thème/ })).toBeInTheDocument();
  });

  it("wraps the nav items in a labelled <nav> landmark", () => {
    renderWithRouter(<Navbar />);
    expect(screen.getByRole("navigation", { name: "Navigation principale" })).toBeInTheDocument();
  });

  it("shows 'Se connecter', not 'Se déconnecter', when logged out", () => {
    renderWithRouter(<Navbar />);
    expect(screen.queryByRole("button", { name: "Se déconnecter" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Se connecter" })).toBeInTheDocument();
  });

  it("shows 'Se déconnecter', not 'Se connecter', when logged in", () => {
    setToken("jwt.token.value");
    renderWithRouter(<Navbar />);
    expect(screen.getByRole("button", { name: "Se déconnecter" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Se connecter" })).not.toBeInTheDocument();
  });

  it("clicking 'Se déconnecter' asks for confirmation before clearing the token", async () => {
    const user = userEvent.setup();
    setToken("jwt.token.value");
    renderWithRouter(<Navbar />);

    await user.click(screen.getByRole("button", { name: "Se déconnecter" }));

    expect(screen.getByRole("dialog", { name: "Se déconnecter ?" })).toBeInTheDocument();
    expect(getToken()).toBe("jwt.token.value");
  });

  it("cancelling the logout confirmation leaves the session untouched", async () => {
    const user = userEvent.setup();
    setToken("jwt.token.value");
    renderWithRouter(<Navbar />);

    await user.click(screen.getByRole("button", { name: "Se déconnecter" }));
    await user.click(screen.getByRole("button", { name: "Annuler" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(getToken()).toBe("jwt.token.value");
    expect(screen.getByRole("button", { name: "Se déconnecter" })).toBeInTheDocument();
  });

  it("confirming the logout clears the token without navigating anywhere — '/' is public, there's nowhere it needs to send you", async () => {
    const user = userEvent.setup();
    setToken("jwt.token.value");
    renderWithRouter(
      <>
        <Navbar />
        <Routes>
          <Route path="/aide" element={<p>Page Aide</p>} />
          <Route path="/login" element={<p>Page de connexion</p>} />
        </Routes>
      </>,
      "/aide"
    );

    await user.click(screen.getByRole("button", { name: "Se déconnecter" }));
    // The dialog's confirm button shares the trigger's accessible name
    // ("Se déconnecter"), so it's queried scoped to the dialog itself.
    const dialog = screen.getByRole("dialog", { name: "Se déconnecter ?" });
    await user.click(within(dialog).getByRole("button", { name: "Se déconnecter" }));

    expect(getToken()).toBeNull();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByText("Page Aide")).toBeInTheDocument();
    expect(screen.queryByText("Page de connexion")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Se connecter" })).toBeInTheDocument();
  });

  it("shows the rate-limit badge (as accessible meters) once logged in and the quotas have loaded", async () => {
    setToken("jwt.token.value");
    mockFetchRateLimits.mockResolvedValueOnce(makeRateLimits());
    renderWithRouter(<Navbar />);

    // `RateLimitBadge.test.tsx` covers the gauges' own rendering in
    // detail; this just confirms `Navbar` actually mounts it once data
    // exists, via the accessible name `role="meter"` exposes (the gauge
    // itself has no plain visible text — see RAPPORT.md).
    expect(await screen.findAllByRole("meter")).toHaveLength(2);
  });

  it("does not show the rate-limit badge when logged out", () => {
    renderWithRouter(<Navbar />);

    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(mockFetchRateLimits).not.toHaveBeenCalled();
  });

  it("hides the rate-limit badge again after logging out", async () => {
    const user = userEvent.setup();
    setToken("jwt.token.value");
    mockFetchRateLimits.mockResolvedValueOnce(makeRateLimits());
    renderWithRouter(<Navbar />);
    await screen.findAllByRole("meter");

    await user.click(screen.getByRole("button", { name: "Se déconnecter" }));
    const dialog = screen.getByRole("dialog", { name: "Se déconnecter ?" });
    await user.click(within(dialog).getByRole("button", { name: "Se déconnecter" }));

    await waitFor(() => {
      expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    });
  });

  it("opens the shared login modal from 'Se connecter', in place, without navigating", async () => {
    const user = userEvent.setup();
    renderWithRouter(
      <>
        <Navbar />
        <LoginModal />
      </>
    );

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Se connecter" }));

    expect(screen.getByRole("dialog", { name: "Connexion" })).toBeInTheDocument();
  });
});
