import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import HelpPage from "./HelpPage";

describe("HelpPage", () => {
  it("renders the page title and the FAQ content", () => {
    render(<HelpPage />);

    expect(screen.getByRole("heading", { name: "Aide", level: 1 })).toBeInTheDocument();
    expect(
      screen.getByText("Quels formats de bulletins de paie sont acceptés ?")
    ).toBeInTheDocument();
  });
});

describe("HelpPage — sample payslip downloads", () => {
  const EXAMPLE_FILES = [
    "bulletin_paie_01_2026.pdf",
    "bulletin_paie_02_2026.pdf",
    "bulletin_paie_03_2026.pdf",
    "bulletin_paie_04_2026.pdf",
    "bulletin_batirenov_06_2022.pdf",
    "bulletin_pharmaouest_03_2023.pdf",
    "bulletin_clinique_11_2024.pdf",
    "bulletin_aerospace_08_2025.pdf",
  ] as const;

  it("has a linkable #exemples section", () => {
    render(<HelpPage />);

    expect(screen.getByRole("heading", { name: "Exemples à télécharger" })).toBeInTheDocument();
    expect(document.getElementById("exemples")).toBeInTheDocument();
  });

  it.each(EXAMPLE_FILES)(
    "links to /exemples/%s with a download attribute, forcing a save",
    (file) => {
      render(<HelpPage />);

      const link = document.querySelector(`a[href="/exemples/${file}"]`);
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute("download");
    }
  );
});

describe("HelpPage — BatiRenov known-limitation entry", () => {
  it("has no separate 'limite-connue' section anymore", () => {
    render(<HelpPage />);

    expect(document.getElementById("limite-connue")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Limite actuelle de l'extraction" })
    ).not.toBeInTheDocument();
  });

  it("is in its own 'Exemples de PDF non acceptés' group, not in 'Exemples de formats variés'", () => {
    render(<HelpPage />);

    const variedHeading = screen.getByRole("heading", { name: "Exemples de formats variés" });
    const rejectedHeading = screen.getByRole("heading", {
      name: "Exemples de PDF non acceptés",
    });
    const link = document.querySelector('a[href="/exemples/bulletin_batirenov_06_2022.pdf"]');
    expect(link).toBeInTheDocument();
    expect(variedHeading.parentElement?.contains(link ?? null)).toBe(false);
    expect(rejectedHeading.parentElement?.contains(link ?? null)).toBe(true);
  });

  it("leaves only PharmaOuest, Clinique and AeroSpace in 'Exemples de formats variés'", () => {
    render(<HelpPage />);

    const variedHeading = screen.getByRole("heading", { name: "Exemples de formats variés" });
    const links = variedHeading.parentElement?.querySelectorAll("a") ?? [];
    expect(links).toHaveLength(3);
    expect(Array.from(links).map((link) => link.getAttribute("href"))).toEqual([
      "/exemples/bulletin_pharmaouest_03_2023.pdf",
      "/exemples/bulletin_clinique_11_2024.pdf",
      "/exemples/bulletin_aerospace_08_2025.pdf",
    ]);
  });

  it("only lists the BatiRenov example once", () => {
    render(<HelpPage />);

    const links = document.querySelectorAll('a[href="/exemples/bulletin_batirenov_06_2022.pdf"]');
    expect(links).toHaveLength(1);
  });

  it("carries an always-visible AlertTriangle icon", () => {
    render(<HelpPage />);

    const link = document.querySelector('a[href="/exemples/bulletin_batirenov_06_2022.pdf"]');
    expect(link?.querySelector("svg.lucide-alert-triangle")).toBeInTheDocument();
  });

  it("styles the link with the alert hover color, not accent", () => {
    render(<HelpPage />);

    const link = document.querySelector('a[href="/exemples/bulletin_batirenov_06_2022.pdf"]');
    expect(link).toBeInTheDocument();
    expect(link).toHaveClass("hover:border-alert", "hover:text-alert");
    expect(link).not.toHaveClass("hover:border-accent", "hover:text-accent");
  });

  it("shows a short caption explaining the extraction limitation", () => {
    render(<HelpPage />);

    expect(
      screen.getByText(/Ce format est non valide et utilisé à titre d'exemple/)
    ).toBeInTheDocument();
  });
});
