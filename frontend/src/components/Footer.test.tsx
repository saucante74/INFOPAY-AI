import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithRouter } from "../test/helpers";
import Footer from "./Footer";

describe("Footer", () => {
  it("renders the copyright line with the current year", () => {
    renderWithRouter(<Footer />);

    const year = String(new Date().getFullYear());
    expect(screen.getByText(`© ${year} InfoPay AI. Tous droits réservés.`)).toBeInTheDocument();
  });

  it("renders the transparency and legal links, pointing to their respective pages", () => {
    renderWithRouter(<Footer />);

    expect(screen.getByRole("link", { name: "Fiabilité" })).toHaveAttribute("href", "/metriques");
    expect(screen.getByRole("link", { name: "Confidentialité" })).toHaveAttribute(
      "href",
      "/confidentialite"
    );
    expect(screen.getByRole("link", { name: "Conditions d'utilisation" })).toHaveAttribute(
      "href",
      "/conditions-utilisation"
    );
  });

  it("no longer renders a 'Contact' link", () => {
    renderWithRouter(<Footer />);
    expect(screen.queryByText("Contact")).not.toBeInTheDocument();
  });
});
