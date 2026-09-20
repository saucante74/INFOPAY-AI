/**
 * The public-home + login-modal flow through the real route tree in
 * App.tsx (its own BrowserRouter, RootLayout + LoginModal, Navbar,
 * AnalyzerPage). Only the API client is mocked, per the suite's
 * convention — `authModal`/`tokenStore` are real, so this is the one
 * place the whole wiring (Navbar/UploadZone/ChatPanel → requireAuth →
 * LoginModal → LoginForm → back to the caller) is exercised together.
 */
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { makePayslip } from "./test/fixtures";
import { assertDefined } from "./test/helpers";

vi.mock("./api/client", () => ({
  login: vi.fn(),
  fetchPayslips: vi.fn(),
  uploadPayslip: vi.fn(),
  sendChatMessage: vi.fn(),
  getApiErrorMessage: vi.fn(),
  getRateLimit: vi.fn(),
  formatRetryDelay: vi.fn(),
}));

import { fetchPayslips, login, sendChatMessage, uploadPayslip } from "./api/client";

const mockLogin = vi.mocked(login);
const mockFetchPayslips = vi.mocked(fetchPayslips);
const mockUploadPayslip = vi.mocked(uploadPayslip);
const mockSendChatMessage = vi.mocked(sendChatMessage);

const pdfFile = new File(["contenu"], "bulletin.pdf", { type: "application/pdf" });

function getFileInput(): HTMLInputElement {
  const input = document.querySelector<HTMLInputElement>('input[type="file"]');
  assertDefined(input, 'expected a <input type="file"> on the analyzer page');
  return input;
}

async function logInThroughModal() {
  const user = userEvent.setup();
  // Scoped to the dialog: Navbar's own "Se connecter" button (still in the
  // DOM behind the modal, just visually covered by the backdrop) shares
  // the same accessible name as the modal's submit button.
  const dialog = within(await screen.findByRole("dialog", { name: "Connexion" }));
  await user.type(dialog.getByLabelText("Identifiant"), "admin");
  await user.type(dialog.getByLabelText("Mot de passe"), "secret");
  await user.click(dialog.getByRole("button", { name: "Se connecter" }));
}

beforeEach(() => {
  mockLogin.mockReset().mockResolvedValue("jwt.token.value");
  mockFetchPayslips.mockReset().mockResolvedValue([]);
  mockUploadPayslip.mockReset();
  mockSendChatMessage.mockReset();
  // App owns a BrowserRouter, which reads the real (jsdom) URL.
  window.history.pushState({}, "", "/");
});

afterEach(() => {
  window.history.pushState({}, "", "/");
});

describe("App — public home page", () => {
  it("shows the analyzer immediately, without a redirect or a login form, for an anonymous visitor", async () => {
    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Assistant et analyse : Bulletin de salaire" })
    ).toBeInTheDocument();
    expect(window.location.pathname).toBe("/");
    expect(screen.queryByLabelText("Identifiant")).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    // No authenticated request was ever made on this visitor's behalf.
    expect(mockFetchPayslips).not.toHaveBeenCalled();
  });

  it("keeps the help and legal pages public", async () => {
    window.history.pushState({}, "", "/confidentialite");
    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Politique de confidentialité" })
    ).toBeInTheDocument();
  });
});

describe("App — login modal triggered by a protected action", () => {
  it("opens the modal on an upload attempt, then resumes that exact upload after logging in", async () => {
    const user = userEvent.setup();
    mockUploadPayslip.mockResolvedValueOnce(makePayslip());
    render(<App />);
    await screen.findByRole("heading", { name: "Assistant et analyse : Bulletin de salaire" });

    await user.upload(getFileInput(), pdfFile);
    expect(mockUploadPayslip).not.toHaveBeenCalled();

    await logInThroughModal();

    expect(mockUploadPayslip).toHaveBeenCalledWith(pdfFile);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    // Logging in also lets the payslip list load for the first time.
    expect(mockFetchPayslips).toHaveBeenCalled();
  });

  it("opens the modal on a chat attempt, then resumes that exact message after logging in", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({ reply: "Réponse de l'assistant.", sources: null });
    render(<App />);
    await screen.findByRole("heading", { name: "Assistant et analyse : Bulletin de salaire" });

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Une question{enter}");
    expect(mockSendChatMessage).not.toHaveBeenCalled();
    // Nothing added to the chat yet — the message hasn't sent.
    expect(screen.queryByText("Une question")).not.toBeInTheDocument();

    await logInThroughModal();

    expect(mockSendChatMessage).toHaveBeenCalledWith("Une question", []);
    expect(await screen.findByText("Réponse de l'assistant.")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("closing the modal without logging in leaves the action un-taken", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("heading", { name: "Assistant et analyse : Bulletin de salaire" });

    await user.upload(getFileInput(), pdfFile);
    await screen.findByRole("dialog", { name: "Connexion" });

    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(mockUploadPayslip).not.toHaveBeenCalled();
  });
});
