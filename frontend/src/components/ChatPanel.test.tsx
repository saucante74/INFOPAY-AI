import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatResponse } from "../api/types";
import ChatPanel from "./ChatPanel";

vi.mock("../api/client", () => ({
  sendChatMessage: vi.fn(),
  getRateLimit: vi.fn(),
  formatRetryDelay: vi.fn(),
}));

vi.mock("../hooks/useRateLimits", () => ({
  decrementRateLimit: vi.fn(),
}));

// Default: run the action immediately, as `requireAuth` does when already
// logged in — every existing test below exercises that path, unchanged
// from before this mock existed. See "requires login before sending" for
// the gated (logged-out) path.
vi.mock("../auth/authModal", () => ({
  requireAuth: vi.fn((action?: () => void) => {
    action?.();
  }),
}));

import { formatRetryDelay, getRateLimit, sendChatMessage } from "../api/client";
import { requireAuth } from "../auth/authModal";
import { decrementRateLimit } from "../hooks/useRateLimits";

const mockSendChatMessage = vi.mocked(sendChatMessage);
const mockGetRateLimit = vi.mocked(getRateLimit);
const mockFormatRetryDelay = vi.mocked(formatRetryDelay);
const mockRequireAuth = vi.mocked(requireAuth);
const mockDecrementRateLimit = vi.mocked(decrementRateLimit);

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function getSubmitButton(): HTMLElement {
  return screen.getByRole("button", { name: "" });
}

beforeEach(() => {
  mockSendChatMessage.mockReset();
  mockGetRateLimit.mockReset();
  mockFormatRetryDelay.mockReset();
  mockRequireAuth.mockReset().mockImplementation((action?: () => void) => {
    action?.();
  });
  mockDecrementRateLimit.mockReset();
});

describe("ChatPanel", () => {
  it("shows the suggestions before any message is sent", () => {
    render(<ChatPanel />);

    expect(screen.getByText("Suggestions")).toBeInTheDocument();
    expect(screen.getByText("Somme des cotisations sociales sur 6 mois")).toBeInTheDocument();
    expect(mockSendChatMessage).not.toHaveBeenCalled();
  });

  it("clicking a suggestion sends it and hides the suggestions", async () => {
    const { promise, resolve } = deferred<ChatResponse>();
    mockSendChatMessage.mockReturnValueOnce(promise);
    render(<ChatPanel />);

    const suggestion = "Quelle est la moyenne de mon net à payer ?";
    await userEvent.click(screen.getByText(suggestion));

    expect(mockSendChatMessage).toHaveBeenCalledWith(suggestion, []);
    expect(screen.queryByText("Suggestions")).not.toBeInTheDocument();
    // The suggestion text now lives in the user's own message bubble.
    expect(screen.getByText(suggestion)).toBeInTheDocument();

    resolve({ reply: "Le net à payer moyen est de 2 340 €.", sources: null });
    expect(await screen.findByText("Le net à payer moyen est de 2 340 €.")).toBeInTheDocument();
    // A successful send consumed one hit of the `chat` scope's budget.
    expect(mockDecrementRateLimit).toHaveBeenCalledWith("chat");
  });

  it("shows a loading indicator while the reply is pending, and disables the submit button", async () => {
    const user = userEvent.setup();
    const { promise, resolve } = deferred<ChatResponse>();
    mockSendChatMessage.mockReturnValueOnce(promise);
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Bonjour");
    const submitButton = getSubmitButton();
    expect(submitButton).toBeEnabled();

    await user.click(submitButton);

    expect(input).toHaveValue(""); // cleared immediately on send
    expect(screen.getByText("Analyse en cours…")).toBeInTheDocument();
    expect(submitButton).toBeDisabled();

    resolve({ reply: "Bonjour ! Comment puis-je vous aider ?", sources: null });

    await waitFor(() => {
      expect(screen.queryByText("Analyse en cours…")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Bonjour ! Comment puis-je vous aider ?")).toBeInTheDocument();
  });

  it("switches to a 'taking longer' message after the slow-reply delay, without claiming to know which tool is running", async () => {
    // `userEvent.type` relies on its own internal timers, which is a poor
    // fit for a test that also fakes timers to control a `setTimeout`
    // inside the component — `fireEvent` sidesteps that entirely (same
    // reasoning CONVENTIONS.md gives for using `fireEvent.drop` over
    // `userEvent` for drag-and-drop).
    vi.useFakeTimers();
    const { promise, resolve } = deferred<ChatResponse>();
    mockSendChatMessage.mockReturnValueOnce(promise);
    const { container } = render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    fireEvent.change(input, { target: { value: "Bonjour" } });
    const form = container.querySelector("form");
    if (!form) throw new Error("form not found");
    fireEvent.submit(form);

    expect(screen.getByText("Analyse en cours…")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(4000);
    });
    expect(screen.getByText("Cela prend un peu plus de temps que d'habitude…")).toBeInTheDocument();
    expect(screen.queryByText("Analyse en cours…")).not.toBeInTheDocument();

    resolve({ reply: "Bonjour ! Comment puis-je vous aider ?", sources: null });
    await act(async () => {
      await promise;
    });
    expect(
      screen.queryByText("Cela prend un peu plus de temps que d'habitude…")
    ).not.toBeInTheDocument();

    vi.useRealTimers();
  });

  it("does not show the 'taking longer' message for a reply that resolves quickly", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({ reply: "Réponse rapide.", sources: null });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Bonjour{enter}");

    await waitFor(() => {
      expect(screen.getByText("Réponse rapide.")).toBeInTheDocument();
    });
    expect(
      screen.queryByText("Cela prend un peu plus de temps que d'habitude…")
    ).not.toBeInTheDocument();
  });

  it("shows a fallback error message when the request fails", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockRejectedValueOnce(new Error("network down"));
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Une question{enter}");

    expect(
      await screen.findByText(
        "Désolé, une erreur est survenue. Vérifiez que le serveur backend est bien lancé."
      )
    ).toBeInTheDocument();
    // A failed send never reached the backend's rate limiter, so nothing
    // should be decremented locally either.
    expect(mockDecrementRateLimit).not.toHaveBeenCalled();
  });

  it("shows a specific message when the hourly question limit is reached", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockRejectedValueOnce(new Error("429"));
    mockGetRateLimit.mockReturnValueOnce({ retryAfterMinutes: 5 });
    mockFormatRetryDelay.mockReturnValueOnce("dans 5 min");
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Une question{enter}");

    expect(
      await screen.findByText(
        "Limite de questions atteinte pour cette heure. Réessayez dans 5 min."
      )
    ).toBeInTheDocument();
    expect(
      screen.queryByText(
        "Désolé, une erreur est survenue. Vérifiez que le serveur backend est bien lancé."
      )
    ).not.toBeInTheDocument();
  });

  it("does not send an empty or whitespace-only message", async () => {
    const user = userEvent.setup();
    render(<ChatPanel />);

    const submitButton = getSubmitButton();
    expect(submitButton).toBeDisabled();

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "   ");
    expect(submitButton).toBeDisabled();

    await user.click(submitButton);
    expect(mockSendChatMessage).not.toHaveBeenCalled();
    expect(screen.getByText("Suggestions")).toBeInTheDocument();
    // An empty message never reaches the auth gate at all.
    expect(mockRequireAuth).not.toHaveBeenCalled();
  });

  it("aligns the user's message to the right and the assistant's reply to the left", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({ reply: "Réponse de l'assistant.", sources: null });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Ma question{enter}");
    await screen.findByText("Réponse de l'assistant.");

    const userRow = screen.getByText("Ma question").closest(".flex");
    const assistantRow = screen.getByText("Réponse de l'assistant.").closest(".flex");
    expect(userRow?.className).toContain("justify-end");
    expect(assistantRow?.className).toContain("justify-start");
    // Every non-empty send goes through the auth gate, even when (as here,
    // and in every test above) it turns out to already be satisfied.
    expect(mockRequireAuth).toHaveBeenCalledTimes(1);
  });

  it("requires login before sending: does not call the API when logged out", async () => {
    const user = userEvent.setup();
    mockRequireAuth.mockImplementation(() => {
      // Simulates "logged out": requireAuth opens the modal instead of
      // running the action — verified here by simply *not* calling it.
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Une question{enter}");

    expect(mockRequireAuth).toHaveBeenCalledTimes(1);
    expect(mockSendChatMessage).not.toHaveBeenCalled();
    // Nothing added to the chat, and the input wasn't cleared — the
    // message hasn't actually "been sent" yet.
    expect(screen.queryByText("Une question")).not.toBeInTheDocument();
    expect(input).toHaveValue("Une question");
  });

  it("resumes the exact message once login succeeds, with the input already cleared", async () => {
    const user = userEvent.setup();
    let resumeSend: (() => void) | undefined;
    mockRequireAuth.mockImplementation((action?: () => void) => {
      resumeSend = action;
    });
    mockSendChatMessage.mockResolvedValueOnce({ reply: "Réponse de l'assistant.", sources: null });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Une question{enter}");
    expect(mockSendChatMessage).not.toHaveBeenCalled();

    // The login modal isn't rendered by ChatPanel itself (see App.test.tsx
    // for the full, real modal flow) — this simulates exactly what it does
    // on success: call the stashed action.
    act(() => {
      resumeSend?.();
    });

    expect(mockSendChatMessage).toHaveBeenCalledWith("Une question", []);
    expect(screen.getByText("Une question")).toBeInTheDocument();
    expect(await screen.findByText("Réponse de l'assistant.")).toBeInTheDocument();
  });
});

describe("ChatPanel — RAG sources", () => {
  it("shows a collapsible Sources section, collapsed by default, when the reply has sources", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "D'après votre bulletin de mars 2025 : la CSG est déductible.",
      sources: [
        {
          mois_annee: "03/2025",
          extrait: "La CSG déductible est assise sur le salaire brut.",
        },
      ],
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "C'est quoi la CSG ?{enter}");
    await screen.findByText("D'après votre bulletin de mars 2025 : la CSG est déductible.");

    const summary = screen.getByText("Sources (1)");
    const details = summary.closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");

    // Collapsed by default, but the content is still in the DOM (native
    // <details>, same pattern as FaqAccordion) — visible once expanded.
    expect(screen.getByText("03/2025")).toBeInTheDocument();
    expect(
      screen.getByText("La CSG déductible est assise sur le salaire brut.")
    ).toBeInTheDocument();
  });

  it("shows one entry per source, each with its own month and extract", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "Réponse combinant deux bulletins.",
      sources: [
        { mois_annee: "01/2025", extrait: "Premier extrait." },
        { mois_annee: "02/2025", extrait: "Second extrait." },
      ],
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Compare deux bulletins{enter}");
    await screen.findByText("Réponse combinant deux bulletins.");

    expect(screen.getByText("Sources (2)")).toBeInTheDocument();
    expect(screen.getByText("01/2025")).toBeInTheDocument();
    expect(screen.getByText("Premier extrait.")).toBeInTheDocument();
    expect(screen.getByText("02/2025")).toBeInTheDocument();
    expect(screen.getByText("Second extrait.")).toBeInTheDocument();
  });

  it("does not show a Sources section when sources is null", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "Le total du net à payer est de 2300.0 euros.",
      sources: null,
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Quel est le total ?{enter}");
    await screen.findByText("Le total du net à payer est de 2300.0 euros.");

    expect(screen.queryByText(/^Sources \(/)).not.toBeInTheDocument();
  });

  it("does not show a Sources section for the user's own message", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "D'après votre bulletin de mars 2025 : ...",
      sources: [{ mois_annee: "03/2025", extrait: "Un extrait." }],
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "C'est quoi la CSG ?{enter}");
    await screen.findByText("D'après votre bulletin de mars 2025 : ...");

    // Exactly one Sources section, not one per message.
    expect(screen.getAllByText("Sources (1)")).toHaveLength(1);
  });
});

describe("ChatPanel — conversation history", () => {
  it("sends the accumulated messages as history on a follow-up question", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "En janvier, le total est de 90.0 euros.",
      sources: null,
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Quel est le total en janvier ?{enter}");
    await screen.findByText("En janvier, le total est de 90.0 euros.");

    // First message of the conversation: no history yet.
    expect(mockSendChatMessage).toHaveBeenNthCalledWith(1, "Quel est le total en janvier ?", []);

    mockSendChatMessage.mockResolvedValueOnce({
      reply: "En février, le total est de 100.0 euros.",
      sources: null,
    });
    await user.type(input, "Et pour février ?{enter}");
    await screen.findByText("En février, le total est de 100.0 euros.");

    // Follow-up: the whole exchange so far, oldest first, not including
    // the new message itself (sent separately as the first argument).
    expect(mockSendChatMessage).toHaveBeenNthCalledWith(2, "Et pour février ?", [
      { role: "user", content: "Quel est le total en janvier ?" },
      { role: "assistant", content: "En janvier, le total est de 90.0 euros." },
    ]);
  });

  it("does not include a past message's sources in the history payload", async () => {
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "D'après votre bulletin de mars 2025 : la CSG est déductible.",
      sources: [{ mois_annee: "03/2025", extrait: "La CSG déductible est assise sur..." }],
    });
    render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "C'est quoi la CSG ?{enter}");
    await screen.findByText("D'après votre bulletin de mars 2025 : la CSG est déductible.");

    mockSendChatMessage.mockResolvedValueOnce({ reply: "Autre chose ?", sources: null });
    await user.type(input, "Merci{enter}");
    await screen.findByText("Autre chose ?");

    // Only role/content is sent back, never the sources from that earlier
    // assistant message.
    expect(mockSendChatMessage).toHaveBeenNthCalledWith(2, "Merci", [
      { role: "user", content: "C'est quoi la CSG ?" },
      {
        role: "assistant",
        content: "D'après votre bulletin de mars 2025 : la CSG est déductible.",
      },
    ]);
  });

  it("resets the conversation on a fresh mount, simulating a page reload", async () => {
    // ChatPanel's history lives only in React state (`useState`), never in
    // browser storage or a server session — unmounting and remounting the
    // component (what a full page reload does) is enough to confirm it
    // starts empty again, without needing to reload jsdom's window itself.
    const user = userEvent.setup();
    mockSendChatMessage.mockResolvedValueOnce({
      reply: "En janvier, le total est de 90.0 euros.",
      sources: null,
    });
    const { unmount } = render(<ChatPanel />);

    const input = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(input, "Quel est le total en janvier ?{enter}");
    await screen.findByText("En janvier, le total est de 90.0 euros.");

    unmount();
    mockSendChatMessage.mockResolvedValueOnce({ reply: "Bonjour !", sources: null });
    render(<ChatPanel />);

    expect(screen.getByText("Suggestions")).toBeInTheDocument();
    expect(screen.queryByText("Quel est le total en janvier ?")).not.toBeInTheDocument();

    const freshInput = screen.getByPlaceholderText("Posez une question sur vos bulletins…");
    await user.type(freshInput, "bonjour{enter}");
    await screen.findByText("Bonjour !");

    // No leftover history from the unmounted instance.
    expect(mockSendChatMessage).toHaveBeenNthCalledWith(2, "bonjour", []);
  });
});
