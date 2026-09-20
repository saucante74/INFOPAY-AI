import { ChevronDown, Loader2, Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { formatRetryDelay, getRateLimit, sendChatMessage } from "../api/client";
import type { ChatSource } from "../api/types";
import { requireAuth } from "../auth/authModal";
import { decrementRateLimit } from "../hooks/useRateLimits";

/**
 * Delay after which the loading indicator switches to a "still working"
 * message. Purely time-based, not tool-based: `/api/chat` is a single
 * classic HTTP request/response (no SSE), so the frontend has no way to
 * know which tool the agent is calling mid-flight — see RAPPORT.md for why
 * a differentiated per-tool indicator isn't feasible without a streaming
 * backend. This is deliberately honest about that: it reflects elapsed
 * time only, never claims to know what the backend is doing.
 */
const SLOW_REPLY_DELAY_MS = 4000;

/**
 * `as const satisfies readonly string[]`: `satisfies` checks the contract
 * without widening, so the array keeps its literal element types (useful if a
 * suggestion is ever referenced by value) while still being rejected if a
 * non-string sneaks in. A plain `: readonly string[]` annotation would widen
 * every entry to `string`; an `as` assertion would check nothing.
 */
const SUGGESTIONS = [
  "Quel est le total de mes cotisations retraite sur les 4 derniers mois ?",
  "Somme des cotisations sociales sur 6 mois",
  "Quelle est la moyenne de mon net à payer ?",
  "À quoi correspond la ligne Sécurité Sociale Déplafonnée ?",
] as const satisfies readonly string[];

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  // Only ever set on assistant messages, and only when
  // search_payslip_knowledge_tool found relevant extracts — see
  // ChatSource/ChatResult in the backend's graph.py.
  sources?: ChatSource[] | null;
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isTakingLonger, setIsTakingLonger] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (!isLoading) return;
    const timer = setTimeout(() => {
      setIsTakingLonger(true);
    }, SLOW_REPLY_DELAY_MS);
    return () => {
      clearTimeout(timer);
    };
  }, [isLoading]);

  const performSend = async (content: string): Promise<void> => {
    setMessages((prev) => [...prev, { role: "user", content }]);
    setInput("");
    setIsLoading(true);
    // Reset for this send — `isLoading` turning true re-arms the effect
    // above, but doesn't itself clear a "taking longer" flag left over
    // from a previous slow reply.
    setIsTakingLonger(false);

    try {
      const { reply, sources } = await sendChatMessage(content);
      setMessages((prev) => [...prev, { role: "assistant", content: reply, sources }]);
      // See UploadZone.tsx's identical call for why this is a safe local
      // update rather than a second network round trip.
      decrementRateLimit("chat");
    } catch (error) {
      const rateLimit = getRateLimit(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: rateLimit
            ? `Limite de questions atteinte pour cette heure. Réessayez ${formatRetryDelay(rateLimit)}.`
            : "Désolé, une erreur est survenue. Vérifiez que le serveur backend est bien lancé.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Logged in: `requireAuth` calls `performSend` immediately — identical to
  // the previous behaviour, including the input clearing and the user's
  // bubble appearing synchronously. Logged out: the shared login modal
  // opens and `performSend(content)` resumes once login succeeds, with the
  // exact text the user typed (captured by the closure) — nothing is added
  // to the chat, and the input isn't cleared, until it actually sends.
  const send = (text: string): void => {
    const content = text.trim();
    if (!content || isLoading) return;
    requireAuth(() => {
      void performSend(content);
    });
  };

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-surface-raised">
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">Assistant InfoPay</h2>
        <p className="text-xs text-ink-soft">Calculs exacts et explications sur vos bulletins</p>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="flex items-center gap-1.5 text-xs font-medium text-ink-soft">
              <Sparkles className="h-3.5 w-3.5" />
              Suggestions
            </p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => {
                  send(s);
                }}
                className="block w-full rounded-md border border-border px-3 py-2 text-left text-sm text-ink-soft transition-colors hover:border-accent hover:text-ink"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i}>
            <div className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-accent text-white"
                    : "bg-surface text-ink border border-border"
                }`}
              >
                {m.content}
              </div>
            </div>

            {m.sources && m.sources.length > 0 && (
              <div className="mt-1 flex justify-start">
                <details className="group max-w-[85%] rounded-md border border-border bg-surface px-2 py-1.5 text-xs text-ink-soft">
                  <summary className="flex cursor-pointer list-none items-center gap-1 font-medium marker:content-none">
                    <ChevronDown className="h-3 w-3 shrink-0 transition-transform duration-200 group-open:rotate-180" />
                    {`Sources (${String(m.sources.length)})`}
                  </summary>
                  <ul className="mt-2 space-y-2">
                    {m.sources.map((s, sourceIndex) => (
                      <li key={sourceIndex}>
                        <span className="inline-block rounded-full border border-border px-2 py-0.5 text-[11px] font-medium text-ink-soft">
                          {s.mois_annee}
                        </span>
                        <p className="mt-1 whitespace-pre-wrap text-ink-soft">{s.extrait}</p>
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink-soft">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              {isTakingLonger
                ? "Cela prend un peu plus de temps que d'habitude…"
                : "Analyse en cours…"}
            </div>
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex items-center gap-2 border-t border-border p-3"
      >
        <input
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
          }}
          placeholder="Posez une question sur vos bulletins…"
          className="flex-1 rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="flex h-9 w-9 items-center justify-center rounded-md bg-accent text-white disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  );
}
