import axios from "axios";

import { attachAuth } from "../auth/attachAuth";
import type {
  ApiErrorBody,
  ChatResponse,
  LoginRequest,
  Payslip,
  RateLimits,
  TokenResponse,
} from "./types";

export const api = axios.create({
  // `||`, not `??`: an env var set but left empty (`VITE_API_URL=` in a
  // .env file) is a misconfiguration, and falling back to localhost is
  // better than a `""` baseURL silently pointing requests at the frontend
  // origin. `??` would only guard against `undefined`.
  // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing -- empty string must fall back too
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

attachAuth(api);

/** Exchanges credentials for a JWT; rejects with a 401 on bad credentials. */
export async function login(credentials: LoginRequest): Promise<string> {
  const { data } = await api.post<TokenResponse>("/api/auth/login", credentials);
  return data.access_token;
}

export async function uploadPayslip(file: File): Promise<Payslip> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post<Payslip>("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function fetchPayslips(): Promise<Payslip[]> {
  const { data } = await api.get<Payslip[]>("/api/payslips");
  return data;
}

/** Rejects with a 404 (via the underlying axios error) if `id` doesn't exist. */
export async function deletePayslip(id: number): Promise<void> {
  await api.delete(`/api/payslips/${String(id)}`);
}

/** The reply text plus any RAG sources cited to produce it (`null` when
 * search_payslip_knowledge_tool wasn't used, or found nothing relevant). */
export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>("/api/chat", { message });
  return data;
}

/** The current upload/chat quotas. Requires auth — rejects with a 401 if
 * called while logged out, same as any other protected endpoint. */
export async function fetchRateLimits(): Promise<RateLimits> {
  const { data } = await api.get<RateLimits>("/api/rate-limits");
  return data;
}

/**
 * Pulls the human-readable message out of a failed request, or `undefined`
 * when there isn't one and the caller should fall back to its own wording.
 *
 * Kept here rather than in the components so that axios stays an
 * implementation detail of the API layer: `UploadZone` shouldn't need to know
 * which HTTP client the app uses to render an error string.
 */
export function getApiErrorMessage(error: unknown): string | undefined {
  // `ApiErrorBody | undefined`: axios types `response.data` as always
  // present, but an error response can genuinely have no body (or a
  // non-JSON one), so the optional chain below is real protection rather
  // than a redundant check the linter would flag.
  if (!axios.isAxiosError<ApiErrorBody | undefined>(error)) return undefined;
  const detail = error.response?.data?.detail;
  // Only the string form is a message meant for a human; FastAPI's
  // validation-error array is a payload shape, not a sentence.
  return typeof detail === "string" ? detail : undefined;
}

export interface RateLimit {
  /** From the `Retry-After` header; `null` if missing or unreadable. */
  retryAfterMinutes: number | null;
}

/** A `RateLimit` if the request failed with a 429, `null` otherwise. */
export function getRateLimit(error: unknown): RateLimit | null {
  if (!axios.isAxiosError(error) || error.response?.status !== 429) return null;
  const seconds = Number(error.response.headers["retry-after"]);
  return {
    retryAfterMinutes: Number.isFinite(seconds) && seconds > 0 ? Math.ceil(seconds / 60) : null,
  };
}

/** "dans 12 min", or a vaguer wording when the delay is unknown. */
export function formatRetryDelay({ retryAfterMinutes }: RateLimit): string {
  return retryAfterMinutes === null ? "plus tard" : `dans ${String(retryAfterMinutes)} min`;
}
