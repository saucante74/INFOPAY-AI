/**
 * Domain types, derived from the backend's OpenAPI schema rather than
 * hand-written a second time on this side.
 *
 * `schema.ts` is generated (`npm run generate:api-types`) — never edit it.
 * This file is the thin, readable layer on top: components import `Payslip`
 * from here instead of reaching into `components["schemas"][...]`, so the
 * generated file's shape stays an implementation detail.
 *
 * Because these are aliases and not copies, renaming a field in the backend's
 * `PayslipExtraction`/`Payslip` (see CLAUDE.md, "Field names") breaks the
 * frontend build on the next regeneration instead of silently rendering
 * `undefined`.
 */
import type { components } from "./schema";

/** A payslip as returned by `POST /api/upload` and `GET /api/payslips`. */
export type Payslip = components["schemas"]["Payslip"];

/** Reply envelope of `POST /api/chat`. */
export type ChatResponse = components["schemas"]["ChatResponse"];

/** One RAG source cited in a `ChatResponse` — a month and the matching extract. */
export type ChatSource = components["schemas"]["ChatSource"];

/** One already-exchanged message, sent back as conversation history. */
export type ChatHistoryMessage = components["schemas"]["ChatHistoryMessage"];

/** Body of `POST /api/auth/login`. */
export type LoginRequest = components["schemas"]["LoginRequest"];

/** Reply of `POST /api/auth/login`. */
export type TokenResponse = components["schemas"]["TokenResponse"];

/** Quota status for one rate-limited scope, as returned within `RateLimits`. */
export type RateLimitStatus = components["schemas"]["RateLimitStatus"];

/** Reply of `GET /api/rate-limits`. */
export type RateLimits = components["schemas"]["RateLimitsResponse"];

/**
 * FastAPI's error body.
 *
 * `detail` is a plain string for the `HTTPException`s our routers raise
 * (`{"detail": "Seuls les fichiers PDF sont acceptés."}`), but an array of
 * validation objects for FastAPI's own request-validation 422s. The generated
 * schema only knows about the second shape, so the union is declared here and
 * narrowed at runtime by `getApiErrorMessage` — not asserted away.
 */
export interface ApiErrorBody {
  detail?: string | components["schemas"]["ValidationError"][];
}
