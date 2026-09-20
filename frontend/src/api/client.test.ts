/**
 * Unit tests for the API client. `axios` is mocked at the module boundary —
 * `axios.create()` returns a fake instance whose `get`/`post` are spies, so
 * no test in this file can reach a real network socket.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import { makePayslip } from "../test/fixtures";
import type { ApiErrorBody } from "./types";

const { mockGet, mockPost, mockRequestUse, mockResponseUse } = vi.hoisted(() => ({
  mockGet: vi.fn(),
  mockPost: vi.fn(),
  mockRequestUse: vi.fn(),
  mockResponseUse: vi.fn(),
}));

/** Minimal stand-in for the shape `axios.isAxiosError` narrows to. */
interface FakeAxiosError {
  isAxiosError: true;
  response?: { status?: number; headers?: Record<string, string>; data?: ApiErrorBody };
}

function isFakeAxiosError(value: unknown): value is FakeAxiosError {
  if (typeof value !== "object" || value === null || !("isAxiosError" in value)) {
    return false;
  }
  // The cast is guarded by the checks above: this is a hand-written
  // stand-in for axios's own `isAxiosError` type predicate (which real
  // axios implements the same way), not a shortcut around a type error.
  // `isAxiosError`'s declared type is the literal `true`, so this is
  // already a boolean — no `=== true` comparison needed.
  return (value as FakeAxiosError).isAxiosError;
}

vi.mock("axios", () => ({
  default: {
    create: vi.fn(() => ({
      get: mockGet,
      post: mockPost,
      interceptors: { request: { use: mockRequestUse }, response: { use: mockResponseUse } },
    })),
    isAxiosError: isFakeAxiosError,
  },
}));

// `vi.mock` calls are hoisted above imports by Vitest, so this static import
// already resolves against the faked axios — `api = axios.create(...)`
// inside client.ts becomes `{ get: mockGet, post: mockPost }`.
import {
  fetchPayslips,
  formatRetryDelay,
  getApiErrorMessage,
  getRateLimit,
  login,
  sendChatMessage,
  uploadPayslip,
} from "./client";

// Captured once, at import: client.ts wires the auth interceptors when the
// module loads, before any `beforeEach` could reset these spies.
const interceptorsRegisteredAtImport = {
  request: mockRequestUse.mock.calls.length,
  response: mockResponseUse.mock.calls.length,
};

beforeEach(() => {
  mockGet.mockReset();
  mockPost.mockReset();
});

describe("fetchPayslips", () => {
  it("GETs /api/payslips and returns the list", async () => {
    const payslips = [makePayslip(), makePayslip()];
    mockGet.mockResolvedValueOnce({ data: payslips });

    await expect(fetchPayslips()).resolves.toEqual(payslips);
    expect(mockGet).toHaveBeenCalledWith("/api/payslips");
  });
});

describe("uploadPayslip", () => {
  it("POSTs the file as multipart form data and returns the created payslip", async () => {
    const payslip = makePayslip({ filename: "mars.pdf" });
    mockPost.mockResolvedValueOnce({ data: payslip });
    const file = new File(["contenu"], "mars.pdf", { type: "application/pdf" });

    await expect(uploadPayslip(file)).resolves.toEqual(payslip);

    expect(mockPost).toHaveBeenCalledTimes(1);
    const [url, body, config] = mockPost.mock.calls[0] as [
      string,
      FormData,
      { headers: Record<string, string> },
    ];
    expect(url).toBe("/api/upload");
    expect(body).toBeInstanceOf(FormData);
    expect(body.get("file")).toBe(file);
    expect(config.headers["Content-Type"]).toBe("multipart/form-data");
  });
});

describe("sendChatMessage", () => {
  it("POSTs the message and returns the reply with no sources", async () => {
    mockPost.mockResolvedValueOnce({
      data: { reply: "La CSG est une cotisation sociale.", sources: null },
    });

    await expect(sendChatMessage("C'est quoi la CSG ?")).resolves.toEqual({
      reply: "La CSG est une cotisation sociale.",
      sources: null,
    });
    expect(mockPost).toHaveBeenCalledWith("/api/chat", { message: "C'est quoi la CSG ?" });
  });

  it("returns the RAG sources alongside the reply when present", async () => {
    const sources = [{ mois_annee: "03/2025", extrait: "La CSG déductible est assise sur..." }];
    mockPost.mockResolvedValueOnce({
      data: { reply: "D'après votre bulletin de mars 2025 : ...", sources },
    });

    await expect(sendChatMessage("C'est quoi la CSG ?")).resolves.toEqual({
      reply: "D'après votre bulletin de mars 2025 : ...",
      sources,
    });
  });
});

describe("getApiErrorMessage", () => {
  it("returns the detail string for a plain HTTPException error", () => {
    const error: FakeAxiosError = {
      isAxiosError: true,
      response: { data: { detail: "Seuls les fichiers PDF sont acceptés." } },
    };
    expect(getApiErrorMessage(error)).toBe("Seuls les fichiers PDF sont acceptés.");
  });

  it("returns undefined when detail is FastAPI's validation-error array", () => {
    // The array form is a payload shape, not a sentence meant for a human —
    // see the comment on ApiErrorBody in api/types.ts.
    const error: FakeAxiosError = {
      isAxiosError: true,
      response: {
        data: { detail: [{ loc: ["body", "file"], msg: "field required", type: "missing" }] },
      },
    };
    expect(getApiErrorMessage(error)).toBeUndefined();
  });

  it("returns undefined when the error has no response body (network failure)", () => {
    const error: FakeAxiosError = { isAxiosError: true };
    expect(getApiErrorMessage(error)).toBeUndefined();
  });

  it("returns undefined for a value that isn't an axios error at all", () => {
    expect(getApiErrorMessage(new Error("boom"))).toBeUndefined();
    expect(getApiErrorMessage("a plain string")).toBeUndefined();
    expect(getApiErrorMessage(null)).toBeUndefined();
  });
});

describe("auth wiring", () => {
  it("registers the auth request and response interceptors on the shared instance", () => {
    // Behaviour of those interceptors is covered in auth/attachAuth.test.ts.
    expect(interceptorsRegisteredAtImport).toEqual({ request: 1, response: 1 });
  });
});

describe("login", () => {
  it("POSTs the credentials and returns only the access token", async () => {
    mockPost.mockResolvedValueOnce({
      data: { access_token: "jwt.token.value", token_type: "bearer", expires_in: 86400 },
    });

    await expect(login({ username: "admin", password: "secret" })).resolves.toBe("jwt.token.value");
    expect(mockPost).toHaveBeenCalledWith("/api/auth/login", {
      username: "admin",
      password: "secret",
    });
  });
});

describe("getRateLimit", () => {
  function tooManyRequests(headers: Record<string, string>): FakeAxiosError {
    return { isAxiosError: true, response: { status: 429, headers } };
  }

  it("converts Retry-After seconds into whole minutes, rounding up", () => {
    expect(getRateLimit(tooManyRequests({ "retry-after": "600" }))).toEqual({
      retryAfterMinutes: 10,
    });
    expect(getRateLimit(tooManyRequests({ "retry-after": "61" }))).toEqual({
      retryAfterMinutes: 2,
    });
  });

  it("still reports the limit when Retry-After is missing or unreadable", () => {
    expect(getRateLimit(tooManyRequests({}))).toEqual({ retryAfterMinutes: null });
    expect(getRateLimit(tooManyRequests({ "retry-after": "soon" }))).toEqual({
      retryAfterMinutes: null,
    });
  });

  it("returns null for any other failure", () => {
    const serverError: FakeAxiosError = { isAxiosError: true, response: { status: 500 } };
    expect(getRateLimit(serverError)).toBeNull();
    expect(getRateLimit({ isAxiosError: true })).toBeNull();
    expect(getRateLimit(new Error("boom"))).toBeNull();
  });
});

describe("formatRetryDelay", () => {
  it("gives the delay in minutes when known, a vaguer wording otherwise", () => {
    expect(formatRetryDelay({ retryAfterMinutes: 12 })).toBe("dans 12 min");
    expect(formatRetryDelay({ retryAfterMinutes: null })).toBe("plus tard");
  });
});
