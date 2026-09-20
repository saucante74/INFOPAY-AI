# CLAUDE.md

This file gives Claude Code the context it needs to work effectively in this
repository. Read this before making changes.

## Project overview

InfoPay AI is a hybrid analytics + RAG assistant for French payslips
("bulletins de paie"). Users upload payslip PDFs; the app extracts structured
data, stores it for exact calculations, and answers natural-language
questions using a hybrid agent (exact Pandas calculations + vector search
explanations).

**Core design principle**: the LLM never performs arithmetic itself. It
decides which tool to call (exact calculation vs. explanatory search) and
delegates execution to deterministic code. This is the single most important
architectural decision in this codebase — preserve it in any change.

## Stack

- **Backend**: Python 3.12 (not 3.14 — see Known issues), FastAPI, SQLModel/SQLite,
  Pandas, ChromaDB, LangChain, LangGraph, Claude API (`langchain-anthropic`);
  PyJWT + bcrypt for auth
- **Frontend**: React 19 + **TypeScript** (strict) + Vite, Tailwind v4, Recharts,
  lucide-react, axios, react-router; ESLint (flat config, type-aware) + Prettier;
  Vitest + React Testing Library for tests
- **Infra**: Docker Compose (backend + frontend services)

## Commands

Backend:
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm run dev          # dev server
npm run build        # tsc -b && vite build — type errors fail the build
npm run typecheck    # tsc -b on its own
npm run lint         # ESLint (type-aware)
npm run format       # Prettier --write  (format:check for CI)
```

Regenerate the frontend's API types after any change to a backend route's
request/response shape — they are generated from the live OpenAPI schema,
not hand-written (backend must be running current code):
```bash
cd frontend && npm run generate:api-types   # -> src/api/schema.ts
```

Full stack:
```bash
docker compose up --build
```

Verify backend syntax/imports without running the server:
```bash
cd backend && python3 -m py_compile app/**/*.py
```

Backend tests (pytest, see `backend/tests/`):
```bash
cd backend && source venv/bin/activate
pytest tests/                                     # full suite
pytest tests/ --cov=app --cov-report=term-missing  # with coverage
mypy --strict app/                                 # type check
```

Frontend tests (Vitest + React Testing Library, see `frontend/src/**/*.test.{ts,tsx}`):
```bash
cd frontend
npm run test              # full suite, non-watch (CI mode)
npm run test:watch        # watch mode, for local development
npm run test:coverage     # with coverage
```

Tests are colocated with the file they cover
(`UploadZone.tsx` + `UploadZone.test.tsx` in the same directory) — see
`CONVENTIONS.md`, "Testing (frontend)", for the convention and the mocking
strategy (API client mocked at the module boundary, no real network calls).

### CI workflows (`.github/workflows/`)

One workflow per responsibility — don't add a new check by growing an
existing file's job list unless it's genuinely the same responsibility:

| File | Checks |
|---|---|
| `mypy.yml` | backend: `mypy --strict app/` |
| `api-tests.yml` | backend: `pytest` (with coverage) |
| `frontend-checks.yml` | frontend: `tsc -b && vite build`, `eslint .`, `prettier --check` |
| `frontend-tests.yml` | frontend: `npm run test` (Vitest, non-watch) |

Each backend workflow triggers on `push`/`pull_request` paths scoped to
`backend/**`; each frontend workflow, to `frontend/**` (plus the workflow
file itself, so editing a workflow re-runs it).

## Shell commands

Never use `source venv/bin/activate && <command>`. Always call the venv's 
binaries directly by relative path from the project root instead:
- `backend/venv/bin/mypy --strict backend/app/`
- `backend/venv/bin/pytest backend/tests/`
- `backend/venv/bin/python3 ...`
This avoids a `source` command that Claude Code's permission system 
cannot statically analyze and will always flag for confirmation.

## Authentication and rate limiting

A single account protects the whole API; see README.md, "Authentication",
for the password-hash command and the `.env` variables (`ADMIN_USERNAME`,
`ADMIN_PASSWORD_HASH`, `JWT_SECRET`, optional `JWT_EXPIRE_HOURS`,
`RATE_LIMIT_PER_HOUR`, optional `RATE_LIMIT_WINDOW_HOURS` (default 1,
currently set to 4 — the sliding-window duration `RATE_LIMIT_PER_HOUR`
applies over) and `LOGIN_RATE_LIMIT_PER_15MIN`). The backend
**refuses to start** if they're missing or malformed, so any command that
runs the app's lifespan (uvicorn, `generate:api-types` against a live
server) needs them set. The test suite doesn't: it never triggers the
lifespan.

Rules to respect when changing the code:

- **Auth stays isolated.** It lives in `backend/app/auth/` and
  `frontend/src/auth/` (+ `pages/LoginPage.tsx`). Business routers,
  services and components never import it. Protection is applied in
  `app/main.py` via `include_router(..., dependencies=_authenticated)`, and
  the token is attached by an axios interceptor (`attachAuth`).
- **A new router is public unless you say otherwise.** When adding one in
  `main.py`, pass `dependencies=_authenticated` unless it is deliberately
  public like `/api/health` and `/api/auth/login`.
- **Any new endpoint that calls the Anthropic API gets a rate limit**:
  `dependencies=[Depends(RateLimit("<scope>"))]` on the route, from
  `app/rate_limit.py`. The limiter is independent of auth on purpose.
- **Login rate limiting is a different mechanism, on purpose.**
  `app/auth/login_rate_limit.py`'s `LoginRateLimit` keys its counter by
  `request.client.host` (one budget per IP), unlike `RateLimit`'s single
  shared counter — a shared counter on `/api/auth/login` would let one
  attacker lock out the real user. Don't reuse `RateLimit` for anything
  that needs per-caller isolation; don't reuse `LoginRateLimit` for a
  shared budget (it would just give every caller the same key's counter
  by accident if you forgot to pass one).
- **Passwords:** `bcrypt` directly, not passlib. passlib 1.7.4 crashes
  with bcrypt >= 5, which chromadb already requires.
- **Backend tests:** the `client` fixture bypasses auth and rate limiting,
  so business tests stay about business logic. Use `auth_client` (real
  `require_auth` and limiters, test settings) to test protection itself.
  For a new rate-limited endpoint, add its limiter to both fixtures in
  `tests/conftest.py`. Note: the pinned Starlette version's `TestClient`
  can't vary `request.client` per instance, so per-IP behavior is tested
  by calling `require_login_rate_limit()` directly with a hand-built
  `Request` (see `tests/functional/test_auth.py`), not through a live
  HTTP round trip.
- **Frontend tests:** components that call the API handle `429` through
  `getRateLimit()` from `api/client.ts`. Component tests mock the client,
  so add new helpers to each `vi.mock("../api/client", ...)` factory.

## Git — absolutely no version control commands

**NEVER run any command that stages, commits, pushes, or otherwise
modifies the repository's history, index, or branch state.** This includes,
but is not limited to:

- `git add`, `git commit`, `git push`
- `git checkout`, `git switch`, `git branch`
- `git stash`, `git reset`, `git restore`
- `git merge`, `git rebase`, `git cherry-pick`
- `git tag`, `git filter-repo`, `git filter-branch`

Version control is managed manually, by the user only.

You MAY use read-only Git commands to understand the repository's current
state: `git status`, `git diff`, `git log`, `git show`, `git blame`.

Leave all changes in the working tree, unstaged. Do not ask whether to
commit — simply stop after making the requested code changes and let the
user handle staging and committing themselves.

## Conventions

See `CONVENTIONS.md` for the full architecture and language-specific style
guide (SOLID applied to this codebase, idiomatic Python patterns, TypeScript
migration standards). Key points to always respect:

- **Backend**: French docstrings/comments are acceptable in existing files
  (the author is French-speaking), but prefer English for new code unless
  told otherwise — this is being standardized project-wide.
- **The agent graph is intentionally hand-written**, not built with
  `create_react_agent` or similar prebuilt abstractions. Keep it explicit and
  legible — this is a deliberate choice for interview/demo purposes, not an
  oversight to "fix" by simplifying to a prebuilt agent.
- **Never let the LLM compute a number.** Any new capability requiring
  arithmetic (sums, averages, comparisons, trends) must go through
  `analytics.py` / Pandas, exposed as a new tool if needed — not inline LLM
  reasoning.
- **Field names**: the 8 canonical payslip fields are fixed by
  `PayslipExtraction` (`mois_annee`, `salaire_brut`, `net_imposable`,
  `net_a_payer`, `total_cotisations_salariales`, `total_cotisations_patronales`,
  `cotisations_retraite`, `prelevement_source`). Don't rename them without
  updating the schema, the DB model, `analytics.py`'s `FIELD_MAP`, and the
  frontend table/chart together.
- **Data persistence paths**: SQLite lives at `backend/data/infopay.db`,
  ChromaDB at `backend/data/chroma_data/`. Both are gitignored and mounted as
  a single Docker volume (`backend/data:/app/data`). Don't reintroduce
  separate top-level paths.
- **Markdown files** (README, this file, future docs) are written in English
  going forward.

## Git commits

Never add a "Co-Authored-By: Claude" line, "Generated with Claude Code" 
signature, or any mention of AI assistance to commit messages. Commits 
are authored solely by the user.

## Workflow

- Git: monorepo, single repo for backend + frontend + docker-compose.yml at
  the root. Branches: `develop` for day-to-day work, `main` for stable
  milestones — but given the one-person, one-week timeline, working directly
  on `main` is also acceptable; don't over-engineer branch discipline for
  this project.
- Before committing generated files (zips, build artifacts, IDE folders like
  `.idea/`), check `.gitignore` covers them. If not, extend `.gitignore`
  rather than committing and cleaning up after.
- **TypeScript migration: done.** `frontend/` is TypeScript end to end
  (branch `feat/migrate-ts`); there are no `.jsx`/`.js` files left under
  `frontend/src/`. New frontend files are `.ts`/`.tsx`, with explicit props
  interfaces and no `any` — see `CONVENTIONS.md`, "TypeScript (frontend)",
  which documents the tsconfig/ESLint choices and the patterns to follow.