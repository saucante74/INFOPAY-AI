# InfoPay AI — Payslip Analytics & Assistant

A hybrid analytics + RAG assistant for payslips: upload PDF payslips, get exact
multi-month calculations (via Pandas), and ask natural-language questions
about payslip line items (via a vector search RAG pipeline on ChromaDB).

## Backend setup

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste your ANTHROPIC_API_KEY and set up the login account
# (see "Authentication" below — the server refuses to start without it)

uvicorn app.main:app --reload --port 8000
```

Health check: `http://localhost:8000/api/health` should return `{"status": "ok"}`.
Interactive docs: `http://localhost:8000/docs` — useful to test `/api/upload`
(with a real PDF) and `/api/chat` without waiting on the frontend. Call
`POST /api/auth/login` first, then paste the `access_token` into the
**Authorize** button (top right).

> Requires Python 3.12. ChromaDB does not currently support Python 3.14
> (native dependency build failures) — see the note at the bottom.

## Authentication

The whole API (`/api/upload`, `/api/payslips`, `/api/chat`) requires a JWT,
obtained from `POST /api/auth/login` with a single account defined in
`backend/.env` — there is no users table. `/api/health` stays public.

**1. Hash the password** (never store it in clear, not even in `.env`). The
prompt hides what you type and keeps it out of your shell history:

```bash
cd backend && source venv/bin/activate
python -c "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass('Mot de passe : ').encode(), bcrypt.gensalt()).decode())"
```

bcrypt only uses the first 72 bytes of a password; longer ones are refused.

**2. Generate the token-signing secret:**

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

**3. Put both in `backend/.env`:**

```dotenv
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH='$2b$12$...paste the hash here...'
JWT_SECRET=...paste the secret here...
JWT_EXPIRE_HOURS=24        # optional, default 24
RATE_LIMIT_PER_HOUR=20     # optional, default 20 (see below)
RATE_LIMIT_WINDOW_HOURS=4  # optional, default 1 (see below)
LOGIN_RATE_LIMIT_PER_15MIN=5   # optional, default 5 (see below)
```

> ⚠️ **Keep the single quotes around the hash.** `docker compose` expands
> `$` in `env_file` values: unquoted (or double-quoted), `$2b$12$abc…`
> reaches the container as just `$2b$12`, and every login fails. The
> backend checks for this and refuses to start with an explicit message
> rather than rejecting logins silently.

The backend validates all of this **at startup**: a missing variable, a
malformed hash or a secret shorter than 32 characters stops the server with
a message naming the variable. Changing the password means generating a new
hash and restarting; changing `JWT_SECRET` or `ADMIN_USERNAME` also logs
out every open session.

**Rate limiting.** `/api/upload` and `/api/chat` call the Anthropic API and
cost money, so each is limited to `RATE_LIMIT_PER_HOUR` requests per
sliding window of `RATE_LIMIT_WINDOW_HOURS` hours (separate counters).
Past it, the API answers `429` with a `Retry-After` header and the UI says
when to retry. Counters live in memory: they reset on restart, and running
several uvicorn workers would multiply the effective limit (the Dockerfile
runs one).

**Login rate limiting.** `POST /api/auth/login` is separately limited to
`LOGIN_RATE_LIMIT_PER_15MIN` attempts per **client IP** (not a shared
counter, unlike upload/chat above) in a sliding 15-minute window — a
global counter would let one attacker lock out the real user by
exhausting it. This slows down password guessing; it isn't a full
account-lockout system, and login attempts are not IP-rate-limited by
anything else (no CAPTCHA, no backoff beyond the 429).

**Removing authentication later.** It is isolated on purpose: delete
`backend/app/auth/` and the two auth lines in `app/main.py` (the login router
and `dependencies=_authenticated`); on the frontend, delete `src/auth/` and
`src/pages/LoginPage.tsx`, then remove the `attachAuth(api)` call in
`src/api/client.ts`, the `/login` route and `<RequireAuth>` wrapper in
`App.tsx`, and the logout button in `Navbar.tsx`. Rate limiting
(`app/rate_limit.py`) is independent and can stay.

## Frontend setup

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_URL, defaults to http://localhost:8000
npm run dev
```

Open `http://localhost:5173` — you land on `/login` until you sign in with
the account from `backend/.env`. The frontend expects the backend to be
running on port 8000 (locally or via Docker).

The JWT is kept in `localStorage` and sent as `Authorization: Bearer` by an
axios interceptor (`src/auth/attachAuth.ts`); any `401` logs the user out and
redirects to `/login`. Help and the legal pages stay public.

The frontend is **TypeScript** (strict). Useful commands:

```bash
npm run build        # tsc -b && vite build — type errors fail the build
npm run typecheck    # tsc -b on its own
npm run lint         # ESLint (flat config, type-aware)
npm run format       # Prettier --write  (npm run format:check to verify)
```

`src/api/schema.ts` is **generated** from the backend's OpenAPI schema — the
`Payslip` type is never hand-written twice. Regenerate it whenever a backend
route's request/response shape changes, with the backend running:

```bash
npm run generate:api-types
```

Tailwind v4 is configured via `@theme` directly in `src/index.css` (no
separate `tailwind.config.js` — that's the new Tailwind 4 approach).

## What's already working

- Full Pydantic extraction schema (all 8 required fields)
- PDF -> LLM structured extraction pipeline (handles varied payslip layouts)
- ChromaDB vector indexing (local embeddings, no external API needed for that)
- Pandas analytics engine (sum/average/min/max over the last N months)
- 2-node LangGraph graph (agent + tools) routed via Claude tool-calling
- `/api/upload`, `/api/payslips`, `/api/chat` endpoints, behind single-account
  JWT authentication, with per-hour rate limiting on upload and chat
- React + TypeScript frontend (strict, no `any`): drag & drop upload,
  summary table, evolution chart, chat — with API types generated from the
  backend's OpenAPI schema
- Docker Compose for both services

## Docker

```bash
# From the project root
cp backend/.env.example backend/.env   # then paste your API key and set up
                                       # the account (see "Authentication")
docker compose up --build
```

If the backend container exits immediately, `docker compose logs backend`
names the missing or malformed variable — most often an unquoted
`ADMIN_PASSWORD_HASH`.

The first build takes a few minutes (langchain, langgraph, chromadb are heavy
dependencies). Data (SQLite + ChromaDB) is persisted on the host in
`backend/data/`, so it survives container rebuilds and restarts.

Backend: `http://localhost:8000/api/health`
Frontend: `http://localhost:5173`

## Running tests

Tests never call the real Anthropic API or touch real data — no real
network calls, no writes to `backend/data/`.

### Backend

The LLM, vector store and DB session are all replaced with fakes/an
in-memory SQLite DB (see `backend/tests/conftest.py`).

```bash
cd backend
source venv/bin/activate
pip install -r requirements-dev.txt   # pytest, pytest-mock, pytest-cov, mypy...

pytest tests/                                      # full suite
pytest tests/unit/test_analytics.py                # one file
pytest tests/unit/test_analytics.py::test_somme     # one test
pytest tests/ -v                                    # verbose (per-test pass/fail)
pytest tests/ --cov=app --cov-report=term-missing    # with coverage
```

Also run `mypy --strict app/` before committing.

### Frontend

The API client (`src/api/client.ts`) is mocked in every test — no test can
reach a real backend. Tests are colocated with the file they cover
(`UploadZone.tsx` + `UploadZone.test.tsx`).

```bash
cd frontend
npm install   # vitest, @testing-library/react, jsdom...

npm run test                                    # full suite, non-watch
npm run test -- src/components/UploadZone.test.tsx   # one file
npm run test:watch                              # watch mode, for local dev
npm run test:coverage                           # with coverage
```

Also run `npm run build` and `npm run lint` before committing.

Both backend and frontend suites run in CI on every push/PR touching their
respective directory — see `.github/workflows/` (`mypy.yml`, `api-tests.yml`,
`frontend-checks.yml`, `frontend-tests.yml`).

## RAG evaluation

`backend/evaluation/` measures how reliable the retrieval-augmented side of
the assistant actually is, against a fixed 36-case benchmark (24 answerable,
12 not) built from the sample payslips in `frontend/public/exemples/`.

Unlike the test suites above, **this one does call the real Anthropic API**
(~75 calls, under $1, about 5 minutes) — it is a manual measurement tool, not
part of CI. It still never touches `backend/data/`: it builds its own SQLite +
ChromaDB corpus under `backend/evaluation/.corpus/` (gitignored).

```bash
# from the project root — needs ANTHROPIC_API_KEY in backend/.env
backend/venv/bin/python3 backend/evaluation/run_benchmark.py

# force re-extraction of the 7 PDFs (7 extra API calls); otherwise the
# corpus from a previous run is reused and extraction costs nothing
backend/venv/bin/python3 backend/evaluation/run_benchmark.py --rebuild-corpus
```

Results are written to `frontend/src/data/rag_run.json` and displayed at
`/metriques` (linked from the footer as "Fiabilité"). Because the page
imports that JSON as a module, updating the published metrics means
re-running the benchmark **and rebuilding the frontend**.

The script refuses to produce metrics if any `expected_evidence` in
`benchmark.json` is no longer literally present in the text pdfplumber
extracts — so a change to the PDFs, the chunking or the extraction fails
loudly instead of silently scoring against stale expectations.

See RAPPORT.md for the measured results, the metric definitions, and why the
citation metric is adapted the way it is.

## What's left to do

1. **Test with real payslip PDFs** (varied formats if possible) to validate
   extraction — this is the most fragile part, test it first.
2. Adjust the extraction prompt (`extraction.py`) if some fields are
   misextracted on your real payslip formats.
3. The frontend bundle has a size warning (~650 kB, due to Recharts) — not
   blocking for a demo, can be optimized later with dynamic `import()`
   code-splitting.

## Notes

- Model used throughout: `claude-sonnet-4-6` (`EXTRACTION_MODEL` in
  `extraction.py`, `CHAT_MODEL` in `graph.py`) — change if needed.
- ChromaDB and langchain/langgraph are heavy dependencies (tens of MB) — first
  install can take a few minutes.
- **Python 3.14 compatibility**: ChromaDB currently fails to build on Python
  3.14 (missing wheels for a native dependency, plus an internal Pydantic v1
  compatibility bug). Use Python 3.12 for this project until upstream fixes
  land.