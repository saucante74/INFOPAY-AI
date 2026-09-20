# Coding conventions

This document defines the architecture principles and language-specific best
practices for this project. It's referenced from `CLAUDE.md` so agents follow
it, and it's also here so you (the author) can point to it in an interview as
evidence of deliberate, clean-code choices — not just "it works".

Every example below is written against this project's actual code, not
generic textbook code.

---

## SOLID, applied to this codebase

### Single Responsibility — already mostly respected, keep it that way

Each service does one thing: `extraction.py` only extracts, `vectorstore.py`
only indexes/searches, `analytics.py` only computes. Routers stay thin
(parse request → call service → return). **When adding a feature, resist the
urge to put logic in the router** — add it to a service instead.

### Open/Closed — tool-based agent design is naturally open for extension

Adding a third tool to the agent (e.g. a trend/anomaly detector) requires
zero changes to `graph.py`'s control flow — just add the function to
`tools.py`'s `TOOLS` list. That's OCP in practice: the graph is closed for
modification, open for extension via new tools.

### Liskov Substitution — relevant once we introduce abstractions (see DIP below)

Not yet exercised in the current code because there's no class hierarchy.
Becomes relevant the moment you introduce the `Protocol`-based abstractions
below: any concrete implementation must be substitutable without breaking
callers.

### Interface Segregation — keep Pydantic schemas focused

`PayslipExtraction` holds exactly the 8 fields needed for extraction — resist
adding unrelated fields (e.g. UI display preferences) to it "for
convenience". A schema that serves two purposes is a sign it should be two
schemas.

### Dependency Inversion — the highest-value refactor to demonstrate in an interview

Right now, `extraction.py` and `vectorstore.py` hardcode their concrete
implementation (`ChatAnthropic`, `chromadb.PersistentClient`). This works,
but it means swapping the LLM provider or the vector DB requires editing
business logic directly.

**Recommended refactor** — define thin `Protocol` interfaces (Python's
structural typing, PEP 544 — no inheritance required, just matching method
signatures) and depend on those instead of concrete classes:

```python
# app/interfaces.py
from typing import Any, Protocol

class Extractor(Protocol):
    def extract(self, raw_text: str) -> PayslipExtraction: ...

class VectorStore(Protocol):
    def index(self, payslip_id: int, mois_annee: str, raw_text: str) -> None: ...
    def search(self, query: str, n_results: int = 3) -> list[dict[str, Any]]: ...
```

`extraction.py`'s current function becomes a class implementing `Extractor`;
`vectorstore.py`'s functions become a class implementing `VectorStore`. The
router and the agent tools depend on the `Protocol`, not on `ChatAnthropic`
or `chromadb` directly.

**Why this matters for the interview**: it's the concrete, defensible answer
to "how would you make this maintainable as it grows?" — swapping Claude for
another provider, or ChromaDB for FAISS/Pinecone, becomes a one-file change
instead of a hunt through the codebase. Don't do this speculatively for every
class — apply it specifically where you'd realistically want to swap an
implementation (LLM provider, vector store), not everywhere.

---

## Static type checking

The `Protocol`-based interfaces above are only as good as their enforcement.
`mypy --strict` is how that enforcement actually happens — a `Protocol`
nobody type-checks against is just a comment.

**Run it locally:**

```bash
cd backend
source venv/bin/activate
pip install -r requirements-dev.txt   # installs mypy on top of requirements.txt
mypy --strict app/
```

A GitHub Actions workflow (`.github/workflows/mypy.yml`) runs the same
command on every push/PR touching `backend/`, so a type error fails CI
instead of surfacing later.

**`requirements-dev.txt`, separate from `requirements.txt`.** The
`Dockerfile` installs `requirements.txt` straight into the runtime image
(`COPY requirements.txt . && pip install -r requirements.txt`). A type
checker and its stub packages have no business shipping to production, so
`requirements-dev.txt` starts with `-r requirements.txt` (single source of
truth for the real dependencies) and adds `mypy` and `pandas-stubs` on top —
only installed in dev and in CI.

**The `pydantic.mypy` plugin (`backend/mypy.ini`) is required, not
optional.** `langchain-anthropic`'s `ChatAnthropic` declares several
constructor fields with a Pydantic `alias`, e.g.
`model: str = Field(alias="model_name")`. Without the plugin, mypy
synthesises `__init__` from the *aliases* and rejects the normal
`ChatAnthropic(model=..., temperature=...)` call sites used throughout this
codebase (`Unexpected keyword argument "model"`, plus spurious "missing
argument" errors for other aliased, defaulted fields). The plugin teaches
mypy about Pydantic's own `populate_by_name` behavior, so the constructor
call sites can stay exactly as `langchain-anthropic`'s own docs write them.

```ini
# backend/mypy.ini
[mypy]
python_version = 3.12
plugins = pydantic.mypy
```

---

## Python idioms (the equivalent of PHP's `array_map`/`match()`)

Python's idiomatic style differs from PHP's in one key way: **comprehensions
are preferred over `map()`/`filter()`** for most cases — the Python community
considers comprehensions more readable than a chain of higher-order function
calls. Use `map()`/`filter()`/`functools.reduce()` only when passing an
already-named function, not a fresh lambda.

| Task | Avoid | Idiomatic Python |
|---|---|---|
| Transform a list | manual loop + `.append()` | list comprehension: `[f(x) for x in xs]` |
| Filter + transform | nested loop with `if` | `[f(x) for x in xs if cond(x)]` |
| Build a dict | loop with `d[k] = v` | dict comprehension: `{k: v(k) for k in ks}` |
| Multi-branch dispatch | long `if/elif/elif` chain | `match` statement (Python 3.10+, like PHP 8's `match()`) |
| String formatting | `.format()` / `%` | f-strings: `f"{value:.2f}"` |
| File/resource handling | manual open/close | `with` context managers |
| Optional/nullable checks | `if x is not None: y = x else: y = default` | walrus operator or `x or default` where safe |
| Path manipulation | string concatenation | `pathlib.Path` |
| Magic strings for fixed options | bare strings compared with `==` | `enum.Enum` or `Literal[...]` (already used in `analytics.py`'s `Operation`) |

### Concrete example from this codebase: `match` instead of `if/elif`

`analytics.py`'s `run_analytics_query` currently uses `if/elif/elif/else` to
dispatch on `operation`. The idiomatic Python 3.10+ equivalent — direct
parallel to PHP 8's `match()` — is:

```python
match operation:
    case "somme":
        value = series.sum()
    case "moyenne":
        value = series.mean()
    case "min":
        value = series.min()
    case "max":
        value = series.max()
    case _:
        return {"error": f"Opération inconnue: {operation}"}
```

Cleaner, and `match` supports structural pattern matching beyond simple
equality (useful if a future case needs to match on tuples/shapes, not just
scalar values).

### Other conventions to apply going forward

- **Type hints everywhere.** Already done in most of this codebase — keep it
  up. Use built-in generics (`list[str]`, `dict[str, int]`) rather than
  `typing.List`/`typing.Dict` (unnecessary since Python 3.9).
- **Pydantic/dataclasses over raw dicts** for any structured data that
  crosses a function boundary — already the pattern here, don't regress to
  passing loose dicts around except at tool I/O boundaries (which is
  reasonable, since LLM tool calls are inherently untyped JSON).
- **Custom exceptions over bare `Exception`.** E.g. `upload.py` currently
  catches a bare `Exception` from extraction — consider a
  `PayslipExtractionError` raised from `extraction.py` so the router can
  catch something specific rather than "anything went wrong".
- **`functools.lru_cache`** for any pure function called repeatedly with the
  same arguments (not yet needed here, but relevant if `_load_dataframe()` in
  `analytics.py` becomes a bottleneck).

---

## Agent tool parameters: a generic range, not one parameter per use case

`query_analytics` (`app/services/analytics.py` + `app/agent/tools.py`)
originally only took `derniers_n_mois` ("the last N months"). A real
question — "somme des prélèvements à la source en 2026" — exposed the gap:
the LLM had no way to express "a specific year" or "an arbitrary period",
and silently summed across every year instead.

**The fix generalizes instead of special-casing.** Rather than adding
`annee: int | None` for "a specific year", then later `trimestre: int |
None` for "a quarter", then a comparison parameter for "compare 2025 and
2026", the tool got exactly two new parameters: `date_debut: str | None`
and `date_fin: str | None`, both `"MM/YYYY"` (the same format `mois_annee`
already uses everywhere in this codebase — no new format to teach the
LLM). Every period a user might name — a year, a quarter, "since March",
an explicit range — reduces to a start/end pair the LLM computes itself
from the question's wording and its own sense of the current date; no new
tool parameter is needed for the next phrasing that comes up. A comparison
between two periods ("compare 2025 and 2026") is deliberately **not** a
third parameter either: the tool's docstring tells the LLM to call
`query_analytics` once per period and write the comparison itself from the
two exact results — comparison is reasoning over already-exact numbers, not
a new kind of calculation that needs its own Pandas path.

**Apply this going forward**: when a new question shape exposes a gap in
what a tool can express, prefer widening an existing generic parameter (a
range, a set, a predicate) over bolting on a parameter named after that one
use case. A parameter per use case is closed for extension in exactly the
way `analytics.py`'s `Operation` dispatch is deliberately open for it (see
"Open/Closed" above) — it works today and starts multiplying special cases
tomorrow. The trade-off is a slightly heavier tool description (the LLM
has to be told *how* to compute the generic parameter, e.g. the `date_debut`
examples in `tools.py`'s docstring) in exchange for a tool surface that
doesn't grow with every new phrasing.

---

## TypeScript (frontend)

The migration is **done** — `frontend/` is TypeScript end to end, on the
`feat/migrate-ts` branch. This section records the setup that actually
shipped, so the next person changes it deliberately rather than by accident.

### Non-negotiables

- **`"strict": true`** from day one. Migrating into a permissive config and
  tightening later is more work than starting strict.
- **No `any`**, explicit or implicit. Where the type is genuinely unknown
  (a caught error, a library value), use `unknown` and narrow with a type
  guard.
- **Type every component's props** with an explicit `interface`, never
  inferred-from-usage.
- **No `as` to silence the compiler.** If an assertion is genuinely the only
  option, comment *why* — the same rule the backend applies to its `cast()`
  calls under `mypy --strict`. The current frontend has none.

### Commands

```bash
cd frontend
npm run dev          # Vite dev server
npm run build        # tsc -b && vite build — type errors fail the build
npm run typecheck    # tsc -b on its own
npm run lint         # ESLint (flat config, type-aware)
npm run format       # Prettier --write
npm run format:check # Prettier --check, for CI
```

### tsconfig layout

Three files, the current Vite convention: `tsconfig.json` is a solution file
holding only references, `tsconfig.app.json` covers `src/` (DOM libs,
`jsx: react-jsx`), `tsconfig.node.json` covers `vite.config.ts` (Node types,
no DOM). Splitting them is what lets the app code be checked against browser
globals while the build config is checked against Node's, instead of one
config that has to be permissive enough for both.

On top of `strict`, these are enabled because `strict` does **not** turn them
on and each has caught something real:

| Option | Why |
|---|---|
| `noUncheckedIndexedAccess` | `files[0]` is `File \| undefined`, which is the truth — this is what forces `handleFile` to accept `undefined` |
| `noUnusedLocals` / `noUnusedParameters` | dead imports after a refactor |
| `noImplicitReturns` | a branch that forgets to return |
| `erasableSyntaxOnly` | keeps the source free of enums/namespaces, which a bundler-only pipeline can't strip |
| `verbatimModuleSyntax` | forces `import type` for type-only imports, so nothing type-only survives into the bundle |

`exactOptionalPropertyTypes` is deliberately **not** enabled: it mostly
distinguishes `{a: undefined}` from `{}`, which interacts badly with React's
optional props for little benefit here.

### ESLint

Flat config (`eslint.config.js`), built directly on
`@typescript-eslint/parser` + `@typescript-eslint/eslint-plugin` rather than
the `typescript-eslint` meta-package's `config()` helper — the same call the
backend makes with its hand-written LangGraph graph: keep the wiring visible
instead of behind a prebuilt abstraction.

**Type-aware linting is on** (`parserOptions.projectService: true`). This is
the part that matters: without type information the `no-unsafe-*` rules are
inert, and an `any` arriving from an untyped dependency passes silently. With
it, `strict-type-checked` + `stylistic-type-checked` catch unsafe
assignments, unsafe member access, floating promises and misused promises.

ESLint **replaced oxlint** (removed in the same commit). oxlint's two rules
are ported: `react/rules-of-hooks` → `eslint-plugin-react-hooks`,
`react/only-export-components` → `eslint-plugin-react-refresh`. Two linters
for one job means two places to configure and two answers to "is this
allowed".

`src/api/schema.ts` is in `ignores` — it's generated, not ours to lint.

### Prettier

`.prettierrc.json`: 100 columns, double quotes, semicolons, `es5` trailing
commas — chosen to match the style `src/` already used, so the migration
diff stayed readable instead of turning into a reformat of every line. No
`eslint-config-prettier`: modern `typescript-eslint` ships no formatting
rules, so there is nothing to turn off.

### Regenerating the API types

```bash
# with the backend running current code:
cd backend && source venv/bin/activate && uvicorn app.main:app --port 8000
cd frontend && npm run generate:api-types
```

`src/api/schema.ts` is generated by `openapi-typescript`; **never edit it by
hand**. `src/api/types.ts` is the thin layer on top, aliasing `Payslip`,
`ChatResponse` etc. so components import a name instead of indexing into
`components["schemas"][...]`.

**The generated types are only as good as the backend's return
annotations.** FastAPI derives a response schema from the route's return
type, so `-> Payslip` / `-> Sequence[Payslip]` on the routers is what makes
`openapi.json` carry a real `Payslip` component. Before those annotations
were added (during the `mypy --strict` pass) every response was
`"schema": {}` and generation produced nothing useful. If `schema.ts` comes
back empty-looking, check the router's return annotation first — and make
sure the backend you generated from is running the *current* code, not a
stale container.

### Patterns worth keeping

**1. Discriminated unions for multi-case UI state.** `UploadZone`'s old
`status` string plus a separate `errorMessage` became:

```typescript
type UploadState =
  | { status: "idle" }
  | { status: "uploading" }
  | { status: "error"; message: string };
```

"Error with no message" and "uploading while a stale message is still shown"
are now unrepresentable. Note there is no `success` variant: the uploaded
payslip goes straight to the parent via `onUploaded`, so a `success` state
would hold a second copy of state nobody reads. Add variants the component
actually distinguishes, not the ones the pattern suggests.

**2. Typed API client return values**, not just parameters:

```typescript
export async function fetchPayslips(): Promise<Payslip[]> {
  const { data } = await api.get<Payslip[]>("/api/payslips");
  return data;
}
```

**3. `satisfies` for constant config.** It checks without widening, so the
literal keeps its exact type:

```typescript
const COLUMNS = [
  { key: "mois_annee", label: "Mois" },
  // ...
] as const satisfies readonly { key: keyof Payslip; label: string }[];
```

Tying `key` to `keyof Payslip` means a renamed backend field breaks the build
here after the next regeneration — the frontend half of the guarantee
CLAUDE.md asks for under "Field names". A plain `: readonly {...}[]`
annotation would widen every `key` to `string` and lose that; an `as`
assertion would check nothing at all.

**4. Close the `import.meta.env` hole.** Vite's own `ImportMetaEnv` has an
`[key: string]: any` index signature, so `import.meta.env.VITE_API_URL` is
`any` — an implicit `any` arriving through a dependency rather than through
your own code. Declaring the variable in `src/vite-env.d.ts` makes the named
property win:

```typescript
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
}
```

Every new `VITE_*` variable belongs there.

**5. `void` for deliberately un-awaited handlers.** Type-aware linting flags
a floating promise in `onClick={() => send(s)}`. The fix is to mark the
intent, not to silence the rule:

```tsx
onClick={() => {
  void send(s);
}}
```

**6. The backend's error body is a union, and needs narrowing.** FastAPI
sends `{"detail": "<string>"}` for `HTTPException`s raised by our routers but
`{"detail": [ValidationError, ...]}` for its own request-validation failures.
The generated schema only knows the second shape. `getApiErrorMessage()` in
`api/client.ts` narrows with `typeof detail === "string"` and keeps axios out
of the components.

**7. `||` and `??` are not interchangeable.** `import.meta.env.VITE_API_URL
|| "http://localhost:8000"` intentionally keeps `||`: `VITE_API_URL=` in a
.env file yields `""`, which should fall back too. `??` would only guard
`undefined` and leave the app with an empty baseURL. The
`prefer-nullish-coalescing` rule is disabled on that one line with that
reason written next to it.

### Version pinning gotcha

TypeScript is pinned to `~5.9`, not the latest. `@typescript-eslint` 8.x
requires `typescript >=4.8.4 <6.1.0` and `openapi-typescript` wants `^5.x`;
installing TypeScript 7 satisfies neither and silently breaks type-aware
linting. Revisit when typescript-eslint supports 7.x.

---

## Testing (frontend)

Vitest + React Testing Library, as named in CLAUDE.md's original commands
section. Config lives in `vitest.config.ts` — a separate file from
`vite.config.ts` rather than a merged `test` key, for the same "one file, one
responsibility" reason the CI workflows are split (see below): `vite build`
shouldn't carry Vitest-only config around for no reason.

### Where tests live

Colocated with the file they cover: `UploadZone.tsx` + `UploadZone.test.tsx`
in the same directory, not a separate `__tests__/` or top-level `tests/`
tree. This is Vitest's own convention (its default `include` glob matches
files anywhere, colocated or not, but colocated is what the ecosystem
actually does and what the docs show), and it keeps a test next to what it
verifies — a rename or a move takes the test with it. Shared, non-test
utilities (`src/test/setup.ts`, `src/test/fixtures.ts`, `src/test/helpers.ts`)
live in `src/test/`, named so they don't match the `*.test.ts` glob
themselves.

### Environment: jsdom, not happy-dom

`jsdom` was chosen over the faster `happy-dom` for one concrete reason:
`UploadZone`'s drag-and-drop tests need a working `DataTransfer`/`files`
simulation via `fireEvent.drop(el, { dataTransfer: { files: [...] } })`,
and jsdom's DOM event/File API coverage is the more complete, better-tested
one of the two for exactly this kind of interaction. Revisit if test suite
startup time becomes a real problem — it isn't at 28 tests.

### Mocking strategy

- **The API client (`src/api/client.ts`) is mocked wholesale**
  (`vi.mock("../api/client")`) in every component/hook test. Components
  don't know or care that axios exists; they call `uploadPayslip()`,
  `fetchPayslips()`, `sendChatMessage()` and react to what resolves or
  rejects. No test outside `client.test.ts` itself knows the HTTP client is
  axios.
- **`client.test.ts` mocks `axios` itself** (`vi.mock("axios")`, with
  `axios.create()` returning a fake instance whose `get`/`post` are
  `vi.fn()`s created via `vi.hoisted()`). This is the one file allowed to
  know the transport is axios, since it's testing that layer specifically.
- **No test reaches a real network socket or a real backend.** Verified,
  not just intended — see RAPPORT.md's verification section for how this
  was checked.

### Gotchas found (fixed in `src/test/setup.ts` unless noted)

- **`Element.prototype.scrollTo` doesn't exist in jsdom**
  ([jsdom#1695](https://github.com/jsdom/jsdom/issues/1695)) — `ChatPanel`
  calls it to auto-scroll the message list, so every render without a stub
  throws. Stubbed as a no-op; there's no real layout to assert a scroll
  position against in jsdom anyway.
- **RTL's auto-cleanup needs a global `afterEach`.** `vitest.config.ts` sets
  `globals: false` (test files import `describe`/`it`/`expect`/`afterEach`
  explicitly, consistent with `verbatimModuleSyntax` forbidding implicit
  anything elsewhere in this codebase) — but React Testing Library's
  automatic unmount-after-each-test registration relies on detecting
  Vitest's *global* `afterEach`. With globals off, it silently never
  attaches, and DOM from one test leaks into the next. Fixed by importing
  `cleanup` and calling it in an explicit `afterEach` in the setup file.
- **Recharts never mounts under jsdom.** `ResponsiveContainer` measures its
  container with a real `ResizeObserver`
  (`typeof ResizeObserver === 'undefined'` guards the whole measurement
  path in its source) and jsdom implements no `ResizeObserver` at all — so
  `PayslipChart`'s axis, lines and tooltip never render in any test.
  Deliberately **not** polyfilled: that would test Recharts' own
  measurement/rendering pipeline, not the one conditional branch that's
  actually this codebase's code
  (`payslips.length < 2` ? placeholder : chart). `PayslipChart.test.tsx`
  asserts on that branch and on the presence/absence of the
  `.recharts-responsive-container` wrapper, nothing past it.
- **`userEvent.upload()` enforces the target `<input accept>`.** It won't
  attach a file whose type doesn't match `accept="application/pdf"` — this
  simulates a real OS file picker filtering its own list, so the component's
  own `file?.type !== "application/pdf"` guard never fires through it.
  Tests that need to exercise that guard use `fireEvent.drop(...)` instead,
  which is also the more realistic path: drag-and-drop bypasses any picker
  filtering entirely, which is exactly why the component still needs its
  own check.
- **RTL's default text normalizer breaks locale-formatted currency
  matches.** It collapses whitespace runs in the *DOM's* text to a single
  ASCII space before comparing, but does not apply that same
  normalization to a string matcher you pass in — so querying for
  `formatEuros(3000)` (whose narrow no-break space, U+202F, is real fr-FR
  formatting) silently fails to match text that's already been collapsed.
  Fixed locally in `PayslipTable.test.tsx` with `{ normalizer: (t) => t }`
  to compare both sides as the exact same bytes, rather than fighting the
  normalizer or hand-typing a Unicode space character into a test file.

---

## What NOT to over-engineer

Given the one-week timeline, apply the DIP refactor (Protocol interfaces) and
the TS migration because they're genuinely good interview talking points and
not expensive. Skip, for now: a full repository/unit-of-work pattern over
SQLModel, a dependency-injection framework, or splitting the FastAPI app into
multiple microservices — none of that is proportionate to this project's
size, and over-engineering a one-week MVP is itself a signal to avoid in an
interview ("do I know when to stop" matters as much as "do I know the
pattern").