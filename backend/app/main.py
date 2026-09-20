from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.auth.login_rate_limit import get_login_rate_limit
from app.auth.router import router as auth_router
from app.auth.security import get_auth_settings, require_auth
from app.db import init_db
from app.rate_limit import get_rate_limit_per_hour
from app.routers import chat, rate_limits, upload


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail at startup, not on the first request, if auth or rate limiting is
    # misconfigured — a server that starts but rejects every login is harder
    # to diagnose than one that refuses to start with a clear message.
    get_auth_settings()
    get_rate_limit_per_hour()
    get_login_rate_limit()
    init_db()
    yield


app = FastAPI(title="InfoPay AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "https://infopay-lyart.vercel.app", "https://infopay-ai.fr", "https://www.infopay-ai.fr"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
    # Browsers hide non-safelisted response headers from cross-origin JS;
    # the frontend reads this one to tell the user when to retry after a 429.
    expose_headers=["Retry-After"],
)


# Every route of these routers requires a valid JWT. Applied here rather
# than inside the routers so the business code stays unaware of auth.
_authenticated = [Depends(require_auth)]

app.include_router(auth_router)
app.include_router(upload.router, dependencies=_authenticated)
app.include_router(chat.router, dependencies=_authenticated)
app.include_router(rate_limits.router, dependencies=_authenticated)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
