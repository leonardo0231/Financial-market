import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .scheduler import start_scheduler
from .workflows.market_analysis import router as market_router
from .workflows.signal_generator import router as signal_router
from .workflows.risk_manager import router as risk_router
from .workflows.trade_executor import router as exec_router
from .workflows.telegram_controller import router as telegram_router
from .security import compute_signature

app = FastAPI(title="n8n-Python Orchestrator", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins= settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers (webhook endpoints)
app.include_router(signal_router)
app.include_router(risk_router)
app.include_router(exec_router)
app.include_router(telegram_router)

@app.on_event("startup")
async def on_startup():
    # Start scheduled Market Analysis job
    start_scheduler()

@app.get("/api/health")
async def health():
    return {"status": "ok", "orchestrator": "running"}

# Utility endpoint to compute HMAC for clients (e.g., internal scheduler)
@app.post("/api/sign")
async def sign(request: Request):
    body = await request.body()
    return {"signature": compute_signature(body)}