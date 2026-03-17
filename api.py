from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from agent import MongoAIAgent

app = FastAPI(title="Invock Analytics API")
# ↑ This title appears in your Custom GPT's OpenAPI spec

app.add_middleware(CORSMiddleware, allow_origins=["*"])

# One shared agent — DB connection reused across requests
_agent = MongoAIAgent()

API_KEY = os.getenv("INVOCK_API_KEY", "change-me-now")

def check_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Bad API key")

class Query(BaseModel):
    question: str

# ── Main endpoint — this is what Custom GPT calls ──────────────
@app.post("/ask", dependencies=[Depends(check_key)])
async def ask(body: Query):
    resp = _agent.query(body.question)
    return {
        "answer": resp.get("answer", "No answer."),
        "row_count": len(resp.get("results", [])),
        "preview": resp.get("results", [])[:5],
    }

# ── Health check (no auth needed) ──────────────────────────────
@app.get("/health")
async def health():
    return {"ok": True, "db": _agent.is_connected()}
