"""
Invock AI Analytics — FastAPI server
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, traceback, urllib.parse

app = FastAPI(title="Invock Analytics API")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from agent import MongoAIAgent
        _agent = MongoAIAgent()
    return _agent

API_KEY = os.getenv("INVOCK_API_KEY", "change-me-now")

def check_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Bad API key")

class Query(BaseModel):
    question: str


def build_quickchart_url(results, chart_meta) -> str:
    """Build a QuickChart.io URL for inline chart images in Custom GPT."""
    if not results or not chart_meta:
        return None
    try:
        chart_type = chart_meta.get("type", "bar")
        title      = chart_meta.get("title", "Results")
        x_field    = chart_meta.get("x")
        y_field    = chart_meta.get("y")

        if chart_type in ("none", "metric") or not x_field or not y_field:
            return None

        top    = results[:12]
        labels = [str(r.get(x_field, ""))[:25] for r in top]
        values = [float(r.get(y_field, 0) or 0) for r in top]

        qc_type = {"bar":"bar","line":"line","donut":"doughnut"}.get(chart_type, "bar")

        config = {
            "type": qc_type,
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": title,
                    "data":  values,
                    "backgroundColor": [
                        "#3b82f6","#8b5cf6","#10b981","#f59e0b",
                        "#ef4444","#06b6d4","#84cc16","#f97316",
                        "#ec4899","#14b8a6","#a78bfa","#fb7185"
                    ]
                }]
            },
            "options": {
                "plugins": {
                    "title":  {"display": True, "text": title},
                    "legend": {"display": False}
                }
            }
        }
        encoded = urllib.parse.quote(json.dumps(config))
        url = f"https://quickchart.io/chart?w=600&h=350&c={encoded}"
        return url if len(url) < 4000 else None
    except Exception:
        return None


@app.post("/ask", dependencies=[Depends(check_key)])
async def ask(body: Query):
    try:
        agent = get_agent()
        resp  = agent.query(body.question)

        if "error" in resp:
            return {"answer": f"Error: {resp['error']}", "row_count": 0, "preview": [], "chart_url": None}

        results   = resp.get("results", [])
        chart_meta = resp.get("chart")

        # Build chart URL for Custom GPT image rendering
        chart_url = build_quickchart_url(results, chart_meta)

        # Sanitize results for JSON — remove any non-serializable objects
        safe_results = []
        for row in results:
            safe_row = {}
            for k, v in row.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    safe_row[k] = v
                else:
                    safe_row[k] = str(v)
            safe_results.append(safe_row)

        return {
            "answer":    resp.get("answer", "No answer."),
            "row_count": len(safe_results),
            "preview":   safe_results[:5],
            "chart_url": chart_url,
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[API ERROR] {e}\n{tb}")
        # Return error details instead of raw 500
        return {
            "answer":    f"⚠️ Internal error: {str(e)[:200]}",
            "row_count": 0,
            "preview":   [],
            "chart_url": None,
        }


@app.get("/health")
async def health():
    try:
        agent = get_agent()
        return {"ok": True, "db": agent.is_connected()}
    except Exception as e:
        return {"ok": False, "db": False, "error": str(e)}


@app.get("/version")
async def version():
    try:
        from agent import AGENT_VERSION
        return {"agent_version": AGENT_VERSION, "api": "v2"}
    except Exception as e:
        return {"agent_version": "unknown", "error": str(e)}
