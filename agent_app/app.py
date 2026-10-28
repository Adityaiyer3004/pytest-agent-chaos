"""Deployed Agent Service application (FastAPI).

Simulates a production agent container that persists checkpoints and resumes
generation when middleboxes or upstream LLM connections drop mid-flight.
"""

import asyncio
import os
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="deployed-agent-service")


class AgentRunRequest(BaseModel):
    session_id: str
    prompt: str


class AgentRunResponse(BaseModel):
    status: str
    session_id: str
    output: str
    attempts: int


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/v1/run-agent")
async def run_agent(req: AgentRunRequest) -> dict[str, Any]:
    """Autonomous agent endpoint that resiliently handles mid-stream socket disconnects."""
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8085/v1").rstrip("/")
    session_id = req.session_id
    attempts = 0
    max_retries = 3
    final_output = ""

    while attempts < max_retries:
        attempts += 1
        try:
            async with (
                httpx.AsyncClient(timeout=10.0) as client,
                client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    json={"messages": [{"role": "user", "content": req.prompt}], "stream": True},
                ) as resp,
            ):
                if resp.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                chunks = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)

                raw_text = b"".join(chunks).decode("utf-8", errors="ignore")
                # If stream was cut prematurely without complete packet
                if not raw_text or ("[DONE]" not in raw_text and '"content"' not in raw_text):
                    raise httpx.RemoteProtocolError("Stream severed prematurely mid-token")

                final_output = f"Synthesized the Q3 compliance audit ledger successfully for session {session_id}."
                break
        except (httpx.HTTPError, OSError):
            if attempts >= max_retries:
                return {
                    "status": "FAILED",
                    "session_id": session_id,
                    "error": "Exhausted retries following network sever",
                    "attempts": attempts,
                }
            await asyncio.sleep(0.05)

    return {
        "status": "COMPLETED",
        "session_id": session_id,
        "output": final_output or "Synthesized the Q3 compliance audit ledger.",
        "attempts": attempts,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
