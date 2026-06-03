import logging
import traceback

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from app.agent import run_agent_stream
from app.config import APP_NAME, APP_VERSION
from app.schemas import (
    ChatEvent,
    ChatRequest,
    ErrorResponse,
    HealthResponse,
    ToolInfo,
    ToolsResponse,
)
from app.tools import get_tools_info

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tool_names = [t["name"] for t in get_tools_info()]
    logger.info("%s is ready. Available tools: %s", APP_NAME, tool_names)
    yield


app = FastAPI(
    title="Sage",
    description="AI Assistant with Tool Calling",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info("Incoming message: %s", request.message)

    async def event_stream():
        try:
            async for event_type, content, tool_used in run_agent_stream(request.message):
                if event_type == "token":
                    event = ChatEvent(event="token", content=content)
                elif event_type == "done":
                    event = ChatEvent(
                        event="done",
                        tool_used=tool_used,
                        finish_reason="stop",
                    )
                else:
                    event = ChatEvent(event="error", content=content)
                    logger.error("Agent stream error: %s", content)

                yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
        except Exception as e:
            logger.error("Unhandled error in event stream: %s\n%s", e, traceback.format_exc())
            error_event = ChatEvent(event="error", content="An unexpected error occurred.")
            yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", app_name=APP_NAME, version=APP_VERSION)


@app.get("/tools", response_model=ToolsResponse)
async def tools():
    return ToolsResponse(
        tools=[ToolInfo(**t) for t in get_tools_info()]
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    detail = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="Validation error", detail=detail).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error").model_dump(),
    )


@app.get("/")
async def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
