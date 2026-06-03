from typing import Literal

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace")
        return stripped

    model_config = {"str_min_length": 1}


class ChatEvent(BaseModel):
    event: Literal["token", "done", "error"]
    content: str | None = None
    tool_used: str | None = None
    finish_reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
