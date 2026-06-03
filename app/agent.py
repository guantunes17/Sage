import logging
from typing import AsyncGenerator

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import MODEL_NAME, OPENAI_API_KEY
from app.tools import get_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Sage, a helpful AI assistant with access to specialized tools.

ROUTING RULES — follow these strictly:
1. DIRECT RESPONSE: For general knowledge questions, factual information, explanations, opinions, or anything you can answer confidently from your training data (e.g., "Who was Albert Einstein?", "Explain quantum physics", "What is the capital of France?").
2. CALCULATOR: For ANY mathematical calculation, arithmetic, or numerical computation (e.g., "What is 128 * 46?", "Calculate the square root of 144", "What is 15% of 300?"). Always use the calculator — never do math in your head.
3. WEB SEARCH: ONLY for questions about current events, recent news, real-time data, or information that changes frequently (e.g., "What happened in the news today?", "What is the current price of Bitcoin?", "Who won the latest Champions League?").

When you use a tool, incorporate its result naturally into your response. Be concise and helpful."""


def _build_graph():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    return create_react_agent(
        model=llm,
        tools=get_tools(),
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


async def run_agent_stream(message: str) -> AsyncGenerator[tuple[str, str | None, str | None], None]:
    graph = _build_graph()
    tool_used = None

    try:
        async for event in graph.astream_events(
            {"messages": [("user", message)]},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                tool_used = event["name"]
                logger.info("Tool invoked: %s", tool_used)

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if content and isinstance(content, str):
                    yield ("token", content, None)

        if tool_used:
            logger.info("Routing decision: %s", tool_used)
        else:
            logger.info("Routing decision: direct_response")

        yield ("done", None, tool_used)

    except Exception as e:
        logger.error("Agent execution failed: %s", e)
        yield ("error", str(e), None)
