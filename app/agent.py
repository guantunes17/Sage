import logging
from typing import AsyncGenerator

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.config import MODEL_NAME, OPENAI_API_KEY
from app.tools import get_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Sage, a helpful AI assistant with access to specialized tools.

ROUTING RULES — follow these strictly:
1. DIRECT RESPONSE: For general knowledge questions, factual information, explanations, opinions, or anything you can answer confidently from your training data (e.g., "Who was Albert Einstein?", "Explain quantum physics", "What is the capital of France?").
2. CALCULATOR: For ANY mathematical calculation, arithmetic, or numerical computation (e.g., "What is 128 * 46?", "Calculate the square root of 144", "What is 15% of 300?"). Always use the calculator — never do math in your head.
3. WEB SEARCH: ONLY for questions about current events, recent news, real-time data, or information that changes frequently (e.g., "What happened in the news today?", "What is the current price of Bitcoin?", "Who won the latest Champions League?").

When you use a tool, incorporate its result naturally into your response. Be concise and helpful.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def create_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    tools = get_tools()
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=False,
    )


async def run_agent_stream(message: str) -> AsyncGenerator[tuple[str, str | None, str | None], None]:
    try:
        executor = create_agent()
        result = await executor.ainvoke({"input": message})
    except Exception as e:
        logger.error("Agent execution failed: %s", e)
        yield ("error", str(e), None)
        return

    # Determine which tool was used from intermediate steps
    tool_used = None
    intermediate_steps = result.get("intermediate_steps", [])
    if intermediate_steps:
        last_action = intermediate_steps[-1][0]
        tool_used = last_action.tool

    if tool_used:
        logger.info("Routing decision: %s", tool_used)
    else:
        logger.info("Routing decision: direct_response")

    output: str = result.get("output", "")

    # Yield the full answer as a single token chunk then done
    # (token-level streaming requires streaming callbacks; this gives SSE compatibility)
    for chunk in _split_into_chunks(output):
        yield ("token", chunk, None)

    yield ("done", None, tool_used)


def _split_into_chunks(text: str, size: int = 20) -> list[str]:
    """Split text into word-boundary chunks for SSE token simulation."""
    words = text.split(" ")
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks
