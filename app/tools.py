import logging

import numexpr
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Use this tool to perform mathematical calculations. Input should be a mathematical expression like '2 + 2', '128 * 46', 'sqrt(144)', or '15 * 300 / 100'. Do NOT use this for non-mathematical questions."""
    logger.info("Calculator invoked with expression: %s", expression)
    try:
        result = numexpr.evaluate(expression.strip())
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception:
        return "Error: Could not evaluate the expression. Please provide a valid mathematical expression."


_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Use this tool to search the web for current events, recent news, real-time information, or data that may not be in your training data. Do NOT use this for general knowledge questions that you can already answer confidently (e.g., 'Who was Albert Einstein?', 'What is photosynthesis?'). Only use when the question requires up-to-date or recent information."""
    logger.info("Web search invoked with query: %s", query)
    try:
        return _ddg.run(query)
    except Exception:
        return "Search temporarily unavailable. I'll answer based on my knowledge."


def get_tools():
    return [calculator, web_search]


def get_tools_info() -> list[dict]:
    return [
        {"name": t.name, "description": t.description}
        for t in get_tools()
    ]
