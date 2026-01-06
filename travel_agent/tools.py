from common.available_tools import ToolExecutor

from .get_attraction import get_attraction
from .get_weather import get_weather

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

tool_executor = ToolExecutor(available_tools)

