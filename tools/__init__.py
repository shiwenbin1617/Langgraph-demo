from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict

from tools.get_medicine_details import get_medicine_details
from tools.get_retriever import retriever_tool
from tools.get_weather import get_weather

# Create the tool node with the tools
tools = [get_weather,get_medicine_details,retriever_tool]
# tools = [retriever_tool]
tool_node = ToolNode(tools)
