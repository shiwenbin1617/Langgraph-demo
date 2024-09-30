import os
from typing import Literal
from dotenv import load_dotenv
load_dotenv()  # 加载环境变量
from pydantic import BaseModel, ConfigDict
from llm import llm_openai
from langgraph.graph import StateGraph, MessagesState
from tools import tools, tool_node

llm_with_tools = llm_openai.bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"  # 如果最后一条消息包含工具调用,继续执行工具
    return "__end__"  # 否则结束

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)  # 调用模型生成响应
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("tools", "agent")

# 编译工作流
app = workflow.compile()

for chunk in app.stream({"messages": [("human", "详细介绍一下地塞米松?")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

for chunk in app.stream({"messages": [("human", "武汉今天天气怎么样，穿什么衣服合适?")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

for chunk in app.stream({"messages": [("human", "谁在用琵琶弹奏一曲东风破")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
