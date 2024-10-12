import pprint
from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from tools import tools, retriever_tool


### 边缘功能

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    确定检索到的文档是否与问题相关。

    参数:
        state (messages): 当前状态

    返回:
        str: 一个决定文档是否相关的字符串
    """

    print("---检查相关性---")

    # 数据模型
    class grade(BaseModel):
        """用于相关性检查的二进制评分。"""

        binary_score: str = Field(description="相关性评分 'yes' 或 'no'")

    # LLM设置
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

    # 带工具和验证的LLM
    llm_with_tool = model.with_structured_output(grade)

    # 提示模板
    prompt = PromptTemplate(
        template="""你是一个评分员，评估检索到的文档与用户问题的相关性。 \n 
        这是检索到的文档: \n\n {context} \n\n
        这是用户的问题: {question} \n
        如果文档包含与用户问题相关的关键词或语义意义，请将其评为相关。 \n
        给出一个二进制评分 'yes' 或 'no' 以指示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    # 链
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    print("评分员的回答：", scored_result)
    score = scored_result.binary_score

    if score == "yes":
        print("---决定：文档相关---")
        return "generate"

    else:
        print("---决定：文档不相关---")
        print(score)
        return "rewrite"


### 节点功能

def agent(state):
    """
    调用代理模型根据当前状态生成响应。给定问题，它将决定使用检索工具或简单结束。

    参数:
        state (messages): 当前状态

    返回:
        dict: 附加了代理响应的更新状态
    """
    print("---调用代理---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # 我们返回一个列表，因为这将添加到现有列表中
    return {"messages": [response]}


def rewrite(state):
    """
    转换查询以生成更好的问题。

    参数:
        state (messages): 当前状态

    返回:
        dict: 附加了重新措辞问题的更新状态
    """

    print("---转换查询---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    查看输入并尝试推理其潜在语义意图/意义。 \n 
    这是初始问题:
    \n ------- \n
    {question} 
    \n ------- \n
    制定一个改进的问题: """,
        )
    ]

    # 评分员
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    生成答案

    参数:
        state (messages): 当前状态

    返回:
         dict: 附加了重新措辞问题的更新状态
    """
    print("---生成---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # 提示
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # 后处理
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 链
    rag_chain = prompt | llm | StrOutputParser()

    # 运行
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()


inputs = {
    "messages": [
        ("user", "贾母等又按品大妆为了什么"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)

