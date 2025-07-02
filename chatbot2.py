from typing import Annotated
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()

groq_key = os.getenv('groq_api_key')

llm = ChatGroq(
    model = "deepseek-r1-distill-llama-70b",
    api_key= groq_key,
    temperature= 0.6
)

tool = TavilySearch(max_results = 2)
tools = [tool]

llm_with_tool = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# ReAct Agent
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}


graph_builder.add_node("tool_calling_llm", tool_calling_llm)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "tool_calling_llm")
graph_builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
graph_builder.add_edge("tools", "tool_calling_llm")
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

response = graph.invoke({"messages":"tell me the recent ai news and tell me news about indian defence"})

for m in response['messages']:
    m.pretty_print()

