from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START

class AgentState(TypedDict):
    number1: int
    operation1: str
    number2: int
    number3: int
    number4: int
    operation2: str
    finalNumber1: int
    finalNumber2: int

def add_node(state: AgentState) -> AgentState:
    """ Addition node 1 """

    state['finalNumber1'] = state['number1']+state['number2']

    return state

def sub_node(state: AgentState) -> AgentState:
    """ Subtraction node 1 """

    state['finalNumber1'] = state['number1']-state['number2']

    return state

def add_node2(state: AgentState) -> AgentState:
    """ Addition node 2 """

    state['finalNumber2'] = state['number3']+state['number4']

    return state

def sub_node2(state: AgentState) -> AgentState:
    """ Subtraction node 2 """

    state['finalNumber2'] = state['number3']-state['number4']

    return state

def decide_next_node(state: AgentState) -> AgentState:
    """ This node will decide the next phase """

    if state['operation1']=="+":
        return "addition_operation1"
    
    if state['operation1']=="-":
        return "subtraction_operation1"
    
def decide_next_node2(state: AgentState) -> AgentState:
    """ This node will decide the next phase in the 2nd layer """

    if state['operation2']=="+":
        return "addition_operation2"
    
    if state['operation2']=="-":
        return "subtraction_operation2"
    
graph = StateGraph(AgentState)

graph.add_node("add_node1", add_node)
graph.add_node("sub_node1", sub_node)
graph.add_node("add_node2", add_node2)
graph.add_node("sub_node2", sub_node2)

graph.add_node("router1", lambda state:state)
graph.add_node("router2", lambda state:state)

graph.add_edge("add_node1", "router2")
graph.add_edge("sub_node1", "router2")


graph.add_edge(START, "router1")


graph.add_conditional_edges(
    "router1",
    decide_next_node,
    {
        "addition_operation1": "add_node1",
        "subtraction_operation1": "sub_node1"
    }
)

graph.add_conditional_edges(
    "router2",
    decide_next_node2,
    {
        "addition_operation2": "add_node2",
        "subtraction_operation2": "sub_node2"
    }
)

graph.add_edge("add_node2", END)
graph.add_edge("sub_node2", END)

app = graph.compile()

from IPython.display import Image, display
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph3.png", "wb") as f:
    f.write(png_bytes)
print("Graph saved as graph.png")

    
