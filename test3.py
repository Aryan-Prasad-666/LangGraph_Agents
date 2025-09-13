from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    message: str

def first_node(state: AgentState) -> AgentState:
    """ First node of the sequence """

    state['message']  = f"Hey {state['name']}!"

    return state

def second_node(state: AgentState) -> AgentState:
    """ Second node of the sequence """

    state['message'] += f" Your age is {state['age']}."

    return state

def third_node(state: AgentState) -> AgentState:
    """ Third node of the sequence """

    state['message'] += f" Your Skills are: {', '.join(state['skills'])}"

    return state

graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)

graph.add_node("second_node", second_node)

graph.add_node("third_node", third_node)

graph.set_entry_point('first_node')
graph.set_finish_point('third_node')

graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")

app = graph.compile()

result = app.invoke({"name": "Aryan", "age": "21","skills":["C", "Java", "Web Development"] })

print(result)







