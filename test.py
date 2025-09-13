from typing import TypedDict, Optional, Union, Any
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    message: str

def greeting_node(state: AgentState) -> AgentState:
    """ Simple node that adds a greeting message to the state """

    state['message'] = "Hey" + state['message'] + ", how is your day going?"

    return state


graph = StateGraph(AgentState)

graph.add_node("greeter", greeting_node)

graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile()


result = app.invoke({"message": "Aryan"})
print(result["message"])


# saving graph image
from IPython.display import Image, display
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
print("Graph saved as graph.png")
