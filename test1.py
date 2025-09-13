from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    message: str

def Compliment(state: AgentState) -> AgentState:
    ''' Node that adds compliment '''

    state['message'] = 'Hey ' + state['message'] + ', you\'re doing an amazing job learning LangGraph!'

    return state

graph = StateGraph(AgentState)

graph.add_node("compliment", Compliment)

graph.set_entry_point("compliment")
graph.set_finish_point("compliment")

app = graph.compile()

result = app.invoke({"message": "Aryan"})

print(result['message'])