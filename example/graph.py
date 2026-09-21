from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: int


def increment(state: State) -> State:
    return {"value": state.get("value", 0) + 1}


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)
graph = builder.compile()
