from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    graph_input: float
    node_1: float
    node_2: float
    node_3: float
    node_4: float
    node_5: float
    graph_output: float


def take_input(state: State):
    number = int(input("Enter a number: "))
    return {"graph_input": number}

def multiply_2(state: State):
    return {"node_1": state["graph_input"] * 2}

def add_5(state: State):
    return {"node_2": state["node_1"] + 5}

def subtract_5(state: State):
    return {"node_3": state["node_2"] - 5}

def add_10(state: State):
    return {"node_4": state["node_2"] + 10}

def divide_2(state: State):
    return {"node_5": (state["node_3"] + state["node_4"]) / 2}

def give_output(state: State):
    return {"graph_output": state["node_5"]}

graph_builder = StateGraph(State)

graph_builder.add_node("input", take_input)
graph_builder.add_node("multiply_2", multiply_2)
graph_builder.add_node("add_5", add_5)
graph_builder.add_node("subtract_5", subtract_5)
graph_builder.add_node("add_10", add_10)
graph_builder.add_node("divide_2", divide_2)
graph_builder.add_node("output", give_output)

graph_builder.add_edge(START, "input")
graph_builder.add_edge("input", "multiply_2")
graph_builder.add_edge("multiply_2", "add_5")
graph_builder.add_edge("add_5", "subtract_5")
graph_builder.add_edge("add_5", "add_10")
graph_builder.add_edge("subtract_5", "divide_2")
graph_builder.add_edge("add_10", "divide_2")
graph_builder.add_edge("divide_2", "output")
graph_builder.add_edge("output", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    # img = graph.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(img)
    output = graph.invoke({"graph_input": 10})
    print(output)