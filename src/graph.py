from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


@tool(parse_docstring=True)
def get_weather(location: str) -> str:
    """This tool returns the weather for a given location.

    Args:
        location: The location to get the weather for.

    Returns:
        str: The weather for the location.
    """
    return f"The weather in {location} is sunny and temperature is 25 degrees."

@tool(parse_docstring=True)
def calculate(equation: str) -> float:
    """This tool returns the result of a given equation.

    Args:
        equation: The equation to calculate written for python eval.

    Returns:
        float: The result of the equation.
    """
    return eval(equation)

tools = [get_weather, calculate]
tool_node = ToolNode(tools=tools)

llm_with_tools = llm.bind_tools(tools=tools)

def agent(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(input=messages)
    return {'messages': [response]}

def direct_agent(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    else:
        return END

graph_builder = StateGraph(MessagesState)

# add the nodes
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("tools", "agent")
graph_builder.add_conditional_edges(source="agent", path=direct_agent, path_map=["tools", END])

graph = graph_builder.compile()

# img = graph.get_graph().draw_mermaid_png()

# save the graph to a file
# with open("graph.png", "wb") as f:
#     f.write(img)

if __name__ == "__main__":
    while True:
        user_input = input("Enter a message: ").strip()
        if user_input.lower() == "q":
            break
        graph_stream =graph.stream(
            input={
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            stream_mode="values"
        )
        
        for chunk in graph_stream:
            for mess in chunk["messages"]:
                mess.pretty_print()
            print(">>>>>>>>>>>>>>>>>>>")
            # print(chunk)
        print("\n\n==========================================\n\n")
        