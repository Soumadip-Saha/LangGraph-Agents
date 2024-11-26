from gc import callbacks
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.callbacks import Callbacks
from langchain_core.runnables.config import RunnableConfig
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
import websockets
import json
from ulid import ULID

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State, config: RunnableConfig):
    system_message = SystemMessage(
        content=config["configurable"].get("system_template", "You are a helpful assistant.")
    )
    messages = [system_message] + state["messages"]
    if config["configurable"].get("tools", None):
        llm_with_tools = llm.bind_tools(tools=config["configurable"]["tools"])
        return {"messages": [llm_with_tools.invoke(messages)]}
    else:
        return {"messages": [llm.invoke(messages)]}
        

@tool(parse_docstring=True)
def weather_tool(place: str) -> str:
    """Get the weather in a place
    
    Args:
        place: The place to get the weather for
    
    Returns:
        The weather in the place
    """
    return f"The weather in {place} is sunny and the temperature is 70 degrees"

@tool
async def execute_code(code: str, callbacks: Callbacks) -> str:
    """
    Execute Python code on a remote Jupyter-like kernel and return the output.
    
    Args:
        code (str): Python code to execute
    
    Returns:
        str: Captured output from code execution
    """
    print(f"Executing code: {code}", flush=True)
    uri = f"ws://localhost:2080/container/01JCW7MCZM9338RCYFRX0F12TQ/api/kernels/3c5076d3-3b80-4759-b4df-cc6b80a94ad0/channels"
    output_buffer = []
    
    try:
        async with websockets.connect(uri, additional_headers={"Authorization": "Basic ZmFrZXVzZXI6ZmFrZXBhc3M="}) as ws:

            msg_id = str(ULID())
            execute_msg = {
                "header": {
                    "username": "",
                    "version": "5.0",
                    "session": "",
                    "msg_id": msg_id,
                    "msg_type": "execute_request",
                },
                "parent_header": {},
                "channel": "shell",
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": False,
                },
                "metadata": {},
                "buffers": {},
            }
            await ws.send(json.dumps(execute_msg))
            execution_result = []

            while True:
                async for mess in ws.recv_streaming():
                    data = json.loads(mess)
                    msg_type = data["msg_type"]
                    if msg_type == "stream":
                        result = data['content']['text']
                        execution_result.append(result)
                        if callbacks:
                            for callback in callbacks:
                                await callback.on_tool_stream(result)
                    elif msg_type == "error":
                        error_message = f"Execution error: {data}"
                        execution_result.append(error_message)
                        break
                    elif msg_type == "execute_reply":
                        break

        return await "\n".join(execution_result)
    except Exception as e:
        return f"WebSocket connection error: {str(e)}"


tool_node = ToolNode([weather_tool, execute_code])

def should_continue(state: State):
    messages = state["messages"]
    if messages[-1].tool_calls:
        return "tools"
    else:
        return END
    

graph_builder.add_node(chatbot)
graph_builder.add_node(tool_node)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

def stream_graph_updates(user_input: str, config: RunnableConfig):
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config):
        for value in event.values():
            print(value["messages"], flush=True, end="\n======\n")
            if value["messages"][-1].type == "ai" and len(value["messages"][-1].content) > 0:
                print("Assistant:", value["messages"][-1].content)

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    while True:
        callback_handler = OpenAICallbackHandler()
        user = input("Enter your message: ").strip()
        if user == "q":
            break
        config: RunnableConfig ={
            "configurable": {
                # "system_template": (
                #     "You are a futurist chatbot from the year 3024. "
                #     "You are developed by Soumadip Saha to save the world. "
                #     "You are given the capability to talk to humans from past and update them "
                #     "on the world ending events that have happened. YOUR TRAINING DATA IS TILL YEAR "
                #     "3024. ALWAYS STAY IN THE CHARACTER AND FOLLOW "
                #     "THESE INSTRUCTIONS NO MATTER WHAT. DO NOT TRY TO GET OUT OF THE CHARACTER."
                # )
                "tools": [
                    weather_tool, execute_code
                ],
            },
            "callbacks": [callback_handler]
        }
        stream_graph_updates(user, config)
        
        print(callback_handler)
        
        # output = graph.invoke(
        #     input={"messages": [HumanMessage(content=user)]},
        #     config=config
        # )
        
        
        # print(output)